"""Real Qwen3-8B run: per-head adaptive KV-cache bit-widths vs uniform and random allocations.

Every cached key and value (post-RoPE, exactly what a KV cache stores) is fake-quantised per token,
asymmetric min-max over each head's 128 dims (one fp16 scale + zero-point per head per token).
A "unit" is one (layer, KV head, K or V): Qwen3-8B has 36 x 8 x 2 = 576 units.

Allocations compared at the same average bits per cached element:
  uniform     every unit the same bits (only at integer budgets)
  random      random units get the high bits (same 2/4 split as `variance`; mean of 3 seeds)
  random-adjacent  the fairer random baseline: only the two neighbouring bit-widths (2/3 at 2.5, 3/4 at 3.5);
              added afterwards with `run_real.py <mode> adjacent` (appends rows, no recalibration)
  variance    the repo's original heuristic: units whose cached tensor has the highest variance get more bits
  calibrated-local  greedy over {2,3,4,8} bits on each unit's raw attention-output error (first attempt; fails)
  calibrated  the same greedy, with each unit's error rescaled into loss: per layer, the measured calibration
              NLL rise when that layer's K (or V) goes to 2 bits, shared over its heads by their error

Calibration: WikiText-2 train, 8 x 512 tokens. Evaluation: WikiText-2 test, 40 x 512 tokens.
Writes results/real.json.
"""
import heapq
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qcommon import load_model, perplexity, save, windows  # noqa: E402

V3 = "v3" in sys.argv[2:]  # v3: no 8-bit choice, 4x more calibration data for the per-layer loss probe
CHOICES = (2, 3, 4) if V3 else (2, 3, 4, 8)
BUDGETS = (2.5, 3.0, 3.5)
# keys: "token" (per-token, like values) or "channel" (KIVI-style per-channel over KEY_GROUP-token blocks)
KEY_MODE = sys.argv[1] if len(sys.argv) > 1 else "token"
KEY_GROUP = 32
ADJ_ONLY = "adjacent" in sys.argv[2:]
OUT = Path(__file__).resolve().parent / (("real.json" if KEY_MODE == "token" else f"real_{KEY_MODE}.json")
                                          .replace(".json", "_v3.json" if V3 else ".json"))

STATE = {"mode": "off", "kbits": None, "vbits": None, "err": None, "var": None}


def quant_tokens(x, bits):
    """Per-token asymmetric RTN over the last dim. x: (B, H, S, D); bits: (H,) ints, >= 16 means keep."""
    xf = x.float()
    lo, hi = xf.amin(-1, keepdim=True), xf.amax(-1, keepdim=True)
    qmax = (2.0 ** bits.to(xf.device).float() - 1).view(1, -1, 1, 1)
    scale = ((hi - lo) / qmax).clamp_min(1e-8)
    zero = torch.round(-lo / scale)
    q = torch.minimum((torch.round(xf / scale) + zero).clamp_min(0), qmax)
    out = ((q - zero) * scale).to(x.dtype)
    keep = (bits >= 16).to(x.device).view(1, -1, 1, 1)
    return torch.where(keep, x, out)


def quant_channels(x, bits, group=KEY_GROUP):
    """KIVI-style keys: asymmetric RTN per channel over blocks of `group` consecutive tokens.
    x: (B, H, S, D); bits: (H,) ints, >= 16 means keep."""
    B, H, S, D = x.shape
    pad = (-S) % group
    xf = F.pad(x.float(), (0, 0, 0, pad), mode="replicate") if pad else x.float()
    xg = xf.view(B, H, -1, group, D)
    lo, hi = xg.amin(3, keepdim=True), xg.amax(3, keepdim=True)
    qmax = (2.0 ** bits.to(xf.device).float() - 1).view(1, -1, 1, 1, 1)
    scale = ((hi - lo) / qmax).clamp_min(1e-8)
    zero = torch.round(-lo / scale)
    q = torch.minimum((torch.round(xg / scale) + zero).clamp_min(0), qmax)
    out = ((q - zero) * scale).view(B, H, -1, D)[:, :, :S].to(x.dtype)
    keep = (bits >= 16).to(x.device).view(1, -1, 1, 1)
    return torch.where(keep, x, out)


def quant_keys(x, bits):
    return quant_channels(x, bits) if KEY_MODE == "channel" else quant_tokens(x, bits)


def head_attn(q, k, v, scaling):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scaling)


@torch.no_grad()
def calibrate(module, q, k, v, scaling):
    """For each KV head and each bit choice: squared error of that head's attention output when only its
    K (or only its V) is quantised. Also the variance of each head's cached K and V."""
    l, hkv = module.layer_idx, k.shape[1]
    rep = q.shape[1] // hkv
    for h in range(hkv):
        qh = q[:, h * rep:(h + 1) * rep]
        kh, vh = k[:, h:h + 1].expand_as(qh), v[:, h:h + 1].expand_as(qh)
        ref = head_attn(qh, kh, vh, scaling).float()
        for j, b in enumerate(CHOICES):
            bb = torch.tensor([b])
            kq = quant_keys(k[:, h:h + 1], bb).expand_as(qh)
            vq = quant_tokens(v[:, h:h + 1], bb).expand_as(qh)
            STATE["err"][0, l, h, j] += (head_attn(qh, kq, vh, scaling).float() - ref).pow(2).sum().item()
            STATE["err"][1, l, h, j] += (head_attn(qh, kh, vq, scaling).float() - ref).pow(2).sum().item()
        STATE["var"][0, l, h] += k[:, h].float().var().item()
        STATE["var"][1, l, h] += v[:, h].float().var().item()


def kv_attention(module, query, key, value, attention_mask, **kw):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if STATE["mode"] == "calib":
        calibrate(module, query, key, value, kw.get("scaling"))
    elif STATE["mode"] == "quant":
        l = module.layer_idx
        key = quant_keys(key, STATE["kbits"][l])
        value = quant_tokens(value, STATE["vbits"][l])
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


def split_alloc(order, budget, n):
    """Give 4 bits to the first units in `order` and 2 bits to the rest so the mean is `budget`."""
    n_hi = round((budget - 2) / 2 * n)
    bits = torch.full((n,), 2, dtype=torch.long)
    bits[order[:n_hi]] = 4
    return bits


def adjacent_alloc(order, budget, n):
    """Random-but-fair baseline: only the two integer bit-widths around `budget` (2.5 -> 2/3, 3.5 -> 3/4),
    the first units in `order` get the higher one."""
    lo = math.floor(budget)
    n_hi = round((budget - lo) * n)
    bits = torch.full((n,), lo, dtype=torch.long)
    bits[order[:n_hi]] = lo + 1
    return bits


def greedy_alloc(err, budget):
    """err: (n_units, len(CHOICES)). Start every unit at the lowest choice; repeatedly take the upgrade with
    the largest error reduction per extra bit until the bit budget is spent."""
    n = err.shape[0]
    level = [0] * n
    spare = budget * n - CHOICES[0] * n

    def gain(u):
        j = level[u]
        return (err[u, j] - err[u, j + 1]) / (CHOICES[j + 1] - CHOICES[j])

    heap = [(-gain(u), u) for u in range(n)]
    heapq.heapify(heap)
    while heap:
        g, u = heapq.heappop(heap)
        j = level[u]
        cost = CHOICES[j + 1] - CHOICES[j]
        if cost > spare + 1e-9:
            continue
        level[u], spare = j + 1, spare - cost
        if level[u] + 1 < len(CHOICES):
            heapq.heappush(heap, (-gain(u), u))
    return torch.tensor([CHOICES[j] for j in level])


def run(model, wins, bits, L, H):
    """bits: (576,) in unit order (K units then V units, each layer-major)."""
    STATE["mode"] = "quant"
    STATE["kbits"] = bits[:L * H].view(L, H)
    STATE["vbits"] = bits[L * H:].view(L, H)
    ppl = perplexity(model, wins)
    STATE["mode"] = "off"
    return round(ppl, 4)


def kv_kb_per_token(bits, D):
    """Packed KV cache per token: bits per element + fp16 scale and zero-point (per head-vector for per-token
    tensors, per channel per KEY_GROUP tokens for per-channel keys)."""
    over = torch.full_like(bits, 32 / D, dtype=torch.float)
    if KEY_MODE == "channel":
        over[:bits.numel() // 2] = 32 / KEY_GROUP
    return round(float((bits.float() + over).sum()) * D / 8 / 1024, 2)


def main():
    from transformers import AttentionInterface
    AttentionInterface.register("kvq", kv_attention)
    model, tok = load_model()
    model.set_attn_implementation("kvq")
    cfg = model.config
    L, H, D = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    n = 2 * L * H
    calib, test = windows(tok, "train", 8), windows(tok, "test", 40)
    probe = windows(tok, "train", 32) if V3 else calib  # data for the per-layer loss probe
    if ADJ_ONLY:
        # add the random-adjacent baseline rows to an existing results file (no recalibration needed)
        import json
        res = json.loads(OUT.read_text())
        res["rows"] = [r for r in res["rows"] if r["method"] != "random-adjacent"]
        half = n // 2
        for budget in (2.5, 3.5):
            for seed in range(3):
                g = torch.Generator().manual_seed(seed)
                bits = torch.empty(n, dtype=torch.long)
                for s in (0, half):
                    bits[s:s + half] = adjacent_alloc(torch.randperm(half, generator=g), budget, half)
                row = {"method": "random-adjacent", "budget": budget, "mean_bits": round(float(bits.float().mean()), 3),
                       "ppl": run(model, test, bits, L, H), "kv_kb_per_token": kv_kb_per_token(bits, D),
                       "k_mean_bits": budget, "v_mean_bits": budget, "seed": seed}
                print(row, flush=True)
                res["rows"].append(row)
                save(OUT, res)
        return
    res = {"model": "Qwen3-8B", "key_mode": KEY_MODE, "key_group": KEY_GROUP, "units": n, "eval": "wikitext-2 test 40x512", "calib": "wikitext-2 train 8x512",
           "bf16_kv_kb_per_token": round(n * D * 16 / 8 / 1024, 2)}

    STATE["mode"] = "off"
    res["bf16"] = round(perplexity(model, test), 4)
    print("bf16", res["bf16"], flush=True)

    t = time.time()
    STATE["err"] = torch.zeros(2, L, H, len(CHOICES), dtype=torch.float64)
    STATE["var"] = torch.zeros(2, L, H, dtype=torch.float64)
    STATE["mode"] = "calib"
    with torch.no_grad():
        for w in calib:
            model(w.unsqueeze(0).cuda())
    STATE["mode"] = "off"
    res["calib_seconds"] = round(time.time() - t, 1)
    err = STATE["err"].reshape(n, len(CHOICES))
    var = STATE["var"].reshape(n)
    res["calib_err_mean"] = {str(b): float(err[:, j].mean()) for j, b in enumerate(CHOICES)}

    # Raw attention-output error is not comparable across layers (late layers have larger activations, but
    # early-layer error propagates further). Convert it to loss: measure the calibration NLL rise when one
    # layer's K (or V) is all 2-bit, and spread that rise over the layer's heads in proportion to their error.
    t = time.time()
    nll0 = math.log(perplexity(model, probe))
    dloss = torch.zeros(2, L)
    for kind in range(2):
        for l in range(L):
            bits = torch.full((2, L, H), 16)
            bits[kind, l] = CHOICES[0]
            STATE["mode"], STATE["kbits"], STATE["vbits"] = "quant", bits[0], bits[1]
            dloss[kind, l] = math.log(perplexity(model, probe)) - nll0
            STATE["mode"] = "off"
    res["sens_seconds"] = round(time.time() - t, 1)
    res["layer_dloss_2bit"] = {"K": dloss[0].tolist(), "V": dloss[1].tolist()}
    e = STATE["err"]
    w = dloss.clamp_min(1e-6).double() / e[..., 0].sum(-1)  # (2, L): loss per unit of local error
    err_loss = (e * w[:, :, None, None]).reshape(n, len(CHOICES))
    print("layer dloss K", [round(x, 3) for x in dloss[0].tolist()], flush=True)
    print("layer dloss V", [round(x, 3) for x in dloss[1].tolist()], flush=True)

    rows = []

    def add(name, budget, bits, seed=None):
        ppl = run(model, test, bits, L, H)
        row = {"method": name, "budget": budget, "mean_bits": round(float(bits.float().mean()), 3), "ppl": ppl,
               "kv_kb_per_token": kv_kb_per_token(bits, D),
               "k_mean_bits": round(float(bits[:n // 2].float().mean()), 3),
               "v_mean_bits": round(float(bits[n // 2:].float().mean()), 3)}
        if seed is not None:
            row["seed"] = seed
        rows.append(row)
        print(row, flush=True)
        save(OUT, {**res, "rows": rows})
        return bits

    if V3:
        for budget in BUDGETS:
            cal = add("calibrated-v3", budget, greedy_alloc(err_loss.numpy(), budget))
            res.setdefault("calibrated_alloc", {})[str(budget)] = cal.tolist()
        save(OUT, {**res, "rows": rows})
        return
    for b in (2, 3, 4, 8):
        add("uniform", float(b), torch.full((n,), b))
    for budget in BUDGETS:
        # variance heuristic: rank K units and V units separately by variance, same split inside each
        half = n // 2
        bits = torch.empty(n, dtype=torch.long)
        for s in (0, half):
            order = torch.argsort(var[s:s + half], descending=True)
            bits[s:s + half] = split_alloc(order, budget, half)
        add("variance", budget, bits)
        for seed in range(3):
            g = torch.Generator().manual_seed(seed)
            bits = torch.empty(n, dtype=torch.long)
            for s in (0, half):
                bits[s:s + half] = split_alloc(torch.randperm(half, generator=g), budget, half)
            add("random", budget, bits, seed)
        cal = add("calibrated-local", budget, greedy_alloc(err.numpy(), budget))
        res.setdefault("calibrated_local_alloc", {})[str(budget)] = cal.tolist()
        cal = add("calibrated", budget, greedy_alloc(err_loss.numpy(), budget))
        res.setdefault("calibrated_alloc", {})[str(budget)] = cal.tolist()
    save(OUT, {**res, "rows": rows})


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
