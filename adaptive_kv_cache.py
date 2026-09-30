"""
Adaptive KV-Cache Quantisation
================================
Per-head bit-width assignment based on attention variance,
designed for HuggingFace Transformers models.

Usage:
    from adaptive_kv_cache import AdaptiveKVCache
    cache = AdaptiveKVCache(model, config)
    cache.capture_kv_state(input_ids)          # profile variance
    assignment = cache.compute_bit_assignment()  # assign bit-widths
    cache.quantise_and_store(kv_cache)          # quantised decode
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AdaptiveKVCacheConfig:
    """Configuration for adaptive KV-cache quantisation."""

    num_heads: int = 32
    num_layers: int = 32
    head_dim: int = 128

    # Bit-width candidates in increasing precision.
    allowed_bits: Tuple[int, ...] = (2, 3, 4, 8)

    # Tolerance hyperparameter — larger tau = more aggressive quantisation.
    tau: float = 1.0

    # Number of initial tokens for profiling.
    profile_tokens: int = 128

    # Recompute assignment every N tokens during generation.
    reassignment_period: Optional[int] = 512

    # Device / dtype for the cache store.
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


@dataclass
class HeadVarianceProfile:
    """Running variance estimate for one head."""

    # Running mean (key elements averaged).
    mean: Optional[torch.Tensor] = None
    # Running M2 (sum of squared differences).
    m2: Optional[torch.Tensor] = None
    count: int = 0

    @property
    def variance(self) -> Optional[torch.Tensor]:
        if self.count < 2:
            return None
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> Optional[torch.Tensor]:
        v = self.variance
        return v.sqrt() if v is not None else None

    def update(self, keys: torch.Tensor) -> None:
        """Online Welford update with the head's key tensor (seq_len, d)."""
        for k in keys:
            self.count += 1
            if self.mean is None:
                self.mean = k.clone()
                self.m2 = torch.zeros_like(k)
            else:
                delta = k - self.mean
                self.mean = self.mean + delta / self.count
                delta2 = k - self.mean
                self.m2 = self.m2 + delta * delta2


class AdaptiveKVCache:
    """
    Manages per-head variance profiling and adaptive bit-width assignment
    for a transformer's KV cache.
    """

    def __init__(self, config: AdaptiveKVCacheConfig):
        self.config = config
        self.profiles: Dict[Tuple[int, int], HeadVarianceProfile] = {}
        self.bit_assignment: Dict[Tuple[int, int], int] = {}
        self._profiling_complete = False
        self._generation_step = 0

    def _key(self, layer: int, head: int) -> Tuple[int, int]:
        return (layer, head)

    def capture_kv_state(self, model: nn.Module, input_ids: torch.Tensor) -> None:
        """
        Run a forward pass through `model` with `input_ids` and record
        the key tensor for every layer and head.

        The model must return a namedtuple-like object whose `.past_key_values`
        attribute holds layer-wise KV tuples.

        This method patches the model's forward hooks to intercept keys before
        the attention output — a production version would read the KV cache
        directly from the model's internal state.
        """
        profiles = {}
        handles = []

        for layer_idx, layer in enumerate(model.layers):  # type: ignore[union-attr]
            for head_idx in range(self.config.num_heads):

                def make_hook(li: int, hi: int) -> callable:
                    def hook(_, __, output):
                        # output is the attention module's output tuple;
                        # index -1 for the key projection result.
                        key_out = output[-1]  # shape (bs, seq, d)
                        if li not in profiles:
                            profiles[li] = {}
                        if hi not in profiles[li]:
                            profiles[li][hi] = HeadVarianceProfile()
                        profiles[li][hi].update(key_out[0])
                    return hook

                # In practice, register hook on the key projection submodule.
                # Here we use a placeholder — real integration depends on
                # model architecture.

        # Placeholder: run forward pass.
        with torch.no_grad():
            _ = model(input_ids)

        for h in handles:
            h.remove()

        # Flatten profiles dict into self.profiles.
        for li in profiles:
            for hi, prof in profiles[li].items():
                self.profiles[self._key(li, hi)] = prof

        self._profiling_complete = True

    def compute_bit_assignment(self) -> Dict[Tuple[int, int], int]:
        """
        Assign the coarsest bit-width to each head such that quantisation
        noise stays below tolerance relative to the head's standard deviation.
        """
        if not self._profiling_complete:
            raise RuntimeError("Call capture_kv_state before bit assignment.")

        cfg = self.config
        global_max_std = 0.0

        # First pass: find maximum standard deviation across all heads.
        for key, prof in self.profiles.items():
            std = prof.std
            if std is not None:
                global_max_std = max(global_max_std, std.max().item())

        if global_max_std == 0.0:
            raise ValueError("All heads have zero variance — check profiling data.")

        # Second pass: assign bits.
        assignment = {}
        for key, prof in self.profiles.items():
            std = prof.std
            if std is None:
                assignment[key] = max(cfg.allowed_bits)
                continue

            head_std = std.mean().item()
            best_bit = max(cfg.allowed_bits)

            for b in cfg.allowed_bits:
                scale_noise = 1.0 / (2 ** (b - 1))
                if scale_noise * head_std < cfg.tau * global_max_std:
                    best_bit = b
                    break  # first (coarsest) satisfying bit-width.

            assignment[key] = best_bit

        self.bit_assignment = assignment
        return assignment

    def quantise_and_store(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Quantise each head's K and V in the cache according to the
        per-head bit assignment. Returns the quantised cache as a list
        of (K_quant, V_quant) tuples, one per layer.

        In production this would store quantised entries in a custom buffer;
        here we demonstrate the per-head quantisation logic.
        """
        quantised = []
        for layer_idx, (k, v) in enumerate(kv_cache):
            # k/v shape: (bs, num_heads, seq_len, head_dim)
            bs, nh, seq, hd = k.shape
            k_quant = torch.zeros_like(k, dtype=self.config.dtype)
            v_quant = torch.zeros_like(v, dtype=self.config.dtype)

            for head_idx in range(nh):
                key = self._key(layer_idx, head_idx)
                bits = self.bit_assignment.get(key, 4)
                k_head = k[:, head_idx, :, :]   # (bs, seq, d)
                v_head = v[:, head_idx, :, :]

                k_quant[:, head_idx, :, :] = self._quantise_head(k_head, bits)
                v_quant[:, head_idx, :, :] = self._quantise_head(v_head, bits)

            quantised.append((k_quant, v_quant))

        return quantised

    @staticmethod
    def _quantise_head(t: torch.Tensor, bits: int) -> torch.Tensor:
        """Uniform quantisation to `bits` with per-tensor scale."""
        amax = t.abs().max()
        if amax < 1e-10:
            return t
        scale = (2 ** (bits - 1) - 1) / amax
        q = torch.round(t * scale).clamp(
            -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        )
        return q / scale  # dequantised — simulate quant round-trip

    def step(self) -> None:
        """Call after each generation step to track periodic reassignment."""
        self._generation_step += 1
        if (
            self.config.reassignment_period is not None
            and self._generation_step % self.config.reassignment_period == 0
        ):
            self._profiling_complete = False  # trigger re-profile on next batch