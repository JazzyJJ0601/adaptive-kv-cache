import torch
from typing import Dict, List, Optional

class ScoringGate:
    """Lightweight heuristic assigning bit-width per KV entry based on attention."""
    def __init__(self, high_bits: int = 16, low_bits: int = 4):
        self.high_bits = high_bits
        self.low_bits = low_bits

    def compute(self, attention_scores: torch.Tensor) -> List[int]:
        """Return per-token bit-widths based on softmax attention weights."""
        threshold = attention_scores.mean()
        return [self.high_bits if s > threshold else self.low_bits for s in attention_scores]

class CompressedCache:
    """Stores KV entries in packed format, unpacks on read."""
    def __init__(self):
        self._kv: Dict[str, torch.Tensor] = {}
        self._bits: Dict[str, List[int]] = {}

    def store(self, key: str, value: torch.Tensor, bits: List[int]):
        self._bits[key] = bits
        self._kv[key] = value

    def load(self, key: str) -> Optional[torch.Tensor]:
        return self._kv.get(key)

class CacheStore:
    """Manages KV entries with per-slot bit-width."""
    def __init__(self, gate: ScoringGate):
        self.gate = gate
        self.cache = CompressedCache()

    def process(self, key: str, value: torch.Tensor, attention_scores: torch.Tensor):
        bits = self.gate.compute(attention_scores)
        self.cache.store(key, value, bits)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self.cache.load(key)
