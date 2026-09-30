"""
Adaptive KV-Cache Quantization Core

Implements:
1. Per-head KV cache quantization with attention-score gating
2. Grouped quantization with shared scale factors
3. Streaming cache with eviction and re-quantization on overflow
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List


class QuantizationConfig:
    """Configuration for quantization parameters."""
    def __init__(self,
        bits: int = 8,
        group_size: int = 64,
        gating_threshold: float = 0.5,
        max_context: int = 4096
    ):
        self.bits = bits
        self.group_size = group_size
        self.gating_threshold = gating_threshold
        self.max_context = max_context


class PerHeadKVQuantizer:
    """
    Quantizes KV cache per attention head with attention-score gating.
    High-attention heads retain higher precision; low-attention heads are compressed.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.max_value = (1 << (config.bits - 1)) - 1
        self.min_value = -(1 << (config.bits - 1))
    
    def quantize(self, x: Tensor) -> Tensor:
        """Quantize tensor to specified bits."""
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
        scale = (x_max - x_min) / (self.max_value - self.min_value)
        quantized = torch.clamp(
            torch.round((x - x_min) / (scale + 1e-8)),
            self.min_value, self.max_value
        )
        return quantized, x_min, x_max, scale
    
    def dequantize(self, q: Tensor, x_min: Tensor, scale: Tensor) -> Tensor:
        """Dequantize back to float."""
        return (q * scale + x_min).to(q.dtype)
    
    def forward(self, k: Tensor, v: Tensor, attention_scores: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantize K and V per head with attention-score gating.
        High attention scores → less quantization error.
        """
        B, H, S, D = k.shape
        
        # Compute per-head attention scores
        if attention_scores is not None:
            # attention_scores: [B, H, S] or [B, H, S, S]
            attn_mean = attention_scores.mean(dim=-1, keepdim=True)
        else:
            attn_mean = torch.ones(B, H, 1) / S
        
        # Create head-wise scale adjustments based on attention
        high_attn_mask = attn_mean > self.config.gating_threshold
        
        # For high attention heads, use finer quantization (keep full precision)
        # For low attention heads, quantize aggressively
        output_k = k.clone()
        output_v = v.clone()
        
        for h in range(H):
            if not high_attn_mask[:, h, 0].all():
                # Quantize this head
                k_head = k[:, h, :, :]
                v_head = v[:, h, :, :]
                q_k, k_min, k_max, k_scale = self.quantize(k_head)
                q_v, v_min, v_max, v_scale = self.quantize(v_head)
                output_k[:, h, :, :] = self.dequantize(q_k, k_min, k_scale)
                output_v[:, h, :, :] = self.dequantize(q_v, v_min, v_scale)
        
        return output_k, output_v


class GroupedKVQuantizer:
    """
    Grouped quantization with shared scale factors.
    Multiple heads share the same quantization parameters.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
    
    def forward(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantize K and V in groups of config.group_size dimensions.
        """
        B, H, S, D = k.shape
        
        # Group dimensions
        groups = D // self.config.group_size
        output_k = k.clone()
        output_v = v.clone()
        
        for g in range(groups):
            start = g * self.config.group_size
            end = start + self.config.group_size
            
            k_group = k[:, :, :, start:end]
            v_group = v[:, :, :, start:end]
            
            # Shared scale for group
            g_min = torch.cat([k_group.min(dim=-1, keepdim=True).values,
                               v_group.min(dim=-1, keepdim=True).values]).min()
            g_max = torch.cat([k_group.max(dim=-1, keepdim=True).values,
                               v_group.max(dim=-1, keepdim=True).values]).max()
            
            scale = (g_max - g_min) / 255.0
            
            k_group_q = torch.clamp(
                torch.round((k_group - g_min) / (scale + 1e-8)),
                0, 255
            ).to(k.dtype)
            
            v_group_q = torch.clamp(
                torch.round((v_group - g_min) / (scale + 1e-8)),
                0, 255
            ).to(v.dtype)
            
            output_k[:, :, :, start:end] = (k_group_q * scale + g_min)
            output_v[:, :, :, start:end] = (v_group_q * scale + g_min)
        
        return output_k, output_v


class StreamingKVCache:
    """
    Streaming KV cache with eviction and re-quantization on context overflow.
    """
    
    def __init__(self, config: QuantizationConfig, n_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        self.config = config
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.k_cache: Optional[Tensor] = None
        self.v_cache: Optional[Tensor] = None
        self.position = 0
        
        # Quantizers
        self.per_head = PerHeadKVQuantizer(config)
        self.grouped = GroupedKVQuantizer(config)
    
    def update(self, k: Tensor, v: Tensor, 
               attention_scores: Optional[Tensor] = None,
               use_per_head: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Update cache with new K and V. Evict if overflow.
        Returns updated K and V tensors.
        """
        B, H, S, D = k.shape
        
        # Initialize cache
        if self.k_cache is None:
            self.k_cache = torch.zeros(B, H, self.config.max_context, D, 
                                       device=k.device, dtype=self.dtype)
            self.v_cache = torch.zeros(B, H, self.config.max_context, D,
                                       device=k.device, dtype=self.dtype)
        
        # Check if we need to evict
        if self.position + S > self.config.max_context:
            # Evict oldest - reset position and cache to fit new data
            self.k_cache = self.k_cache[:, :, :S, :]
            self.v_cache = self.v_cache[:, :, :S, :]
            self.position = 0
        
        # Write new values
        self.k_cache[:, :, self.position:self.position + S, :] = k
        self.v_cache[:, :, self.position:self.position + S, :] = v
        
        # Advance position
        self.position += S
        
        # Quantize based on settings - use all cached data
        start_pos = max(0, self.position - S)
        if use_per_head and attention_scores is not None:
            k_q, v_q = self.per_head.forward(self.k_cache[:, :, start_pos:, :],
                                     self.v_cache[:, :, start_pos:, :],
                                     attention_scores)
        else:
            k_q, v_q = self.grouped.forward(self.k_cache[:, :, start_pos:, :],
                                    self.v_cache[:, :, start_pos:, :])
        
        # Return only current context
        current_len = S
        return k_q[:, :, -current_len:, :], v_q[:, :, -current_len:, :]
    
    def get_cache(self, start: int = 0) -> Tuple[Tensor, Tensor]:
        """Get stored cache from position."""
        return self.k_cache[:, :, start:], self.v_cache[:, :, start:]
    
    def reset(self):
        """Reset cache."""
        self.k_cache = None
        self.v_cache = None
        self.position = 0
