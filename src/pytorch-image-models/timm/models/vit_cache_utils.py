"""
ViT KV Cache utilities for DCVLA project.

This module provides caching mechanisms for Vision Transformer (ViT) models,
similar to the DynamicCache in transformers library used by VLA-Cache.

The caching strategy:
- Store K and V for each ViT block across frames
- Always recompute Q to maintain context awareness
- Support selective reuse based on patch-level stability
"""

import torch
from typing import List, Optional, Tuple


class ViTCache:
    """
    Cache for storing Key and Value tensors across ViT blocks.

    This is analogous to transformers.DynamicCache but designed for ViT's
    spatial self-attention on image patches.

    Usage:
        >>> cache = ViTCache()
        >>> # In first frame
        >>> k, v = cache.update(key_states, value_states, layer_idx=0)
        >>> # In subsequent frames
        >>> cached_k, cached_v = cache.get_kv(layer_idx=0)
    """

    def __init__(self):
        """Initialize empty cache."""
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # For API compatibility

    def __len__(self):
        """Return number of cached layers."""
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific block.

        Args:
            key_states: Tensor of shape [batch, num_heads, num_patches, head_dim]
            value_states: Tensor of shape [batch, num_heads, num_patches, head_dim]
            layer_idx: Index of the transformer block (0-indexed)

        Returns:
            Tuple of (key_states, value_states) - the same tensors passed in
        """
        # Detach to prevent gradient accumulation across frames
        key_states = key_states.detach()
        value_states = value_states.detach()

        # Update or append cache
        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Append new layer cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        return key_states, value_states

    def get_kv(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve cached K and V for a specific block.

        Args:
            layer_idx: Index of the transformer block

        Returns:
            Tuple of (cached_key, cached_value) if available, else (None, None)
        """
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        return None, None

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Get sequence length (number of patches) in the cache.

        Args:
            layer_idx: Which layer to check (default: 0)

        Returns:
            Number of patches, or 0 if cache is empty
        """
        if self.key_cache and layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx].shape[2]  # num_patches dimension
        return 0

    def reset(self):
        """Clear all cached states."""
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

    def to(self, device: torch.device):
        """
        Move all cached tensors to specified device.

        Args:
            device: Target device (e.g., 'cuda:0', 'cpu')

        Returns:
            Self for method chaining
        """
        self.key_cache = [k.to(device) for k in self.key_cache]
        self.value_cache = [v.to(device) for v in self.value_cache]
        return self

    def __repr__(self):
        return (
            f"ViTCache("
            f"num_layers={len(self.key_cache)}, "
            f"num_patches={self.get_seq_length()}, "
            f"device={self.key_cache[0].device if self.key_cache else 'empty'}"
            f")"
        )


def merge_cached_kv(
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    cached_k: torch.Tensor,
    cached_v: torch.Tensor,
    reuse_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge current and cached K, V based on reuse mask.

    This is a utility function for selective KV reuse. Patches where
    reuse_mask is True will use cached K, V; otherwise use current K, V.

    Args:
        current_k: Newly computed K, shape [B, num_heads, N, head_dim]
        current_v: Newly computed V, shape [B, num_heads, N, head_dim]
        cached_k: Cached K from previous frame, same shape
        cached_v: Cached V from previous frame, same shape
        reuse_mask: Boolean tensor [N], True = use cache, False = use current

    Returns:
        Tuple of (merged_k, merged_v)

    Example:
        >>> k_new = compute_k(x)
        >>> v_new = compute_v(x)
        >>> k, v = merge_cached_kv(k_new, v_new, k_cached, v_cached, static_mask)
    """
    # Expand mask to match K, V dimensions
    # reuse_mask: [N] -> [1, 1, N, 1]
    mask_expanded = reuse_mask.view(1, 1, -1, 1).expand_as(current_k)

    # Use torch.where for efficient selection
    merged_k = torch.where(mask_expanded, cached_k, current_k)
    merged_v = torch.where(mask_expanded, cached_v, current_v)

    return merged_k, merged_v
