"""Bucketed CUDA Graph runtime for OpenVLA LLM prefill.

This module keeps the experimental graph backend out of the main Prismatic
model file. It is intentionally gated by config flags and only targets the
multimodal prefill step; decode continues to use the full physical KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class BucketSpec:
    query_positions: torch.Tensor
    kv_positions: torch.Tensor
    cache_position: torch.Tensor
    position_ids: torch.Tensor
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


class BucketCache:
    """Minimal cache object used inside the graph-captured decoder."""

    def __init__(self, key_cache: List[torch.Tensor], value_cache: List[torch.Tensor]) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs["cache_position"] if cache_kwargs is not None else None
        key_out = self.key_cache[layer_idx]
        value_out = self.value_cache[layer_idx]
        if cache_position is None:
            key_out.copy_(key_states.to(key_out.dtype))
            value_out.copy_(value_states.to(value_out.dtype))
        else:
            key_out.index_copy_(2, cache_position, key_states.to(key_out.dtype))
            value_out.index_copy_(2, cache_position, value_states.to(value_out.dtype))
        return key_out, value_out


def _empty_long(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.long, device=device)


def _to_positions(value: Any, device: torch.device, total_len: int) -> torch.Tensor:
    if value is None:
        return _empty_long(device)
    if torch.is_tensor(value):
        positions = value.to(device=device, dtype=torch.long).flatten()
    else:
        positions = torch.tensor(value, dtype=torch.long, device=device).flatten()
    if positions.numel() == 0:
        return positions
    positions = positions[(positions >= 0) & (positions < total_len)]
    return torch.unique(positions, sorted=True)


def _build_position_mask(
    query_positions: torch.Tensor,
    kv_positions: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    min_dtype = torch.finfo(dtype).min
    allowed = kv_positions[None, :] <= query_positions[:, None]
    mask = torch.zeros(
        query_positions.numel(),
        kv_positions.numel(),
        dtype=dtype,
        device=query_positions.device,
    )
    mask = mask.masked_fill(~allowed, min_dtype)
    return mask[None, None, :, :].contiguous()


def _copy_cache_(dst: BucketCache, src: BucketCache) -> None:
    for dst_key, src_key in zip(dst.key_cache, src.key_cache):
        dst_key.copy_(src_key)
    for dst_value, src_value in zip(dst.value_cache, src.value_cache):
        dst_value.copy_(src_value)


def _make_empty_like_cache(cache: BucketCache) -> BucketCache:
    return BucketCache(
        [torch.empty_like(tensor) for tensor in cache.key_cache],
        [torch.empty_like(tensor) for tensor in cache.value_cache],
    )


def _clone_bucket_cache(full_cache: Any, kv_positions: torch.Tensor) -> BucketCache:
    keys = []
    values = []
    for key_states, value_states in zip(full_cache.key_cache, full_cache.value_cache):
        keys.append(key_states.index_select(2, kv_positions).clone())
        values.append(value_states.index_select(2, kv_positions).clone())
    return BucketCache(keys, values)


def _build_bucket_spec(
    hidden_states: torch.Tensor,
    reusable_patches: Any,
    deleted_patches: Any,
    cache_effective: bool,
    prune_effective: bool,
) -> BucketSpec:
    total_len = int(hidden_states.shape[1])
    device = hidden_states.device
    full_positions = torch.arange(total_len, dtype=torch.long, device=device)

    if not cache_effective:
        reusable = _empty_long(device)
        pruned = _empty_long(device)
    else:
        reusable = _to_positions(reusable_patches, device, total_len)
        pruned = _to_positions(deleted_patches, device, total_len)
        if not prune_effective:
            pruned = _empty_long(device)

    skip_positions = torch.unique(torch.cat([reusable, pruned], dim=0), sorted=True)
    query_visible = torch.ones(total_len, dtype=torch.bool, device=device)
    if skip_positions.numel() > 0:
        query_visible[skip_positions] = False
    kv_visible = torch.ones(total_len, dtype=torch.bool, device=device)
    if pruned.numel() > 0:
        kv_visible[pruned] = False

    query_positions = full_positions[query_visible]
    kv_positions = full_positions[kv_visible]
    original_to_compact = torch.full((total_len,), -1, dtype=torch.long, device=device)
    original_to_compact[kv_positions] = torch.arange(kv_positions.numel(), dtype=torch.long, device=device)

    cache_position = original_to_compact.index_select(0, query_positions).contiguous()
    position_ids = query_positions.unsqueeze(0).contiguous()
    bucket_hidden = hidden_states.index_select(1, query_positions).contiguous()
    attention_mask = _build_position_mask(query_positions, kv_positions, hidden_states.dtype)

    return BucketSpec(
        query_positions=query_positions,
        kv_positions=kv_positions,
        cache_position=cache_position,
        position_ids=position_ids,
        hidden_states=bucket_hidden,
        attention_mask=attention_mask,
    )


def _run_decoder(
    language_model: Any,
    hidden_states: torch.Tensor,
    cache: BucketCache,
    spec: BucketSpec,
    output_attentions: bool,
    output_hidden_states: bool,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], Optional[Tuple[torch.Tensor, ...]]]:
    model = language_model.model
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in model.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=spec.attention_mask,
            position_ids=spec.position_ids,
            past_key_value=cache,
            output_attentions=output_attentions,
            use_cache=True,
            cache_position=spec.cache_position,
        )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = model.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    return hidden_states, all_hidden_states, all_self_attns


def _restore_full_cache_(
    full_cache: Any,
    compact_cache: BucketCache,
    spec: BucketSpec,
    total_len: int,
) -> None:
    dst_idx = spec.query_positions
    src_idx = spec.cache_position
    for full_key, full_value, compact_key, compact_value in zip(
        full_cache.key_cache,
        full_cache.value_cache,
        compact_cache.key_cache,
        compact_cache.value_cache,
    ):
        full_key.index_copy_(2, dst_idx, compact_key.index_select(2, src_idx))
        full_value.index_copy_(2, dst_idx, compact_value.index_select(2, src_idx))

    full_positions = torch.arange(total_len, dtype=torch.long, device=spec.query_positions.device)
    full_cache._seen_tokens = total_len
    full_cache.cache_position = full_positions.unsqueeze(0)
    full_cache.vla_original_positions = full_positions
    full_cache.vla_next_original_position = total_len


class BucketGraphRunner:
    def __init__(
        self,
        language_model: Any,
        spec: BucketSpec,
        base_cache: BucketCache,
        output_attentions: bool,
        output_hidden_states: bool,
        graph_warmup: int,
    ) -> None:
        self.language_model = language_model
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.static_hidden = torch.empty_like(spec.hidden_states)
        self.static_position_ids = torch.empty_like(spec.position_ids)
        self.static_cache_position = torch.empty_like(spec.cache_position)
        self.static_attention_mask = torch.empty_like(spec.attention_mask)
        self.static_spec = BucketSpec(
            query_positions=spec.query_positions,
            kv_positions=spec.kv_positions,
            cache_position=self.static_cache_position,
            position_ids=self.static_position_ids,
            hidden_states=self.static_hidden,
            attention_mask=self.static_attention_mask,
        )
        self.cache = _make_empty_like_cache(base_cache)
        self.graph = torch.cuda.CUDAGraph()
        self.output = None

        self.copy_inputs_(spec, base_cache)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(graph_warmup):
                self.copy_inputs_(spec, base_cache)
                self.output = _run_decoder(
                    self.language_model,
                    self.static_hidden,
                    self.cache,
                    self.static_spec,
                    self.output_attentions,
                    self.output_hidden_states,
                )
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self.copy_inputs_(spec, base_cache)
        with torch.cuda.graph(self.graph):
            self.output = _run_decoder(
                self.language_model,
                self.static_hidden,
                self.cache,
                self.static_spec,
                self.output_attentions,
                self.output_hidden_states,
            )

    def copy_inputs_(self, spec: BucketSpec, cache: BucketCache) -> None:
        self.static_hidden.copy_(spec.hidden_states)
        self.static_position_ids.copy_(spec.position_ids)
        self.static_cache_position.copy_(spec.cache_position)
        self.static_attention_mask.copy_(spec.attention_mask)
        _copy_cache_(self.cache, cache)

    def replay(self, spec: BucketSpec, cache: BucketCache):
        self.copy_inputs_(spec, cache)
        self.graph.replay()
        return self.output, self.cache


class BucketGraphRuntime:
    def __init__(self) -> None:
        self.runners: Dict[Tuple[Any, ...], BucketGraphRunner] = {}

    def _runner_key(
        self,
        spec: BucketSpec,
        output_attentions: bool,
        output_hidden_states: bool,
        language_model: Any,
    ) -> Tuple[Any, ...]:
        return (
            str(spec.hidden_states.device),
            spec.hidden_states.dtype,
            int(spec.hidden_states.shape[1]),
            int(spec.kv_positions.numel()),
            bool(output_attentions),
            bool(output_hidden_states),
            id(language_model),
        )

    @torch.inference_mode()
    def run(
        self,
        language_model: Any,
        hidden_states: torch.Tensor,
        past_key_values: Any,
        reusable_patches: Any,
        deleted_patches: Any,
        cache_effective: bool,
        prune_effective: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        use_graph: bool,
        graph_warmup: int,
        max_graphs: int,
    ) -> CausalLMOutputWithPast:
        total_len = int(hidden_states.shape[1])
        spec = _build_bucket_spec(
            hidden_states,
            reusable_patches,
            deleted_patches,
            cache_effective,
            prune_effective,
        )
        base_cache = _clone_bucket_cache(past_key_values, spec.kv_positions)

        if use_graph and torch.cuda.is_available():
            key = self._runner_key(spec, output_attentions, output_hidden_states, language_model)
            runner = self.runners.get(key)
            if runner is None:
                if max_graphs > 0 and len(self.runners) >= max_graphs:
                    use_graph = False
                else:
                    runner = BucketGraphRunner(
                        language_model,
                        spec,
                        base_cache,
                        output_attentions,
                        output_hidden_states,
                        graph_warmup,
                    )
                    self.runners[key] = runner
            if runner is not None and use_graph:
                decoder_output, compact_cache = runner.replay(spec, base_cache)
            else:
                work_cache = _make_empty_like_cache(base_cache)
                _copy_cache_(work_cache, base_cache)
                decoder_output = _run_decoder(
                    language_model,
                    spec.hidden_states,
                    work_cache,
                    spec,
                    output_attentions,
                    output_hidden_states,
                )
                compact_cache = work_cache
        else:
            work_cache = _make_empty_like_cache(base_cache)
            _copy_cache_(work_cache, base_cache)
            decoder_output = _run_decoder(
                language_model,
                spec.hidden_states,
                work_cache,
                spec,
                output_attentions,
                output_hidden_states,
            )
            compact_cache = work_cache

        hidden, hidden_history, attentions = decoder_output
        _restore_full_cache_(past_key_values, compact_cache, spec, total_len)

        logits = language_model.lm_head(hidden).float()
        if attentions is None:
            attentions = ()
        attentions = attentions + ((spec.query_positions, spec.kv_positions),)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_history,
            attentions=attentions,
        )


def should_use_bucket_graph(
    model: Any,
    input_ids: Optional[torch.Tensor],
    inputs_embeds: Optional[torch.Tensor],
    past_key_values: Any,
    labels: Optional[torch.Tensor],
    use_cache: bool,
) -> bool:
    if not bool(getattr(model.config, "llm_bucket_graph_enable", False)):
        return False
    if not torch.cuda.is_available() or model.training:
        return False
    if labels is not None or not use_cache:
        return False
    if past_key_values is None or not hasattr(past_key_values, "key_cache"):
        return False
    if len(past_key_values.key_cache) == 0 or past_key_values.get_seq_length() == 0:
        return False
    input_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
    if past_key_values.get_seq_length() < input_len:
        return False
    if input_len == 1:
        return False
    if input_ids is not None and input_ids.shape[0] != 1:
        return False
    if inputs_embeds is not None and inputs_embeds.shape[0] != 1:
        return False
    return True
