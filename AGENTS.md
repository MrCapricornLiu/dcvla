# Codex Handoff Memory

This file is the working memory for future Codex sessions in this repository. Read it before making code or paper changes.

## Project Goal

The project is about efficient inference for Vision-Language-Action (VLA) models. The current method evolved from the ACL-submitted CTR paper into an EMNLP revision plan:

- Original method: coordinated cross-modal token reuse using one patch-level mask for both the vision encoder (VE/ViT) and the LLM cache.
- Current extension: add LLM-side token pruning, producing a CRP three-way handling of visual tokens:
  - compute: dynamic or task-relevant tokens that are recomputed,
  - reuse: static and task-irrelevant tokens that keep cached K/V,
  - prune: lowest-value subset among reusable static tokens, removed from the effective attention context.
- Motivation for EMNLP: reviewers liked the practical topic but criticized limited comparisons, overhead analysis, and the vision-side acceleration ceiling. Adding pruning helps make the LLM-side contribution less identical to VLA-Cache and combines the spirit of VLA-Cache and EfficientVLA.

Main paper paths:

- ACL paper: `acl/main.tex`
- ACL reviews: `acl/reviews.md`
- Prior long local discussion log: `sessions/rollout-2026-05-12T19-16-10-019e1be6-c899-7523-bcf3-2df05f456444.jsonl`
- Older paper files: `Papers/dcvla_paper/main-acl.tex`, `Papers/dcvla_paper/appendix.tex`

## Current Git / Code State

Recent commits:

- `96ebe55 Add coordinated token delete support`
- `16564f1 Add different mask for ablation`
- `78f9ea3 Add dual cache func for openvla-oft`
- `4c85fec Add 3 different pixel change detection methods`
- `c212431 Add vit keyframe interval code`
- `e4d681b kv reuse logic, optimize further`
- `95ef1d4 Add kv reuse logic`
- `c651726 KV reuse logic, but without real computation reduced`

Important: early commit `c651726` was a fake/incorrect-ish version that did not really reduce computation. Current HEAD is different.

As of the handoff, `git status --short` was clean before creating this file.

## Core Implementation Files

OpenVLA:

- `src/openvla/experiments/robot/libero/run_libero_eval.py`
- `src/openvla/experiments/robot/openvla_utils.py`
- `src/openvla/experiments/robot/vla_cache_utils.py`

OpenVLA-OFT:

- `src/openvla-oft/experiments/robot/libero/run_libero_eval.py`
- `src/openvla-oft/experiments/robot/openvla_utils.py`
- `src/openvla-oft/experiments/robot/vla_cache_utils.py`

ViT/timm:

- `src/pytorch-image-models/timm/models/vision_transformer.py`

LLM/Transformers:

- `src/transformers/src/transformers/models/llama/modeling_llama.py`

Do not replace the local `src/transformers` with upstream HuggingFace `transformers`; this repo relies on a modified transformer implementation for VLA-Cache / CTR behavior.

## What the Current Method Actually Does

### Overhead-Control Baseline

Current code supports an overhead-controlled baseline for latency comparisons:

- `cache_overhead_benchmark=True` and `llm_cache_benchmark=True` make the baseline run ViT/LLM mask selection, reuse/prune bookkeeping, cache overwrite/update paths, and dummy pruning indexing even when acceleration switches are disabled.
- Effective acceleration is still controlled separately:
  - `use_vit_cache` + `vit_cache_reuse` decide whether ViT actually skips recomputation;
  - `vit_delete_enable` is the legacy code flag for whether ViT pruning actually hides K/V context;
  - `use_vla_cache` decides whether LLM reuse actually prunes query rows;
  - `llm_delete_enable` is the legacy code flag for whether LLM pruning actually shortens K/V context.
- For overhead-controlled dense baseline, use switches off but keep ratios set, e.g. `use_vla_cache=False`, `use_vit_cache=False`, `vit_delete_enable=False`, `llm_delete_enable=False`, `vit_delete_ratio=0.1`, `llm_delete_ratio=0.1`. These are legacy CLI names; paper/discussion should call the operation prune.
- This baseline is intentionally not a pure dense model. It is meant to subtract common mask/cache/bookkeeping overhead so the latency difference better reflects the net benefit of reduced token computation.

### Patch Selection

The patch mask is built from:

- temporal consistency: cosine, grayscale diff, or RGB diff;
- task relevance: top-K text-to-vision attention from the previous timestep.

Conceptually:

- `D_t`: dynamic patches from temporal comparison;
- `A_{t-1}`: task-relevant patches from previous-step attention;
- compute set: `D_t union A_{t-1}`;
- reuse/prune candidates: complement of recompute set among visual patches;
- prune set: optional lowest-attention subset of the reuse/prune candidates;
- reuse set: remaining candidates after removing the prune set.

The current code still has some variable names inherited from static/reuse wording. Be careful when editing paper notation: the clean explanation should use dynamic/task/recompute/reuse rather than only `S \ A`.

### ViT / Vision-Encoder Reuse

This point was a major source of confusion and must be described accurately.

Current HEAD does reduce ViT attention computation. It is not the fake full `N x N` version.

In `src/pytorch-image-models/timm/models/vision_transformer.py`:

- static/reused tokens do not issue current-frame queries;
- dynamic tokens compute Q/K/V;
- dynamic K/V update the per-block cache;
- without ViT pruning, attention is computed as `Q_dynamic x K_full/V_full`;
- with optional ViT pruning enabled, attention is computed as `Q_dynamic x K_visible/V_visible`, where pruned static tokens are filtered out of the effective K/V context;
- MLP is computed only for dynamic tokens;
- static tokens reuse cached post-block / sublayer outputs;
- full-length output sequence and token order are preserved.

Therefore ViT attention is effectively:

```text
dense baseline: N x N
VE reuse: D x N
VE reuse + prune: D x (N - P)
not: D x D
```

This means reviewer criticism saying "no token grid shortening" is partly correct, but "no attention compute reduction" is wrong for current HEAD. The paper should call the reuse-only path **partial-query attention / query-side sparse recomputation** and the prune extension **effective K/V-context pruning** rather than physical token-grid shortening.

Important latest latency result: current ViT prune does **not** show positive wall-time contribution in fixed-input microbenchmarks. ViT reuse is the stable speedup source; prune currently adds K/V-context filtering and tends to slow the ViT path.

Important superseding profiling note from 2026-05-22:

- A detailed ViT profiling pass found that some earlier ViT latency numbers overstated reuse speedup because the overhead-controlled baseline did not enter the same compact-dynamic wrapper as the reuse path.
- Before the fix, fixed-input `skip=0` vs `skip=75` showed `41.325 ms -> 19.915 ms` CUDA. Torch profiler showed the cause was not token FLOPs: `skip=0` spent about `3.190 ms` self CUDA in `aten::nonzero` across `414` calls, while `skip=75` spent only about `0.046 ms` across `6` calls. The method path computed dynamic indices once in the compact wrapper, while the baseline recomputed mask indices inside each block.
- The fix in `src/pytorch-image-models/timm/models/vision_transformer.py` lets the benchmark/cache path use the same compact wrapper even when static reuse is disabled or `dynamic_idx` covers all tokens. In that case FLOPs are not reduced, but the control path matches the method more closely. The ViT FLOPs estimator was also corrected: output projection is computed only for dynamic tokens in the compact reuse path.
- After the fix, fixed-input `skip=0` vs `skip=75` with no pruning is `18.308 ms -> 18.253 ms` mean CUDA (`~0.3%` speedup), output `results_sanity/vit_reuse_theory_check_skip0_75_compactbaseline_20260522/summary.json`. Torch profiler after the fix shows similar top ops and no large `aten::nonzero` imbalance (`skip=0` self CUDA total `6.523 ms`, `skip=75` `6.151 ms`).
- Therefore, do not use pre-fix ViT latency runs to claim large VE reuse speedups. Re-run rollout latency after this fix before making paper claims. The corrected interpretation is that current ViT reuse reduces theoretical FLOPs, but on this small token count and current compact/cache implementation the wall-time gain can be tiny once the baseline control path is made fair.
- Follow-up shape profile after the fairness fix: `results_sanity/vit_profile_afterfix_shapes_20260522/timing_summary.json`. Mean CUDA time by skipped patch count was `skip=0`: `18.001 ms`, `skip=75`: `17.948 ms` (`0.30%` faster), `skip=128`: `17.716 ms` (`1.59%`), `skip=192`: `17.618 ms` (`2.13%`), `skip=224`: `17.435 ms` (`3.15%`), `skip=250`: `16.641 ms` (`7.55%`). This confirms that even very aggressive query-side sparsity gives only small wall-time gains with the current backend.
- The detailed profiler diagnosis is that the reduced dynamic-token shapes do appear in `aten::addmm`, but the kernels are too small to scale linearly with FLOPs. Flash attention also barely changes because reuse-only remains `D x N` rather than `D x D`, and launch/fixed overhead dominates at `N ~= 256`. The hidden bottleneck is per-block cache scatter/update: `aten::index_copy_` still writes into full-size `[1, N, C]` and `[1, H, N, d]` buffers in every block, so shrinking the source index from all tokens to a small dynamic subset does not remove most of that kernel time. Current VE logic is behaviorally correct, but meaningful latency speedup likely requires a fused/compact implementation that avoids per-block full-buffer scatter, keeps compact dynamic streams across blocks, or uses custom kernels.
- Oracle microbench added: `scripts/oracle_crp_microbench.py`. It does not run CRP; it runs dense blocks/model on physically shorter sequences to estimate the upper bound if gather/scatter/cache overhead disappeared.
- Oracle run `results_sanity/oracle_crp_20260522/summary.json`: ViT combined alpha+beta blocks, dense short sequence, `15` measured iterations. Mean CUDA was `skip=0`: `11.072 ms`, `skip=75`: `10.909 ms` (`1.47%`), `skip=128`: `10.650 ms` (`3.81%`), `skip=192`: `10.587 ms` (`4.38%`), `skip=224`: `10.480 ms` (`5.35%`), `skip=250`: `10.064 ms` (`9.11%`). This means that at OpenVLA's small ViT token counts, even an ideal physical short-sequence dense backend has only modest wall-time headroom.
- Superseding LLM oracle correction: the first LLM oracle (`results_sanity/oracle_crp_20260522/summary.json` and `results_sanity/oracle_crp_llm_attn_modes_20260522/summary.json`) was flawed because it passed `DynamicCache()` to work around a local `LlamaModel.forward` assumption; attention layers update cache whenever `past_key_value is not None`, even with `use_cache=False`. Do not use those LLM oracle numbers.
- Corrected clean LLM oracle: `results_sanity/oracle_crp_llm_decoder_clean_20260522/summary.json`. This directly runs LLaMA decoder layers + final norm + `lm_head`, with no `DynamicCache`, on physically shorter dense sequences. Full multimodal length was `283` tokens (`27` text/action-prompt + `256` visual). With `output_attentions=False`, mean CUDA was `skip=0`: `18.360 ms`, `skip=75`: `17.617 ms` (`4.05%`), `skip=128`: `17.594 ms` (`4.17%`), `skip=192`: `17.418 ms` (`5.13%`), `skip=224`: `17.508 ms` (`4.64%`), `skip=250`: `17.470 ms` (`4.85%`). With `output_attentions=True`, mean CUDA was `skip=0`: `21.234 ms`, `skip=75`: `20.970 ms` (`1.24%`), `skip=128`: `20.687 ms` (`2.58%`), `skip=192`: `20.782 ms` (`2.13%`), `skip=224`: `20.843 ms` (`1.84%`), `skip=250`: `20.943 ms` (`1.37%`). Correct interpretation: physical short LLM prefill does speed up, but much less than FLOPs at this short sequence length; output-attention/manual-attention mode further reduces the wall-time sensitivity to sequence length.
- Follow-up LLM component/profiler check: `results_sanity/llm_gemm_scaling_20260522.json` and `results_sanity/llm_lmhead_scaling_20260522.json`. Individual layer GEMMs do shrink with sequence length, but not FLOPs-linearly: e.g. layer-0 `mlp_full` was `0.234 ms` at `L=283` vs `0.151 ms` at `L=33`, and `lm_head` was `0.169 ms` vs `0.122 ms`. Torch profiler on clean decoder `output_attentions=False` showed summed kernel self CUDA time dropping from about `16.53 ms` at `L=283` to `10.24 ms` at `L=33`; however CUDA event end-to-end latency drops much less because it includes gaps between many short kernels, CPU launch overhead, and low-occupancy small GEMMs. This suggests the missing speedup is largely a backend scheduling/fusion/CUDA-graph issue, not that shorter sequences do no less work.
- CUDA Graph/static-bucket oracle added on 2026-05-22: `scripts/llm_cuda_graph_oracle.py`. It captures fixed-shape dense LLaMA decoder graphs for each sequence length and compares normal eager execution, graph replay, and copy-into-static-buffer plus graph replay. Full run `results_sanity/llm_cuda_graph_oracle_full_20260522/summary.json`, 10 measured iterations, confirmed that the short-sequence speedup appears when launch/dispatch gaps are removed. With `output_attentions=False`, normal eager was nearly flat (`18.350 ms` at `L=283` to `17.843 ms` at `L=33`, only `2.76%` faster), while CUDA Graph replay was `16.072 ms` to `10.059 ms` (`37.41%` faster). With `output_attentions=True`, normal eager was `21.441 ms` to `21.238 ms` (`0.95%` faster), while CUDA Graph replay was `18.217 ms` to `10.615 ms` (`41.73%` faster). Copy+replay was essentially identical to replay, so input buffer copy is not the limiting factor in this oracle.
- Superseding implementation direction: do not abandon CRP, but switch the LLM implementation away from per-forward dynamic PyTorch indexing/control flow. The next viable path is bucketed/static-shape CRP: precompute masks/plans, compact into a fixed bucket once, run the LLM prefill through a captured graph or compile-friendly static path, and restore/maintain the full decode cache outside the captured compute path. Current `LlamaModel.forward` still has tensor-dependent Python branches (`bool(tensor.any())`, `.item()`, `torch.unique`, `torch.isin`, per-layer `index_select`/`index_copy_`) and is not the right execution surface for graph-captured pruning.
- Bucketed CRP CUDA Graph prototype added on 2026-05-22: `scripts/llm_bucketed_crp_graph_microbench.py`. Unlike the dense oracle, this uses real OpenVLA multimodal embeddings, a warmed previous-frame KV cache, and real masks from `scripts/llm_crp_microbench.py`. It constructs two fixed buckets: dense (`q=291, kv=291` for the current prompt) and CRP (`q=155, kv=231` for `--target-skip-count 180 --target-prune-count 60`; actual masks were reuse `76`, prune `60`, because top-attention protected patches limit the skip pool). It uses original-position RoPE ids and an explicit original-position causal mask; graph-vs-eager validation showed `max_abs_diff=0.0` for both dense and CRP buckets.
- Bucketed prototype results: `results_sanity/llm_bucketed_crp_graph_skip180_prune60_false_20260522/summary.json` (`output_attentions=False`, 10 iterations) showed eager dense `20.138 ms`, eager CRP `20.174 ms` (no speedup), graph dense `17.069 ms`, graph CRP `13.124 ms` (`23.11%` faster), copy+graph dense `17.398 ms`, copy+graph CRP `13.453 ms` (`22.67%` faster). `results_sanity/llm_bucketed_crp_graph_skip180_prune60_true_20260522/summary.json` (`output_attentions=True`, 8 iterations) showed eager dense `22.240 ms`, eager CRP `22.016 ms` (`1.01%`), graph dense `18.890 ms`, graph CRP `14.166 ms` (`25.01%`), copy+graph dense `19.226 ms`, copy+graph CRP `14.499 ms` (`24.59%`). This confirms that actual CRP masks become latency-positive under a fixed-shape graph backend; the current eager implementation is the bottleneck.
- First OpenVLA rollout-path integration added on 2026-05-22. New source module `src/openvla/prismatic/extern/hf/llm_bucket_graph.py` implements an exact-shape bucket graph runtime for multimodal LLM prefill. It is gated by config flags and defaults off. `src/openvla/prismatic/extern/hf/modeling_prismatic.py` now routes only eligible multimodal prefill calls through this runtime; `q_len=1` action decode remains on the normal full-cache path. `src/openvla/experiments/robot/libero/run_libero_eval.py` adds flags: `llm_bucket_graph_enable`, `llm_bucket_graph_capture`, `llm_bucket_graph_warmup`, `llm_bucket_graph_max_graphs`, and `llm_bucket_graph_fallback`. `src/openvla/experiments/robot/openvla_utils.py` copies these flags onto `vla.config` and now syncs `llm_bucket_graph.py` into checkpoint dynamic-module folders together with `modeling_prismatic.py`.
- Fixed-input main-path checks after integration: for CRP bucket semantics (`target_skip_count=180`, `target_prune_count=60`, actual reuse `76`, prune `60`), `predict_action/generate` with `llm_bucket_graph_capture=False` and `True` produced identical actions (`max_action_abs_diff=0.0`), full restored cache length `291`, and attention meta lengths `q=155, kv=231`. Dense baseline graph path was also checked with `vla_cache_effective=False`/`vla_delete_effective=False`; capture off vs on again had `max_action_abs_diff=0.0`, attention meta `q=291, kv=291`.
- One real LIBERO Spatial rollout smoke with OpenVLA graph path enabled succeeded (`1/1` success) using `--use_vla_cache True --use_vit_cache True --llm_delete_enable True --llm_delete_ratio 0.1 --llm_bucket_graph_enable True --llm_bucket_graph_capture True --llm_bucket_graph_warmup 1 --llm_bucket_graph_fallback False`. The run showed expected graph-capture latency spikes when new exact `(q_len, kv_len, output_attentions)` shapes appeared; after capture, steady LLM wall time was around `159-170 ms`. Average for this one rollout was inflated by captures: total wall `343.362 ms`, LLM wall `292.185 ms`. Next step before paper-quality timing is to add padded/static buckets or a graph warmup phase so capture overhead is not paid during measured rollout.
- Padded static bucket execution was added after the exact-shape rollout integration. Runtime now rounds real CRP lengths upward to fixed buckets via `llm_bucket_graph_q_buckets` and `llm_bucket_graph_kv_buckets` (default `64,96,128,160,192,224,256,288,320,352`, plus the current dense `total_len` automatically). Padding changes only the execution tensor shape: dummy queries are not returned in logits, dummy K/V are causally masked from real queries, attention maps/position meta are cropped back to real query/KV lengths, and full-cache restore scatters only real computed query rows. Fixed-input check with actual reuse `76`, prune `60` confirmed `q_real=155`, `kv_real=231` executed as bucket `q=160`, `kv=256`; exact-bucket eager vs padded-bucket eager vs padded-bucket graph all produced identical actions (`max_abs_diff=0.0`) and restored full cache length `291`.
- Padded bucket real LIBERO Spatial smoke succeeded (`1/1` success) with `--llm_bucket_graph_enable True --llm_bucket_graph_capture True --llm_bucket_graph_warmup 1 --llm_bucket_graph_max_graphs 16 --llm_bucket_graph_fallback False`, log `experiments/logs/EVAL-libero_spatial-openvla-2026_05_22-13_18_51--padded_bucket_graph_smoke.txt`. Average for this one rollout was total wall `258.377 ms`, LLM wall `204.845 ms`; the first occurrences of a few bucket shapes still caused capture spikes, but steady post-capture steps were around total `208-218 ms`, LLM `162-170 ms`. For paper-quality timing, run an unmeasured graph warmup rollout or otherwise pre-capture common bucket shapes before measuring.
- `src/openvla/experiments/robot/libero/run_libero_eval.py` now has `num_warmup_trials_per_task` for same-process unmeasured warmup rollouts, which is required for CUDA Graph capture caches to survive into measured trials.
- First 10-rollout LIBERO Spatial comparison after padded buckets, both with `num_warmup_trials_per_task=1`, both using `llm_bucket_graph_enable=True`, `llm_bucket_graph_capture=True`, and overhead-controlled bookkeeping. Baseline command used actual acceleration off (`use_vla_cache=False`, `use_vit_cache=False`, `vit_delete_enable=False`, `llm_delete_enable=False`) but kept benchmark overhead and dense bucket graph on; log `experiments/logs/EVAL-libero_spatial-openvla-2026_05_22-13_24_42--baseline_padded_bucket_cmp10.txt`. All-on used `use_vla_cache=True`, `use_vit_cache=True`, `vit_delete_enable=True`, `vit_delete_ratio=0.1`, `llm_delete_enable=True`, `llm_delete_ratio=0.1`; log `experiments/logs/EVAL-libero_spatial-openvla-2026_05_22-13_24_42--allon_padded_bucket_cmp10.txt`. Per-episode latency averages over 10 measured rollouts: baseline success `9/10`, total wall `214.777 ms`, ViT `21.471`, LLM `169.489`; all-on success `8/10`, total wall `213.814 ms`, ViT `23.610`, LLM `165.440`. Interpretation: padded bucket LLM path shows a small positive rollout signal (`~2.39%` LLM speedup), but all-on overall is only `~0.45%` faster because ViT prune/extra overhead adds about `2.14 ms`. Do not claim strong all-on speedup from this short run.

- Script added: `scripts/vit_crp_microbench.py`.
- Run 1, close to spatial rollout ratios: `skip=80`, prune counts `0/8/24/48`, 20 measured iterations. Output `results_sanity/vit_prune_microbench_20260522/summary.json`.
  - reuse-only `prune=0`: mean CUDA `18.138 ms`.
  - reuse-only with dummy prune overhead `prune=8`: `18.156 ms`.
  - effective reuse+prune `prune=8`: `18.948 ms`.
  - effective reuse+prune `prune=24`: `18.967 ms`; `prune=48`: `19.007 ms`.
- Run 2, more aggressive: `skip=160`, prune counts `0/16/48/96/128`, 20 measured iterations. Output `results_sanity/vit_prune_microbench_skip160_20260522/summary.json`.
  - reuse-only `prune=0`: mean CUDA `17.764 ms`.
  - effective reuse+prune `prune=16`: `18.447 ms`; `prune=48`: `18.459 ms`; `prune=96`: `18.533 ms`; `prune=128`: `18.297 ms`.
- Run 3, larger grid after noting rollout prune was too small to conclude from: `skip=64/128/192/224`, prune counts up to full skipped set, 15 measured iterations. Output `results_sanity/vit_prune_microbench_grid_20260522/summary.json`.
  - `skip=64`: reuse-only `18.887 ms`; reuse+prune `P=32` `19.660 ms`, `P=64` `19.645 ms`.
  - `skip=128`: reuse-only `18.488 ms`; reuse+prune `P=32` `19.266 ms`, `P=64` `19.331 ms`, `P=96` `19.284 ms`, `P=128` `19.084 ms`.
  - `skip=192`: reuse-only `18.455 ms`; reuse+prune `P=32` `19.167 ms`, `P=128` `18.978 ms`, `P=192` `19.036 ms`.
  - `skip=224`: reuse-only `18.254 ms`; reuse+prune `P=32` `19.004 ms`, `P=128` `18.846 ms`, `P=224` `18.801 ms`.
- Detail profile `results_sanity/vit_prune_microbench_detail_20260522/stdout.log` shows the attention computation itself can decrease when K/V context is pruned, but the current implementation's K/V `index_select` is not separately reported in the detail keys and appears as hidden wall-time overhead. Net effect remains negative.
- Important comparison nuance: the above fixed-`skip` tests compare `reuse R + prune P` against `reuse R+P`, i.e. whether converting reuse tokens into prune tokens has marginal benefit. The paper's useful ablation may instead compare `reuse R + prune P` against `reuse R`, i.e. keeping the reuse budget fixed and adding extra pruned tokens.
- Additive-prune run: `results_sanity/vit_prune_additive_microbench_20260522/summary.json`, 15 measured iterations. It compares `reuse-only R` with `reuse R + prune P` by using `skip=R+P, prune=P`.
  - `R=50,P=50`: `18.858 -> 19.295 ms` CUDA, slower.
  - `R=50,P=150`: `18.858 -> 18.871 ms`, roughly neutral.
  - `R=50,P=200`: `18.858 -> 17.877 ms`, `+5.20%` faster.
  - `R=100,P=150`: `18.539 -> 18.026 ms`, `+2.77%` faster.
  - `R=150,P=100`: `18.538 -> 18.064 ms`, `+2.55%` faster.
  - `R=200,P=50`: `18.433 -> 18.192 ms`, `+1.31%` faster.
- Updated interpretation: current ViT prune is not useful as a replacement for reuse under a fixed total skipped-token budget. It can be useful as an **additional** CRP branch when prune adds enough extra skipped/context-pruned tokens beyond a fixed reuse set. For the paper, ablate both views: `reuse R` vs `reuse R + prune P` for the method claim, and optionally `reuse R+P` vs `reuse R + prune P` to show prune-vs-reuse tradeoff.

Important code locations:

- dynamic indices in attention path: `vision_transformer.py`, around the `dynamic_idx = (~effective_static)` branch.
- SDPA uses `q_dyn, k_ctx, v_ctx`: dynamic query against full cached K/V in reuse-only mode, or visible K/V after applying the legacy `delete_mask` variable when ViT pruning is enabled.
- MLP dynamic-only path: `dynamic_idx = (~reuse_mask)` and `self.mlp(x_dyn_norm)`.
- ViT tri-state masks are controlled by legacy code flags `vit_delete_enable` and `vit_delete_ratio`. Internally, `reuse_mask` means "skip current recomputation" (`reuse union prune`), while `delete_mask` is the legacy variable name for the prune subset hidden from effective attention K/V.
- keyframe refresh exists; it periodically disables reuse and refreshes ViT cache.

### LLM Reuse and Prune

LLM side builds on VLA-Cache style prefill K/V reuse.

- `reusable_patches`: visual positions that reuse previous-frame K/V.
- `deleted_patches`: legacy code name for the lowest-attention subset among reusable static candidates when pruning is enabled.
- `proportion_attn_var`: entropy-derived layer-wise reuse schedule from VLA-Cache logic.

In OpenVLA `openvla_utils.py`, the LLM-side reuse and prune masks are set on `vla.language_model.config`:

- `config.reusable_patches`
- `config.deleted_patches`
- `config.proportion_attn_var`

The EMNLP story should emphasize CRP three-way token treatment on the LLM side:

- compute tokens: dynamic/task-relevant tokens recompute/update K/V,
- static/task-irrelevant tokens: reuse cached K/V,
- prune tokens: least useful reusable static tokens are removed from the effective attention context.

Important precision for paper/code discussion:

- Current pruning is effective K/V-context pruning during multimodal prefill. It filters low-value visual tokens out of the attention K/V seen by the current prefill forward.
- Current pruning is not persistent physical cache compaction. The underlying `DynamicCache` still keeps the original full-position K/V layout, and `q_len=1` action decoding continues to use the full physical cache.
- Therefore, do not claim current pruning reduces persistent KV-cache memory. Its current benefit is prefill attention compute reduction by shortening the effective K/V context.
- Future work / TODO for the paper: implement true physical pruning of the persistent cache. Relative to the current effective prefill pruning, the certain extra benefit is lower persistent KV-cache memory; prefill attention compute is already reduced by the current effective K/V filtering. It would only further reduce computation if action decoding were also changed to attend over the compacted/pruned cache, which is a stronger behavioral change and requires position-aware cache compaction plus careful handling of cross-frame reuse.

Current LLM latency diagnosis:

- A fixed-input LLM CRP microbenchmark script exists at `scripts/llm_crp_microbench.py`. It avoids LIBERO, fixes prompt/image/cache/masks, and compares overhead-controlled baseline vs effective CRP with `output_attentions=True/False` and prefill/generate modes.
- Stable run: `results_sanity/llm_crp_microbench_interleaved_noempty_20260522_0640/summary.json`, GPU0, 5 warmup + 20 measured iterations, no CUDA cache clearing between runs.
- Mask stats in that run: stable patches 130, reuse positions 58, prune positions 6, schedule layers 32, schedule mean about 0.81.
- Result: effective CRP is consistently slower than the overhead-controlled baseline by about 6 ms in the LLM path:
  - prefill, `output_attentions=True`: 35.41 ms -> 41.54 ms LLM wall (+6.13 ms);
  - prefill, `output_attentions=False`: 31.52 ms -> 37.39 ms (+5.87 ms);
  - full 7-token generate, `output_attentions=True`: 158.79 ms -> 164.91 ms (+6.12 ms);
  - full generate, `output_attentions=False`: 139.79 ms -> 145.77 ms (+5.98 ms).
- Interpretation: this is not rollout noise. The current LLM CRP implementation reduces theoretical token work, but its real shape-changing path (`hidden_states`/`causal_mask`/`position_ids` slicing plus irregular downstream execution) costs more than the saved work at this problem size. The overhead-controlled baseline runs the bookkeeping and dummy slices, but it does not reproduce the downstream cost of actually changing tensor shapes.
- `output_attentions=True` still matters because `LlamaSdpaAttention` falls back to manual attention when attentions are requested, but both baseline and CRP pay that in real eval. The CRP-vs-baseline gap remains about 6 ms even with `output_attentions=False`.
- Later microbench runs added a fast precomputed CRP plan in `src/transformers/src/transformers/models/llama/modeling_llama.py`, plus optional env flags `VLA_LLM_SINGLE_COMPACT=1` and `VLA_LLM_SINGLE_COMPACT_LAYER=...` to apply all selected skip/prune rows at one layer. These remove repeated `torch.isin`/`unique`/`sort` from the decoder hot loop and are useful for diagnosis.
- The key split result is:
  - reuse-only can reduce wall time: with `VLA_LLM_SINGLE_COMPACT=1 VLA_LLM_SINGLE_COMPACT_LAYER=0`, `target_skip_count=90`, `target_prune_count=0`, prefill `output_attentions=False` improved from 21.30 ms to 20.74 ms; with about 136 skipped rows, it improved from 69.93 ms to 60.61 ms.
  - current K/V-context pruning is the slow path: with `target_skip_count=90`, `target_prune_count=30`, prefill `output_attentions=False` was 25.91 ms baseline vs 32.76 ms CRP; with about 136 skipped rows plus 30 pruned rows, it was 26.66 ms baseline vs 33.11 ms CRP; with `target_skip_count=120`, `target_prune_count=60`, it was 25.98 ms baseline vs 31.99 ms CRP.
- Practical conclusion as of 2026-05-22: query-side reuse can translate into real LLM latency savings, but the current pruning implementation's per-layer K/V `index_select`/shape-changing path is a net negative. More prune ratio alone does not fix this. To make LLM pruning match the expected FLOP reduction, the implementation likely needs a compact-cache or fused attention path where pruned K/V are not gathered inside every attention layer. Otherwise, for wall-time experiments, LLM-side prune should be treated as a TODO or disabled while keeping query-side reuse.
- FastV reference code was cloned to `for_ref/FastV` on 2026-05-22. The relevant implementation physically drops visual token rows once after a selected layer by updating `hidden_states`, `position_ids`, and `attention_mask`; it does not perform per-layer K/V filtering. TopV's paper makes the same systems point more explicitly: efficient VLM token pruning should be compatible with FlashAttention and KV cache and should perform pruning once during prefill.
- A compact-KV prototype now exists behind env flag `VLA_LLM_COMPACT_KV=1` in `src/transformers/src/transformers/models/llama/modeling_llama.py`. Use it together with `VLA_LLM_SINGLE_COMPACT=1 VLA_LLM_SINGLE_COMPACT_LAYER=0` for the current microbench. It compacts the cached K/V once using the prune-token complement, maps current-frame computed tokens from original positions to compact cache positions, and keeps RoPE `position_ids` on original positions.
- Compact-KV microbench results show the correct direction:
  - `results_sanity/llm_crp_microbench_prune30_skip90_compactkv_l0_rerun_20260522/summary.json`: `target_skip_count=90`, `target_prune_count=30`, prefill `output_attentions=False` improved from 24.99 ms to 22.03 ms; `output_attentions=True` improved from 28.98 ms to 25.63 ms.
  - `results_sanity/llm_crp_microbench_skip150_prune30_compactkv_l0_20260522/summary.json`: actual mask reuse 106 and prune 30, prefill `output_attentions=False` improved from 25.88 ms to 22.78 ms; `output_attentions=True` improved from 29.87 ms to 26.50 ms.
- Rollout validation showed that persistent physical compact/prune changes action behavior. A compact cache carried into `q_len=1` action decode caused the first spatial episode to fail, and an earlier 5-rollout all-on compact run got `0/5` success. This matches the intended method semantics: pruning should reduce multimodal prefill work, not permanently remove tokens from action decode.
- Current compact-KV implementation therefore restores the full original-position cache after multimodal prefill. The fast restore path scatters only current-frame recomputed compact rows back into the previous full cache. `VLA_LLM_PERSISTENT_COMPACT=1` disables this restore for diagnostics only and should not be used for correctness experiments.
- Spatial rollout checks on 2026-05-22:
  - overhead-controlled baseline, 5 rollouts: `4/5` success, avg wall `248.867 ms`, overhead `24.683`, ViT `32.751`, LLM `186.223`, action head `5.210`; log `results_sanity/openvla_spatial_compactkv_rollout5_20260522/baseline/stdout.log`.
  - reuse-only, 5 rollouts, VE reuse on, LLM reuse on, both VE/LLM prune off: `4/5` success, avg wall `219.845 ms`, overhead `22.346`, ViT `20.626`, LLM `172.575`, action head `4.298`; log `results_sanity/openvla_spatial_reuseonly_rollout5_20260522/stdout.log`. This small check suggests reuse-only preserves the baseline success rate on task 0 (`4/5` vs `4/5`) while improving total latency by about `11.7%` and LLM latency by about `7.3%`.
  - all-on without compact-KV, 1 rollout: success, avg wall `251.705 ms`, LLM `196.855`; log `results_sanity/openvla_spatial_compactkv_rollout5_20260522/all_on_no_compact_smoke/stdout.log`.
  - all-on compact with full-cache rebuild restore, 5 rollouts: `5/5` success, avg wall `253.340 ms`, LLM `200.953`; correct but restore copy overhead is too high.
  - all-on compact with in-place scatter restore, 5 rollouts: `5/5` success, avg wall `239.492 ms`, overhead `23.669`, ViT `22.363`, LLM `188.442`, action head `5.017`; log `results_sanity/openvla_spatial_compactkv_rollout5_20260522/all_on_restore_scatter5/stdout.log`.
- Latest interpretation: the total rollout latency now improves versus baseline because ViT/VE is faster, and correctness is restored. LLM compact/prune is not yet a clean rollout win: in the 5-rollout scatter-restore run, LLM latency is still slightly slower than baseline (`188.442 ms` vs `186.223 ms`). Do not claim LLM-side rollout speedup until the restore/compact path is optimized further or a fused attention/FlashAttention-compatible implementation is added.

### OpenVLA vs OpenVLA-OFT

Both have been modified to support the method. OpenVLA-OFT uses more vision branches (four ViT measurements in logs) and different checkpoint layout.

Checkpoint sizes seen locally:

- `src/openvla/checkpoints`: about `71G`
- `src/openvla-oft/checkpoints`: about `60G`
- `src/SimplerEnv-OpenVLA/checkpoints`: about `15G`

## Key Parameters Used Historically

Do not overstate that these were found by exhaustive search in the paper. We previously described them as yielding a moderate reuse ratio.

Frequently used / selected settings:

- patch metric: often `gray_diff` or `cosine`, depending on experiment table.
- main gray setting discussed for paper: gray threshold around `0.004`, attention top-K `160`, keyframe interval `5`.
- OpenVLA previous good cosine group: `g15_cos_0992_k160_s160_kf5` had strong Goal/Object but Long mixed.
- OpenVLA-OFT selected group: `20260104_001915_g25_gray_004_k160_s120_kf5`.

For final EMNLP experiments with pruning, re-check the actual command-line parameters and record them in results.

## Important Existing Results

Old results are not all obsolete. Treat them as follows:

- Dense baseline results can be reused if code path did not change.
- Old no-prune CTR results may be reused only if they were run after the `D x N` ViT implementation. If unsure, rerun no-prune CTR for key tables.
- Switch ablations and mask consistency ablations exist and can inform paper, but EMNLP method with pruning needs new final-method numbers.

Useful result files:

- OpenVLA grid summary: `results/table.md`
- OpenVLA-OFT grid summary: `results_oft/table-oft.md`
- Ablation summary: `results_ablation/result_ablation_all.md`
- Mask ablation summary: `results_mask_ablation/result.md`
- SimplerEnv table/logs: `results_simplerenv/table.md`, `results_simplerenv/*/*.log`

Known previous SimplerEnv reported table:

```text
Task       OpenVLA   OpenVLA+CTR
Move Near   49.6      51.3
Pick Coke   17.3      19.0
Drawer      32.4      33.3
Average     33.1      34.5
```

## Experiments Needed for EMNLP

The user is likely moving experiments to a company machine with more GPUs. The school server only has four usable GPUs, which is slow.

Minimal set:

1. Final method main experiment:
   - OpenVLA and OpenVLA-OFT
   - LIBERO Spatial/Object/Goal/Long
   - baseline vs `VE reuse + LLM reuse + LLM prune`
   - metrics: success, VE latency, LLM latency, total latency, TFLOPs, reuse ratio, prune ratio.

2. Prune ablation:
   - OpenVLA, four LIBERO suites if possible.
   - baseline
   - no-prune CTR (`VE reuse + LLM reuse`)
   - final method with pruning
   - random prune with same prune count/ratio.

3. Overhead/latency breakdown:
   - mask selection time,
   - VE attention time,
   - VE MLP time,
   - LLM prefill/generation time,
   - prune/reuse bookkeeping time.
   This directly answers ACL reviewer comments.

Recommended if time allows:

4. Prune ratio sweep:
   - ratio `0`, `0.05`, `0.10`, `0.15`, `0.20`
   - only 1-2 suites if time is tight.

5. Mask consistency ablation:
   - unified mask,
   - VE mask perturbed,
   - LLM mask perturbed.
   Existing results may already be enough for paper narrative.

6. SimplerEnv:
   - old results can be reused,
   - if time allows, rerun final prune method on Move Near / Pick Coke / Drawer.

## Reviewer Feedback to Address in EMNLP Revision

ACL scores were around 2.5 / 2.5 / 3 and the paper was rejected.

Main review concerns:

- Need detailed latency breakdown: VE, LLM, masking overhead, memory/bookkeeping.
- ViT side still preserves full token grid; reviewers worried about attention coupling and acceleration ceiling.
- Need clearer statement that current VE implementation reduces query-side attention (`D x N`) while preserving full sequence.
- Missing comparison with other training-free acceleration/token pruning methods.
- Evaluation perceived thin by one reviewer: limited actions/scenes and limited model diversity.
- Sensitivity to hard thresholds and mask consistency.

Paper changes to make:

- Update Method to describe VE reuse as partial-query recomputation, not merely cached output injection.
- Update complexity analysis: VE attention changes from `N x N` to `D x N`; LLM side may reduce effective sequence and also prune tokens.
- Add LLM-side pruning as a distinct contribution.
- Mention as future work that true persistent cache pruning could reduce persistent KV-cache memory. Current pruning already shortens the effective prefill K/V context for compute, but does not physically compact the cache; extra compute savings would only come if decode were also made to use the compacted/pruned cache.
- Keep training-free and architecture-preserving claim, but be precise: no weight changes or retraining; inference graph/cache logic changes.
- Add overhead breakdown and prune ablation tables.
- Discuss relationship to VLA-Cache, EfficientVLA, and TTF-VLA without making the contribution look derivative.

## Environment / Migration Notes

For LIBERO experiments, data is not too heavy; checkpoints and environments are the main burden.

Minimum migration package:

- whole repo `/home/lch/Documents/dcvla`;
- `src/openvla/checkpoints`;
- `src/openvla-oft/checkpoints`;
- relevant HuggingFace cache if avoiding downloads;
- conda env specs or conda-pack for `dcvla` and `dcvlaoft`.

Environment commands normally require:

```bash
conda activate dcvla
```

On the current company machine / this workspace, the working LIBERO/OpenVLA environment is a micromamba env at:

```bash
/mnt/bn/arnold-recommend/liuchenghao/micromamba/envs/dcvla
```

Prefer this environment for smoke tests and experiments here. Reliable direct form:

```bash
/mnt/bn/arnold-recommend/liuchenghao/micromamba/envs/dcvla/bin/python ...
```

Or activate it with micromamba:

```bash
eval "$(/mnt/bn/arnold-recommend/liuchenghao/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate dcvla
```

For LIBERO smoke/eval from Codex shells on this machine, set the MuJoCo renderer explicitly to avoid EGL device enumeration failures:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

Without this, robosuite may fail before model inference with an error like `MUJOCO_EGL_DEVICE_ID ... between 0 and -1`. A successful all-switch OpenVLA smoke was run on 2026-05-20 with:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=0 \
/mnt/bn/arnold-recommend/liuchenghao/micromamba/envs/dcvla/bin/python \
experiments/robot/libero/run_libero_eval.py ...
```

For OpenVLA-OFT use the OFT environment if available:

```bash
conda activate dcvlaoft
```

Common pitfalls:

- Do not install upstream `transformers -U`; local modified `src/transformers` is required.
- OpenVLA-OFT imports must resolve to `src/openvla-oft`, not `src/openvla`. Path pollution previously caused `get_libero_wrist_image` import errors.
- `diffusers`, `huggingface_hub`, and local `transformers` versions can conflict. Pin versions carefully instead of upgrading everything.
- For latency, run one process per GPU if possible. Multi-process sharing makes latency numbers unreliable.
- New GPU architectures may require a PyTorch/CUDA build that supports the GPU capability.
- SimplerEnv has extra environment issues: TensorFlow, SAPIEN/Vulkan, ManiSkill2-Real2Sim, FlashAttention toggles, and TimeLimit wrappers. Do not prioritize it unless LIBERO experiments are done.

## Script / Scheduling Lessons

Past GPU polling script bug:

- With `set -e`, `found=$(find_free_gpu)` exits the script when all GPUs are busy because `find_free_gpu` returns 1.
- Fix is:

```bash
found=$(find_free_gpu || true)
```

For reliable latency:

- use a low memory threshold so only one process lands on a GPU;
- or explicitly assign `CUDA_VISIBLE_DEVICES`.

For experiment outputs:

- Do not write rollout/evaluation logs under `/tmp`; use a persistent project directory such as `results_sanity/`, `results/`, `results_oft/`, or a clearly named run directory under the repository.

## Paper Terminology

Use consistent names:

- Method name: CRP / Compute-Reuse-Prune, or updated EMNLP method name if renamed.
- From now on, use **prune** in the paper and discussions. Avoid **delete** except when referring to legacy code/API names such as `llm_delete_enable`, `vit_delete_enable`, `deleted_patches`, or `delete_mask`.
- Implementation alias: current code names containing `delete` mean **prune**. Treat `delete_mask`, `deleted_patches`, `*_delete_enable`, and `*_delete_ratio` as legacy names for prune behavior. New code/comments/results should prefer `prune` names unless changing public CLI/config names would add risk.
- VE = vision encoder. Avoid mixing "ViT" and "VE" in tables unless defining once.
- LLMCache = LLM-side K/V reuse.
- VECache = vision-encoder-side reuse.
- Token classes are CRP:
  - compute = dynamic union task-relevant tokens that must be recomputed,
  - reuse = static and task-irrelevant tokens that reuse cached representations/K/V,
  - prune = lowest-value subset from reusable/static candidates removed from effective attention context.

Avoid saying VE "does not reduce attention computation." Correct statement:

```text
VE preserves the full output sequence but reduces current-frame attention queries from all tokens to recomputed tokens, changing the attention pattern from N x N to D x N.
```

## When Continuing

First questions to answer in any new session:

1. Which server is this running on, and are checkpoints present?
2. Is `git status` clean?
3. Are we running no-prune old CTR, final CRP method, or an ablation?
4. Are logs recording commit hash and parameters?
5. Are latency measurements isolated to one process per GPU?

Before a large run, always run a smoke test:

```bash
cd /path/to/dcvla/src/openvla
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=0 \
/mnt/bn/arnold-recommend/liuchenghao/micromamba/envs/dcvla/bin/python \
  experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir ../../results_sanity/dcvla_smoke \
  --num_trials_per_task 1 \
  --num_tasks 1 \
  --task_start_id 0 \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_keyframe_interval 5 \
  --llm_delete_enable True
```

Adjust checkpoint path and task suite for the target model.
