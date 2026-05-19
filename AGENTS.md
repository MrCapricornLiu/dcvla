# Codex Handoff Memory

This file is the working memory for future Codex sessions in this repository. Read it before making code or paper changes.

## Project Goal

The project is about efficient inference for Vision-Language-Action (VLA) models. The current method evolved from the ACL-submitted CTR paper into an EMNLP revision plan:

- Original method: coordinated cross-modal token reuse using one patch-level mask for both the vision encoder (VE/ViT) and the LLM cache.
- Current extension: add LLM-side token delete, producing a three-way handling of visual tokens:
  - recompute: dynamic or task-relevant tokens,
  - reuse: static and task-irrelevant tokens that keep cached K/V,
  - delete: lowest-value subset among reusable static tokens, removed on the LLM side.
- Motivation for EMNLP: reviewers liked the practical topic but criticized limited comparisons, overhead analysis, and the vision-side acceleration ceiling. Adding delete helps make the LLM-side contribution less identical to VLA-Cache and combines the spirit of VLA-Cache and EfficientVLA.

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

### Patch Selection

The patch mask is built from:

- temporal consistency: cosine, grayscale diff, or RGB diff;
- task relevance: top-K text-to-vision attention from the previous timestep.

Conceptually:

- `D_t`: dynamic patches from temporal comparison;
- `A_{t-1}`: task-relevant patches from previous-step attention;
- recompute set: `D_t union A_{t-1}`;
- reuse set: complement of recompute set among visual patches.

The current code still has some variable names inherited from static/reuse wording. Be careful when editing paper notation: the clean explanation should use dynamic/task/recompute/reuse rather than only `S \ A`.

### ViT / Vision-Encoder Reuse

This point was a major source of confusion and must be described accurately.

Current HEAD does reduce ViT attention computation. It is not the fake full `N x N` version.

In `src/pytorch-image-models/timm/models/vision_transformer.py`:

- static/reused tokens do not issue current-frame queries;
- dynamic tokens compute Q/K/V;
- dynamic K/V update the per-block cache;
- attention is computed as `Q_dynamic x K_full/V_full`;
- MLP is computed only for dynamic tokens;
- static tokens reuse cached post-block / sublayer outputs;
- full-length output sequence and token order are preserved.

Therefore ViT attention is effectively:

```text
dense baseline: N x N
current VE reuse: D x N
not: D x D
```

This means reviewer criticism saying "no token grid shortening" is partly correct, but "no attention compute reduction" is wrong for current HEAD. The paper should call this **partial-query attention / query-side sparse recomputation** rather than sequence pruning.

Important code locations:

- dynamic indices in attention path: `vision_transformer.py`, around the `dynamic_idx = (~effective_static)` branch.
- SDPA uses `q_dyn, cache.k, cache.v`: dynamic query against full cached K/V.
- MLP dynamic-only path: `dynamic_idx = (~reuse_mask)` and `self.mlp(x_dyn_norm)`.
- keyframe refresh exists; it periodically disables reuse and refreshes ViT cache.

### LLM Reuse and Delete

LLM side builds on VLA-Cache style prefill K/V reuse.

- `reusable_patches`: visual positions that reuse previous-frame K/V.
- `deleted_patches`: lowest-attention subset among reusable static candidates when delete is enabled.
- `proportion_attn_var`: entropy-derived layer-wise reuse schedule from VLA-Cache logic.

In OpenVLA `openvla_utils.py`, the LLM-side mask and delete are set on `vla.language_model.config`:

- `config.reusable_patches`
- `config.deleted_patches`
- `config.proportion_attn_var`

The EMNLP story should emphasize a three-way token treatment on the LLM side:

- dynamic/task tokens: recompute/update K/V,
- static/task-irrelevant tokens: reuse cached K/V,
- least useful static tokens: delete.

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

For final EMNLP experiments with delete, re-check the actual command-line parameters and record them in results.

## Important Existing Results

Old results are not all obsolete. Treat them as follows:

- Dense baseline results can be reused if code path did not change.
- Old no-delete CTR results may be reused only if they were run after the `D x N` ViT implementation. If unsure, rerun no-delete CTR for key tables.
- Switch ablations and mask consistency ablations exist and can inform paper, but EMNLP method with delete needs new final-method numbers.

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
   - baseline vs `VE reuse + LLM reuse + LLM delete`
   - metrics: success, VE latency, LLM latency, total latency, TFLOPs, reuse ratio, delete ratio.

2. Delete ablation:
   - OpenVLA, four LIBERO suites if possible.
   - baseline
   - no-delete CTR (`VE reuse + LLM reuse`)
   - final method with delete
   - random delete with same delete count/ratio.

3. Overhead/latency breakdown:
   - mask selection time,
   - VE attention time,
   - VE MLP time,
   - LLM prefill/generation time,
   - delete/reuse bookkeeping time.
   This directly answers ACL reviewer comments.

Recommended if time allows:

4. Delete ratio sweep:
   - ratio `0`, `0.05`, `0.10`, `0.15`, `0.20`
   - only 1-2 suites if time is tight.

5. Mask consistency ablation:
   - unified mask,
   - VE mask perturbed,
   - LLM mask perturbed.
   Existing results may already be enough for paper narrative.

6. SimplerEnv:
   - old results can be reused,
   - if time allows, rerun final delete method on Move Near / Pick Coke / Drawer.

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
- Update complexity analysis: VE attention changes from `N x N` to `D x N`; LLM side may reduce effective sequence and also delete tokens.
- Add LLM-side delete as a distinct contribution.
- Keep training-free and architecture-preserving claim, but be precise: no weight changes or retraining; inference graph/cache logic changes.
- Add overhead breakdown and delete ablation tables.
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

## Paper Terminology

Use consistent names:

- Method name: CTR or updated EMNLP method name if renamed.
- VE = vision encoder. Avoid mixing "ViT" and "VE" in tables unless defining once.
- LLMCache = LLM-side K/V reuse.
- VECache = vision-encoder-side reuse.
- Token classes:
  - dynamic,
  - task-relevant,
  - recompute = dynamic union task-relevant,
  - reuse = static and task-irrelevant,
  - delete = lowest-value subset from reusable/static candidates on LLM side.

Avoid saying VE "does not reduce attention computation." Correct statement:

```text
VE preserves the full output sequence but reduces current-frame attention queries from all tokens to recomputed tokens, changing the attention pattern from N x N to D x N.
```

## When Continuing

First questions to answer in any new session:

1. Which server is this running on, and are checkpoints present?
2. Is `git status` clean?
3. Are we running no-delete old CTR, final delete method, or an ablation?
4. Are logs recording commit hash and parameters?
5. Are latency measurements isolated to one process per GPU?

Before a large run, always run a smoke test:

```bash
conda activate dcvla
cd /path/to/dcvla/src/openvla
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir /tmp/dcvla_smoke \
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
