# Code Synchronization Notes

This repository intentionally does not version model checkpoints or experiment outputs. To reproduce the full method on another server, clone the repository and initialize the code dependencies below.

## Included in the Main Repository

The main repository tracks:

- OpenVLA changes under `src/openvla/`
- OpenVLA-OFT changes under `src/openvla-oft/`
- ViT/timm changes under `src/pytorch-image-models/`
- LIBERO submodule declarations in `.gitmodules`
- experiment launch scripts under `scripts*/` and `src/openvla*.sh`
- the Codex handoff memory in `AGENTS.md`

## Transformers Fork

The LLM-side cache reuse and visual-token-delete implementation lives in a modified `transformers` checkout. The local change is stored as a git-managed patch:

```text
patches/transformers/0001-visual-token-delete-support.patch
```

After cloning this repository on a new machine, run:

```bash
bash scripts/setup_transformers.sh
```

This clones `https://github.com/siyuhsu/transformers.git` at branch `vla-cache-openvla` into `src/transformers/` and applies the local visual-token-delete patch. Override the source if needed:

```bash
TRANSFORMERS_REMOTE_URL=<your-fork-url> TRANSFORMERS_BRANCH=<branch> bash scripts/setup_transformers.sh
```

Do not replace this checkout with upstream HuggingFace `transformers`; the method depends on the patched LLaMA cache code.

## LIBERO

LIBERO is tracked as submodules:

```bash
git submodule update --init --recursive
```

This should populate:

- `src/openvla/LIBERO`
- `src/openvla-oft/LIBERO`

## Checkpoints

Checkpoints are intentionally not tracked by git. Download or copy them manually to:

- `src/openvla/checkpoints/`
- `src/openvla-oft/checkpoints/`

The checkpoint directories are large and should not be committed.

## Minimal Setup Flow

```bash
git clone --recursive <dcvla-repo>
cd dcvla
git submodule update --init --recursive
bash scripts/setup_transformers.sh
# download checkpoints into src/openvla/checkpoints and src/openvla-oft/checkpoints
```
