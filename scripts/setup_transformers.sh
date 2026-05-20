#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRANSFORMERS_DIR="${REPO_ROOT}/src/transformers"
PATCH_FILE="${REPO_ROOT}/patches/transformers/0001-visual-token-delete-support.patch"
REMOTE_URL="${TRANSFORMERS_REMOTE_URL:-https://github.com/siyuhsu/transformers.git}"
BRANCH="${TRANSFORMERS_BRANCH:-vla-cache-openvla}"
COMMIT_SUBJECT="Add visual token delete support to LLaMA cache reuse"

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Missing patch file: ${PATCH_FILE}" >&2
  exit 1
fi

if [[ ! -d "${TRANSFORMERS_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${TRANSFORMERS_DIR}")"
  git clone --branch "${BRANCH}" "${REMOTE_URL}" "${TRANSFORMERS_DIR}"
else
  if ! git -C "${TRANSFORMERS_DIR}" diff --quiet || ! git -C "${TRANSFORMERS_DIR}" diff --cached --quiet; then
    echo "src/transformers has local changes. Commit/stash them before running this setup script." >&2
    exit 1
  fi
  git -C "${TRANSFORMERS_DIR}" fetch origin "${BRANCH}"
  git -C "${TRANSFORMERS_DIR}" checkout "${BRANCH}"
fi

if git -C "${TRANSFORMERS_DIR}" log --oneline -50 | grep -Fq "${COMMIT_SUBJECT}"; then
  echo "Transformers visual-token-delete patch is already present."
else
  git -C "${TRANSFORMERS_DIR}" am "${PATCH_FILE}"
fi

echo "Transformers setup complete:"
git -C "${TRANSFORMERS_DIR}" log --oneline -3
