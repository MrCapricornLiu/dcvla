#!/usr/bin/env bash
# Unified GPU-polling script for minimal ablations on OpenVLA and OpenVLA-OFT.
# Four ablations total:
# 1) OpenVLA switch (VLA/Vit on-off combos) with one param set
# 2) OpenVLA mask inconsistency (ViT-only change, LLM-only change) with one param set
# 3) OpenVLA-OFT switch with one param set
# 4) OpenVLA-OFT mask inconsistency with one param set

set -e

ts=$(date +"%Y%m%d_%H%M%S")
BASE_PATH="/home/lch/Documents/dcvla/results_ablation"

CHECK_INTERVAL=10
COOLDOWN=40
WAIT_LOG_INTERVAL=40
MONITOR_GPUS=(0 1 3 4)

get_deploy_id() {
  local mon_id=$1
  if [ "$mon_id" -eq 3 ]; then
    echo 2
  elif [ "$mon_id" -eq 4 ]; then
    echo 3
  else
    echo "$mon_id"
  fi
}

get_threshold() {
  local mon_id=$1
  echo 2000
}

find_free_gpu() {
  for id in "${MONITOR_GPUS[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $id 2>/dev/null)
    if [ -z "$used" ]; then
      used=999999
    fi
    thr=$(get_threshold "$id")
    if [ "$used" -lt "$thr" ]; then
      echo "$id"
      return 0
    fi
  done
  return 1
}

run_task() {
  local cmd="$1"
  local found=""
  local last_log=0
  while true; do
    found=$(find_free_gpu || true)
    if [ -n "$found" ]; then
      break
    fi
    now=$(date +%s)
    if [ $((now - last_log)) -ge $WAIT_LOG_INTERVAL ]; then
      echo "⏳ 等待空闲GPU... (阈值=20000MB)"
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true
      last_log=$now
    fi
    sleep $CHECK_INTERVAL
  done
  local deploy_id
  deploy_id=$(get_deploy_id "$found")
  echo "✅ 使用GPU 监听ID: $found -> 部署ID: $deploy_id"
  CUDA_VISIBLE_DEVICES=$deploy_id bash -c "$cmd" &
  sleep $COOLDOWN
}

run_suite_openvla() {
  local task=$1
  local ckpt=$2
  local logdir=$3
  local use_vla=$4
  local use_vit=$5
  shift 5
  local extra_flags="$*"

  local logfile="$logdir/${task}.log"
  echo "  ↳ 任务 ${task} -> Log: ${logfile}"

  run_task "source /home/lch/miniconda3/etc/profile.d/conda.sh && conda activate dcvla && cd /home/lch/Documents/dcvla/src/openvla && python /home/lch/Documents/dcvla/src/openvla/experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${ckpt} \
    --task_suite_name ${task} \
    --local_log_dir ${logdir} \
    --num_trials_per_task 20 \
    --num_tasks 10 \
    --task_start_id 0 \
    --vit_cache_standalone True \
    --use_vla_cache ${use_vla} \
    --use_vit_cache ${use_vit} \
    --vit_cache_reuse ${use_vit} \
    ${extra_flags} \
    > ${logfile} 2>&1"
}

run_suite_oft() {
  local task=$1
  local ckpt=$2
  local logdir=$3
  local use_vla=$4
  local use_vit=$5
  shift 5
  local extra_flags="$*"

  local logfile="$logdir/${task}.log"
  echo "  ↳ 任务 ${task} -> Log: ${logfile}"

  run_task "source /home/lch/miniconda3/etc/profile.d/conda.sh && conda activate dcvlaoft && cd /home/lch/Documents/dcvla/src/openvla-oft && python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${ckpt} \
    --task_suite_name ${task} \
    --local_log_dir ${logdir} \
    --num_trials_per_task 20 \
    --num_tasks 10 \
    --task_start_id 0 \
    --vit_cache_standalone True \
    --use_vla_cache ${use_vla} \
    --use_vit_cache ${use_vit} \
    --vit_cache_reuse ${use_vit} \
    ${extra_flags} \
    > ${logfile} 2>&1"
}

run_group_openvla() {
  local tag=$1
  local extra_flags="$2"
  local use_vla=$3
  local use_vit=$4
  local logdir="${BASE_PATH}/${ts}_${tag}"
  mkdir -p "$logdir"
  echo "🚀 启动参数组 ${tag} -> Log: $logdir"
  run_suite_openvla libero_spatial checkpoints/openvla-7b-finetuned-libero-spatial "$logdir" "$use_vla" "$use_vit" "$extra_flags"
  run_suite_openvla libero_object  checkpoints/openvla-7b-finetuned-libero-object  "$logdir" "$use_vla" "$use_vit" "$extra_flags"
  run_suite_openvla libero_goal    checkpoints/openvla-7b-finetuned-libero-goal    "$logdir" "$use_vla" "$use_vit" "$extra_flags"
  run_suite_openvla libero_10      checkpoints/openvla-7b-finetuned-libero-10      "$logdir" "$use_vla" "$use_vit" "$extra_flags"
}

run_group_oft() {
  local tag=$1
  local extra_flags="$2"
  local use_vla=$3
  local use_vit=$4
  local logdir="${BASE_PATH}/${ts}_${tag}"
  mkdir -p "$logdir"
  echo "🚀 启动参数组 ${tag} -> Log: $logdir"
  run_suite_oft libero_spatial checkpoints/openvla-7b-oft-finetuned-libero-spatial "$logdir" "$use_vla" "$use_vit" "$extra_flags"
  run_suite_oft libero_object  checkpoints/openvla-7b-oft-finetuned-libero-object  "$logdir" "$use_vla" "$use_vit" "$extra_flags"
  run_suite_oft libero_goal    checkpoints/openvla-7b-oft-finetuned-libero-goal    "$logdir" "$use_vla" "$use_vit" "$extra_flags"
  run_suite_oft libero_10      checkpoints/openvla-7b-oft-finetuned-libero-10      "$logdir" "$use_vla" "$use_vit" "$extra_flags"
}

# -------------------------
# OpenVLA-OFT: switch ablation (one param set)
# g25: gray 0.004, attn 160, static 120, kf 5
# -------------------------
OFT_BASE_FLAGS="--vit_cache_patch_metric gray_diff --vit_cache_gray_diff_threshold 0.004 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 120 --vit_cache_keyframe_interval 5"
# run_group_oft "oft_switch_vla1_vit0" "$OFT_BASE_FLAGS" True False
run_group_oft "oft_switch_vla0_vit1" "$OFT_BASE_FLAGS" False True
run_group_oft "oft_switch_vla1_vit1" "$OFT_BASE_FLAGS" True True

# -------------------------
# OpenVLA-OFT: mask inconsistency (one param set)
# ViT-only change and LLM-only change
# -------------------------
OFT_INCONSIST_VIT_FLAGS="--vit_cache_patch_metric gray_diff --vit_cache_gray_diff_threshold 0.006 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 120 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric gray_diff --llm_cache_gray_diff_threshold 0.004 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 120"
OFT_INCONSIST_LLM_FLAGS="--vit_cache_patch_metric gray_diff --vit_cache_gray_diff_threshold 0.004 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 120 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric gray_diff --llm_cache_gray_diff_threshold 0.006 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 120"
run_group_oft "oft_mask_inconsistent_vit_hi" "$OFT_INCONSIST_VIT_FLAGS" True True
run_group_oft "oft_mask_inconsistent_llm_hi" "$OFT_INCONSIST_LLM_FLAGS" True True

# -------------------------
# OpenVLA: switch ablation (one param set)
# g15: cos 0.992, attn 160, static 160, kf 5
# -------------------------
OPENVLA_BASE_FLAGS="--vit_cache_patch_metric cosine --vit_cache_sim_threshold 0.992 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 160 --vit_cache_keyframe_interval 5"
run_group_openvla "ovla_switch_vla1_vit0" "$OPENVLA_BASE_FLAGS" True False
run_group_openvla "ovla_switch_vla0_vit1" "$OPENVLA_BASE_FLAGS" False True
run_group_openvla "ovla_switch_vla1_vit1" "$OPENVLA_BASE_FLAGS" True True

# -------------------------
# OpenVLA: mask inconsistency (one param set)
# ViT-only change and LLM-only change
# -------------------------
OPENVLA_INCONSIST_VIT_FLAGS="--vit_cache_patch_metric cosine --vit_cache_sim_threshold 0.995 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 160 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric cosine --llm_cache_sim_threshold 0.992 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 160"
OPENVLA_INCONSIST_LLM_FLAGS="--vit_cache_patch_metric cosine --vit_cache_sim_threshold 0.992 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 160 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric cosine --llm_cache_sim_threshold 0.995 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 160"
run_group_openvla "ovla_mask_inconsistent_vit_hi" "$OPENVLA_INCONSIST_VIT_FLAGS" True True
run_group_openvla "ovla_mask_inconsistent_llm_hi" "$OPENVLA_INCONSIST_LLM_FLAGS" True True
