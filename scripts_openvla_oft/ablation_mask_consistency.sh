#!/usr/bin/env bash
# OpenVLA-OFT ablation: mask consistency vs inconsistency (ViT vs LLM params).

set -e

ts=$(date +"%Y%m%d_%H%M%S")
BASE_PATH="/home/lch/Documents/dcvla/results_oft/ablation_mask_consistency"

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
  echo 3000
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

run_suite() {
  local task=$1
  local ckpt=$2
  local logdir=$3
  local vit_gray=$4
  local vit_attn=$5
  local vit_static=$6
  local kf=$7
  local llm_gray=$8
  local llm_attn=$9
  local llm_static=${10}

  local logfile="$logdir/${task}.log"
  echo "  ↳ 任务 ${task} -> Log: ${logfile}"

  run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${ckpt} \
    --task_suite_name ${task} \
    --local_log_dir ${logdir} \
    --num_trials_per_task 5 \
    --num_tasks 10 \
    --task_start_id 0 \
    --vit_cache_standalone True \
    --use_vla_cache True \
    --use_vit_cache True \
    --vit_cache_reuse True \
    --vit_cache_patch_metric gray_diff \
    --vit_cache_gray_diff_threshold ${vit_gray} \
    --vit_cache_attention_top_k ${vit_attn} \
    --vit_cache_static_top_k ${vit_static} \
    --vit_cache_keyframe_interval ${kf} \
    --llm_cache_patch_metric gray_diff \
    --llm_cache_gray_diff_threshold ${llm_gray} \
    --llm_cache_attention_top_k ${llm_attn} \
    --llm_cache_static_top_k ${llm_static} \
    > ${logfile} 2>&1"
}

run_group() {
  local tag=$1
  local vit_gray=$2
  local vit_attn=$3
  local vit_static=$4
  local kf=$5
  local llm_gray=$6
  local llm_attn=$7
  local llm_static=$8

  local logdir="${BASE_PATH}/${ts}_${tag}"
  mkdir -p "$logdir"
  echo "🚀 启动参数组 ${tag} | vit(gray=${vit_gray}, attn=${vit_attn}, static=${vit_static}, kf=${kf}) | llm(gray=${llm_gray}, attn=${llm_attn}, static=${llm_static}) -> Log: $logdir"

  run_suite libero_spatial checkpoints/openvla-7b-oft-finetuned-libero-spatial "$logdir" "$vit_gray" "$vit_attn" "$vit_static" "$kf" "$llm_gray" "$llm_attn" "$llm_static"
  run_suite libero_object  checkpoints/openvla-7b-oft-finetuned-libero-object  "$logdir" "$vit_gray" "$vit_attn" "$vit_static" "$kf" "$llm_gray" "$llm_attn" "$llm_static"
  run_suite libero_goal    checkpoints/openvla-7b-oft-finetuned-libero-goal    "$logdir" "$vit_gray" "$vit_attn" "$vit_static" "$kf" "$llm_gray" "$llm_attn" "$llm_static"
  run_suite libero_10      checkpoints/openvla-7b-oft-finetuned-libero-10      "$logdir" "$vit_gray" "$vit_attn" "$vit_static" "$kf" "$llm_gray" "$llm_attn" "$llm_static"
}

# Baseline (consistent)
run_group "c00_consistent" 0.004 160 120 5 0.004 160 120

# ViT-only perturbations (LLM fixed)
run_group "v01_vit_gray_hi" 0.006 160 120 5 0.004 160 120
run_group "v02_vit_gray_lo" 0.002 160 120 5 0.004 160 120
run_group "v03_vit_attn_lo" 0.004 120 120 5 0.004 160 120
run_group "v04_vit_attn_hi" 0.004 200 120 5 0.004 160 120

# LLM-only perturbations (ViT fixed)
run_group "l01_llm_gray_hi" 0.004 160 120 5 0.006 160 120
run_group "l02_llm_gray_lo" 0.004 160 120 5 0.002 160 120
run_group "l03_llm_attn_lo" 0.004 160 120 5 0.004 120 120
run_group "l04_llm_attn_hi" 0.004 160 120 5 0.004 200 120
