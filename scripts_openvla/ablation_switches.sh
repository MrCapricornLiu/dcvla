#!/usr/bin/env bash
# OpenVLA ablation: VLA/Vit cache on/off combinations across 5 parameter sets.

set -e

ts=$(date +"%Y%m%d_%H%M%S")
BASE_PATH="/home/lch/Documents/dcvla/results/ablation_switches"

CHECK_INTERVAL=10
COOLDOWN=80
WAIT_LOG_INTERVAL=60
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
  echo 20000
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
  local use_vla=$4
  local use_vit=$5
  local metric=$6
  local sim=$7
  local gray=$8
  local attn=$9
  local static_top_k=${10}
  local kf=${11}

  local logfile="$logdir/${task}.log"
  echo "  ↳ 任务 ${task} -> Log: ${logfile}"

  local metric_flags="--vit_cache_patch_metric ${metric}"
  if [ "$metric" = "cosine" ]; then
    metric_flags+=" --vit_cache_sim_threshold ${sim}"
  else
    metric_flags+=" --vit_cache_gray_diff_threshold ${gray}"
  fi

  run_task "python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${ckpt} \
    --task_suite_name ${task} \
    --local_log_dir ${logdir} \
    --num_trials_per_task 5 \
    --num_tasks 10 \
    --task_start_id 0 \
    --vit_cache_standalone True \
    --use_vla_cache ${use_vla} \
    --use_vit_cache ${use_vit} \
    --vit_cache_reuse ${use_vit} \
    ${metric_flags} \
    --vit_cache_attention_top_k ${attn} \
    --vit_cache_static_top_k ${static_top_k} \
    --vit_cache_keyframe_interval ${kf} \
    > ${logfile} 2>&1"
}

run_group() {
  local tag=$1
  local metric=$2
  local sim=$3
  local gray=$4
  local attn=$5
  local static_top_k=$6
  local kf=$7

  for combo in "vla1_vit0" "vla0_vit1" "vla1_vit1"; do
    local use_vla="True"
    local use_vit="True"
    if [ "$combo" = "vla1_vit0" ]; then
      use_vit="False"
    elif [ "$combo" = "vla0_vit1" ]; then
      use_vla="False"
    fi

    local logdir="${BASE_PATH}/${ts}_${tag}_${combo}"
    mkdir -p "$logdir"
    echo "🚀 启动参数组 ${tag} | ${combo} | metric=${metric}, sim=${sim}, gray=${gray}, attn_topk=${attn}, static_topk=${static_top_k}, kf=${kf} -> Log: $logdir"

    run_suite libero_spatial checkpoints/openvla-7b-finetuned-libero-spatial "$logdir" "$use_vla" "$use_vit" "$metric" "$sim" "$gray" "$attn" "$static_top_k" "$kf"
    run_suite libero_object  checkpoints/openvla-7b-finetuned-libero-object  "$logdir" "$use_vla" "$use_vit" "$metric" "$sim" "$gray" "$attn" "$static_top_k" "$kf"
    run_suite libero_goal    checkpoints/openvla-7b-finetuned-libero-goal    "$logdir" "$use_vla" "$use_vit" "$metric" "$sim" "$gray" "$attn" "$static_top_k" "$kf"
    run_suite libero_10      checkpoints/openvla-7b-finetuned-libero-10      "$logdir" "$use_vla" "$use_vit" "$metric" "$sim" "$gray" "$attn" "$static_top_k" "$kf"
  done
}

# 5 parameter sets
run_group "p01_cos_0992_k160_s160_kf5" cosine 0.992 - 160 160 5
run_group "p02_cos_0995_k140_s120_kf5" cosine 0.995 - 140 120 5
run_group "p03_cos_0989_k200_s160_kf5" cosine 0.989 - 200 160 5
run_group "p04_cos_0993_k120_s160_kf8" cosine 0.993 - 120 160 8
run_group "p05_gray_003_k160_s120_kf5" gray_diff - 0.003 160 120 5
