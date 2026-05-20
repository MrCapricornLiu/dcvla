#!/usr/bin/env bash
set -e
ts=$(date +"%Y%m%d_%H%M%S")
BASE_PATH="/home/lch/Documents/dcvla/results"

logdir="${BASE_PATH}/${ts}_g01_cos_0996_k120_s140_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g01_cos_0996_k120_s140_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g01_cos_0996_k120_s140_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g01_cos_0996_k120_s140_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g01_cos_0996_k120_s140_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g01_cos_0996_k120_s140_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g01_cos_0996_k120_s140_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g01_cos_0996_k120_s140_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g02_cos_0997_k120_s140_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g02_cos_0997_k120_s140_kf5 | metric=cosine, sim=0.997, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g02_cos_0997_k120_s140_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g02_cos_0997_k120_s140_kf5 | metric=cosine, sim=0.997, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g02_cos_0997_k120_s140_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g02_cos_0997_k120_s140_kf5 | metric=cosine, sim=0.997, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g02_cos_0997_k120_s140_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g02_cos_0997_k120_s140_kf5 | metric=cosine, sim=0.997, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g03_cos_0994_k140_s160_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g03_cos_0994_k140_s160_kf5 | metric=cosine, sim=0.994, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g03_cos_0994_k140_s160_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g03_cos_0994_k140_s160_kf5 | metric=cosine, sim=0.994, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g03_cos_0994_k140_s160_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g03_cos_0994_k140_s160_kf5 | metric=cosine, sim=0.994, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g03_cos_0994_k140_s160_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g03_cos_0994_k140_s160_kf5 | metric=cosine, sim=0.994, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g04_cos_0996_k160_s180_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g04_cos_0996_k160_s180_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g04_cos_0996_k160_s180_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g04_cos_0996_k160_s180_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g04_cos_0996_k160_s180_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g04_cos_0996_k160_s180_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g04_cos_0996_k160_s180_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g04_cos_0996_k160_s180_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g05_cos_0996_k120_s140_kf3_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g05_cos_0996_k120_s140_kf3 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g05_cos_0996_k120_s140_kf3_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g05_cos_0996_k120_s140_kf3 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g05_cos_0996_k120_s140_kf3_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g05_cos_0996_k120_s140_kf3 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g05_cos_0996_k120_s140_kf3_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g05_cos_0996_k120_s140_kf3 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g06_cos_0996_k120_s140_kf8_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g06_cos_0996_k120_s140_kf8 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g06_cos_0996_k120_s140_kf8_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g06_cos_0996_k120_s140_kf8 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g06_cos_0996_k120_s140_kf8_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g06_cos_0996_k120_s140_kf8 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g06_cos_0996_k120_s140_kf8_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g06_cos_0996_k120_s140_kf8 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g07_cos_0995_k100_s120_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g07_cos_0995_k100_s120_kf5 | metric=cosine, sim=0.995, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g07_cos_0995_k100_s120_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g07_cos_0995_k100_s120_kf5 | metric=cosine, sim=0.995, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g07_cos_0995_k100_s120_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g07_cos_0995_k100_s120_kf5 | metric=cosine, sim=0.995, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g07_cos_0995_k100_s120_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g07_cos_0995_k100_s120_kf5 | metric=cosine, sim=0.995, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g08_cos_0993_k100_s120_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g08_cos_0993_k100_s120_kf5 | metric=cosine, sim=0.993, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g08_cos_0993_k100_s120_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g08_cos_0993_k100_s120_kf5 | metric=cosine, sim=0.993, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g08_cos_0993_k100_s120_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g08_cos_0993_k100_s120_kf5 | metric=cosine, sim=0.993, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g08_cos_0993_k100_s120_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g08_cos_0993_k100_s120_kf5 | metric=cosine, sim=0.993, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g09_cos_0998_k80_s100_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g09_cos_0998_k80_s100_kf5 | metric=cosine, sim=0.998, gray=-, rgb=-, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g09_cos_0998_k80_s100_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g09_cos_0998_k80_s100_kf5 | metric=cosine, sim=0.998, gray=-, rgb=-, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g09_cos_0998_k80_s100_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g09_cos_0998_k80_s100_kf5 | metric=cosine, sim=0.998, gray=-, rgb=-, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g09_cos_0998_k80_s100_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g09_cos_0998_k80_s100_kf5 | metric=cosine, sim=0.998, gray=-, rgb=-, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g10_cos_0996_k140_s160_kf8_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g10_cos_0996_k140_s160_kf8 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g10_cos_0996_k140_s160_kf8_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g10_cos_0996_k140_s160_kf8 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g10_cos_0996_k140_s160_kf8_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g10_cos_0996_k140_s160_kf8 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g10_cos_0996_k140_s160_kf8_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g10_cos_0996_k140_s160_kf8 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g11_gray_003_k120_s140_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g11_gray_003_k120_s140_kf5 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g11_gray_003_k120_s140_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g11_gray_003_k120_s140_kf5 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g11_gray_003_k120_s140_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g11_gray_003_k120_s140_kf5 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g11_gray_003_k120_s140_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g11_gray_003_k120_s140_kf5 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g12_gray_002_k120_s140_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g12_gray_002_k120_s140_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g12_gray_002_k120_s140_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g12_gray_002_k120_s140_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g12_gray_002_k120_s140_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g12_gray_002_k120_s140_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g12_gray_002_k120_s140_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g12_gray_002_k120_s140_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g13_gray_004_k140_s160_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g13_gray_004_k140_s160_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g13_gray_004_k140_s160_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g13_gray_004_k140_s160_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g13_gray_004_k140_s160_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g13_gray_004_k140_s160_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g13_gray_004_k140_s160_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g13_gray_004_k140_s160_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g14_gray_003_k160_s180_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g14_gray_003_k160_s180_kf5 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g14_gray_003_k160_s180_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g14_gray_003_k160_s180_kf5 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g14_gray_003_k160_s180_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g14_gray_003_k160_s180_kf5 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g14_gray_003_k160_s180_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g14_gray_003_k160_s180_kf5 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g15_gray_003_k120_s140_kf3_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g15_gray_003_k120_s140_kf3 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g15_gray_003_k120_s140_kf3_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g15_gray_003_k120_s140_kf3 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g15_gray_003_k120_s140_kf3_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g15_gray_003_k120_s140_kf3 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g15_gray_003_k120_s140_kf3_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g15_gray_003_k120_s140_kf3 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g16_gray_003_k120_s140_kf8_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g16_gray_003_k120_s140_kf8 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g16_gray_003_k120_s140_kf8_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g16_gray_003_k120_s140_kf8 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g16_gray_003_k120_s140_kf8_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g16_gray_003_k120_s140_kf8 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g16_gray_003_k120_s140_kf8_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g16_gray_003_k120_s140_kf8 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g17_gray_005_k100_s120_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g17_gray_005_k100_s120_kf5 | metric=gray_diff, sim=-, gray=0.05, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g17_gray_005_k100_s120_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g17_gray_005_k100_s120_kf5 | metric=gray_diff, sim=-, gray=0.05, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g17_gray_005_k100_s120_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g17_gray_005_k100_s120_kf5 | metric=gray_diff, sim=-, gray=0.05, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g17_gray_005_k100_s120_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g17_gray_005_k100_s120_kf5 | metric=gray_diff, sim=-, gray=0.05, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g18_gray_004_k100_s120_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g18_gray_004_k100_s120_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g18_gray_004_k100_s120_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g18_gray_004_k100_s120_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g18_gray_004_k100_s120_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g18_gray_004_k100_s120_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g18_gray_004_k100_s120_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g18_gray_004_k100_s120_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g19_gray_002_k80_s100_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g19_gray_002_k80_s100_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g19_gray_002_k80_s100_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g19_gray_002_k80_s100_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g19_gray_002_k80_s100_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g19_gray_002_k80_s100_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g19_gray_002_k80_s100_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g19_gray_002_k80_s100_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g20_gray_003_k140_s160_kf8_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g20_gray_003_k140_s160_kf8 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g20_gray_003_k140_s160_kf8_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g20_gray_003_k140_s160_kf8 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g20_gray_003_k140_s160_kf8_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g20_gray_003_k140_s160_kf8 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g20_gray_003_k140_s160_kf8_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g20_gray_003_k140_s160_kf8 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g21_rgb_005_k120_s140_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g21_rgb_005_k120_s140_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g21_rgb_005_k120_s140_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g21_rgb_005_k120_s140_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g21_rgb_005_k120_s140_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g21_rgb_005_k120_s140_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g21_rgb_005_k120_s140_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g21_rgb_005_k120_s140_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g22_rgb_004_k120_s140_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g22_rgb_004_k120_s140_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g22_rgb_004_k120_s140_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g22_rgb_004_k120_s140_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g22_rgb_004_k120_s140_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g22_rgb_004_k120_s140_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g22_rgb_004_k120_s140_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g22_rgb_004_k120_s140_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=120, static_topk=140, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g23_rgb_006_k140_s160_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g23_rgb_006_k140_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g23_rgb_006_k140_s160_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g23_rgb_006_k140_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g23_rgb_006_k140_s160_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g23_rgb_006_k140_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g23_rgb_006_k140_s160_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g23_rgb_006_k140_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g24_rgb_005_k160_s180_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g24_rgb_005_k160_s180_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g24_rgb_005_k160_s180_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g24_rgb_005_k160_s180_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g24_rgb_005_k160_s180_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g24_rgb_005_k160_s180_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g24_rgb_005_k160_s180_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g24_rgb_005_k160_s180_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=160, static_topk=180, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 180 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g25_rgb_005_k120_s140_kf3_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g25_rgb_005_k120_s140_kf3 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g25_rgb_005_k120_s140_kf3_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g25_rgb_005_k120_s140_kf3 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g25_rgb_005_k120_s140_kf3_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g25_rgb_005_k120_s140_kf3 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g25_rgb_005_k120_s140_kf3_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g25_rgb_005_k120_s140_kf3 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=3 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 3 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g26_rgb_005_k120_s140_kf8_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g26_rgb_005_k120_s140_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g26_rgb_005_k120_s140_kf8_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g26_rgb_005_k120_s140_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g26_rgb_005_k120_s140_kf8_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g26_rgb_005_k120_s140_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g26_rgb_005_k120_s140_kf8_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g26_rgb_005_k120_s140_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=120, static_topk=140, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 140 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g27_rgb_007_k100_s120_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g27_rgb_007_k100_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.07, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g27_rgb_007_k100_s120_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g27_rgb_007_k100_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.07, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g27_rgb_007_k100_s120_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g27_rgb_007_k100_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.07, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g27_rgb_007_k100_s120_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g27_rgb_007_k100_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.07, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g28_rgb_006_k100_s120_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g28_rgb_006_k100_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g28_rgb_006_k100_s120_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g28_rgb_006_k100_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g28_rgb_006_k100_s120_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g28_rgb_006_k100_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g28_rgb_006_k100_s120_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g28_rgb_006_k100_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g29_rgb_004_k80_s100_kf5_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g29_rgb_004_k80_s100_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g29_rgb_004_k80_s100_kf5_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g29_rgb_004_k80_s100_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g29_rgb_004_k80_s100_kf5_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g29_rgb_004_k80_s100_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g29_rgb_004_k80_s100_kf5_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g29_rgb_004_k80_s100_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=80, static_topk=100, kf=5 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 100 \
  --vit_cache_keyframe_interval 5 > "$logdir/libero_10.log" 2>&1

logdir="${BASE_PATH}/${ts}_g30_rgb_005_k140_s160_kf8_libero_spatial"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_spatial | g30_rgb_005_k140_s160_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_spatial.log" 2>&1

logdir="${BASE_PATH}/${ts}_g30_rgb_005_k140_s160_kf8_libero_object"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_object | g30_rgb_005_k140_s160_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_object.log" 2>&1

logdir="${BASE_PATH}/${ts}_g30_rgb_005_k140_s160_kf8_libero_goal"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_goal | g30_rgb_005_k140_s160_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_goal.log" 2>&1

logdir="${BASE_PATH}/${ts}_g30_rgb_005_k140_s160_kf8_libero_10"
mkdir -p "$logdir"
echo "🚀 启动任务 libero_10 | g30_rgb_005_k140_s160_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > "$logdir/libero_10.log" 2>&1
