# 获取当前执行时间（格式：YYYYmmdd_HHMMSS）
ts=$(date +"%Y%m%d_%H%M%S")

# 原始基路径
base="/home/lch/Documents/dcvla/results_remained_vlaonvitoff"

# 生成带时间戳的新路径
logdir="${base}/${ts}_vlaon_vitoff"

# 创建目录
mkdir -p "$logdir"



logfile="${logdir}/spatial.log"
CUDA_VISIBLE_DEVICES=0 nohup python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache False \
  --vit_cache_reuse False \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logfile" 2>&1 &

logfile="${logdir}/object.log"
CUDA_VISIBLE_DEVICES=1 nohup python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache False \
  --vit_cache_reuse False \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logfile" 2>&1 &

logfile="${logdir}/goal.log"
CUDA_VISIBLE_DEVICES=0  nohup python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache False \
  --vit_cache_reuse False \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logfile" 2>&1 &

logfile="${logdir}/10.log"
CUDA_VISIBLE_DEVICES=1  nohup python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache False \
  --vit_cache_reuse False \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > "$logfile" 2>&1 &