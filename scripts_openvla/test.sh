# 获取当前执行时间（格式：YYYYmmdd_HHMMSS）
ts=$(date +"%Y%m%d_%H%M%S")

# 原始基路径
base="/home/lch/Documents/dcvla/results/"

# 生成带时间戳的新路径
logdir="${base}/${ts}_spatial_keyframe_interval_5_test"

# 创建目录
mkdir -p "$logdir"

logfile="${logdir}/vlaTrue_vitTrue.log"
CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 2 \
  --num_tasks 1 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_keyframe_interval 5  > "$logfile" 2>&1 &
  # --vit_cache_drop_k 30 \
