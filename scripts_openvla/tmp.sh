
base="/home/lch/Documents/dcvla/results/"


ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_long_baseline"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache False \
  --use_vit_cache True \
  --vit_cache_reuse False  > "${logdir}/baseline(vlaFalse_vitFalse).log" 2>&1 &



base="/home/lch/Documents/dcvla/results/"



ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_long_vitcache-gray0.03-keyframe3_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 3 \
  --vit_cache_patch_metric "gray_diff" \
  --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &


ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_goal_vitcache-gray0.03-keyframe3_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 3 \
  --vit_cache_patch_metric "gray_diff" \
  --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &


ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_object_vitcache-gray0.03-keyframe3_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 3 \
  --vit_cache_patch_metric "gray_diff" \
  --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &







ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_spatial_vitcache-gray0.03-keyframe3_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=x python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 3 \
  --vit_cache_patch_metric "gray_diff" \
  --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &






ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_long_vitcache-rgb0.03-keyframe3_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=x python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 3 \
  --vit_cache_patch_metric "rgb_diff" \
  --vit_cache_rgb_diff_threshold 0.05 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &


ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_goal_vitcache-rgb0.03-keyframe3_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=x python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 3 \
  --vit_cache_patch_metric "rgb_diff" \
  --vit_cache_rgb_diff_threshold 0.05 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &


ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_object_vitcache-rgb0.03-keyframe3_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=x python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 3 \
  --vit_cache_patch_metric "rgb_diff" \
  --vit_cache_rgb_diff_threshold 0.05 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &



ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_spatial_vitcache-rgb0.03-keyframe3_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=x python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 3 \
  --vit_cache_patch_metric "rgb_diff" \
  --vit_cache_rgb_diff_threshold 0.05 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &














ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_long_vitcache-gray0.03-keyframe5_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=x python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 5 \
  --vit_cache_patch_metric "gray_diff" \
  --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &


ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_goal_vitcache-gray0.03-keyframe5_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=x python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 5 \
  --vit_cache_patch_metric "gray_diff" \
  --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &


ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_object_vitcache-gray0.03-keyframe5_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=x python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 5 \
  --vit_cache_patch_metric "gray_diff" \
  --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &



ts=$(date +"%Y%m%d_%H%M%S")
logdir="${base}/${ts}_spatial_vitcache-gray0.03-keyframe5_llmcache-default"
mkdir -p "$logdir"
echo "   -> 日志目录: $logdir"
CUDA_VISIBLE_DEVICES=x python experiments/robot/libero/run_libero_eval.py \
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
  --vit_cache_keyframe_interval 5 \
  --vit_cache_patch_metric "gray_diff" \
  --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &