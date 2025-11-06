```bash
mamba activate dcvla
cd /home/lch/Documents/dcvla/src/openvla
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --num_tasks 1 \
  --task_start_id 0 \
  --use_vla_cache True \
  --use_vit_cache True

CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --num_tasks 1 \
  --use_vla_cache True \
  --use_vit_cache True \
  --local_log_dir "/home/lch/Documents/dcvla/results/vit_cache_ttf_test"  \
  --task_start_id 10 > "/home/lch/Documents/dcvla/results/vit_cache_ttf_test/output10.log" 2>&1 &
```



# 添加参数配置总结

### 1. 新增的参数（第78-79行）

num_tasks: Optional[int] = None                  # Number of tasks to 
evaluate from the suite. If None, all tasks are evaluated.
task_start_id: int = 0                           # Starting task ID 
(0-indexed) for evaluation

### 2. 修改的推理逻辑（第149-168行）

添加了任务范围计算和验证：
- 计算要评估的任务结束ID
- 验证任务范围的有效性
- 输出评估信息
- 修改循环为 range(cfg.task_start_id, task_end_id)

### 使用示例

#### 示例 1: 评估所有任务（默认行为）

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint
checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50

#### 示例 2: 只评估第一个任务，每个任务执行10次

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint
checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 10 \
  --num_tasks 1 \
  --task_start_id 0

#### 示例 3: 从第2个任务开始，评估3个任务，每个任务执行5次

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint
checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 5 \
  --num_tasks 3 \
  --task_start_id 2
这将评估任务ID 2, 3, 4（总共3个任务）

#### 示例 4: 快速测试（1个任务，1次试验）

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint
checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --num_tasks 1 \
  --task_start_id 0 \
  --use_vla_cache True

### 功能特性

1. 灵活的任务选择: 可以从任意任务ID开始评估
2. 可控的任务数量: 可以指定评估多少个任务
3. 自动边界检查: 确保不会超出可用任务范围
4. 详细的日志输出: 显示正在评估的任务范围和次数
5. 向后兼容: 不指定新参数时，行为与原来完全一致（评估所有任务）