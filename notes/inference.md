```bash
python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object --task_suite_name libero_object --run_id_note baseline-test --center_crop True
```

执行上面的代码之后，函数调用顺序如下：

```python
eval_libero in "experiments/robot/libero/run_libero_eval.py"
-> get_action in "experiments/robot/robot_utils.py"
-> get_vla_action in "experiments/robot/openvla_utils.py"
-> predict_action in "prismatic/extern/hf/modeling_prismatic.py"
-> prepare_inputs_for_generation in "prismatic/extern/hf/modeling_prismatic.py"
-> forward in "prismatic/extern/hf/modeling_prismatic.py"
-> PrismaticVisionBackbone.forward in "prismatic/extern/hf/modeling_prismatic.py"
```