---
title: ViT Cache Reuse Implementation Plan
last_updated: 2025-11-05
---

# 1. 项目背景

DC-VLA 希望在保持 TTF（Token Temporal Fusion）带来的推理效果提升的同时，进一步减少 Vision Encoder 在连续帧上的重复计算。目前的时间一致性由 VLA-Cache 提供，但 Vision Transformer 每个 patch、每一层仍全部重算。目标是让静态补丁在各层完全复用上一帧的计算结果，而动态补丁继续重算，从而兼顾准确率与速度。

# 2. 当前代码现状（2025-11-05）

## 2.1 基线代码
- `src/pytorch-image-models`：回滚到原始 TIMM；VisionTransformer 没有任何缓存逻辑。
- `src/openvla`：原始 VLA-Cache + 我们新增的部分（详见 2.2）。
- `tools/compare_vit_cache_vs_ttf.py`：比对 TTF token 融合与 ViT cache reuse 的脚本。
- `notes/inference.md`：推理调用链文档。

## 2.2 现有新增逻辑
- `PrismaticVisionBackbone` 已实现临时缓存：  
  `set_reuse_mask(mask: Optional[Tensor])` 和 `reset_cache()`，按掩码把静态补丁替换为上一帧缓存，使当前输出与 TTF 融合完全一致。
- 对比脚本验证结果：  
  `Max abs diff (all) = 0`，说明复用逻辑与纯 TTF 融合等价。
- 尚未减少 FLOPs：当前静态补丁仍参与 Attention/MLP；仅在输出阶段替换 token。

# 3. 即将开展的工作

## 3.1 目标：实现真正的部分计算
1. **缓存结构**
   - 储存每层静态补丁的 hidden output 及 K/V；
   - 支持按照 token 索引读取/写入。
2. **VisionTransformer 裁剪**
   - 在 `forward_features` 的每个 block 中，拆分静态/动态补丁；
   - 动态补丁正常运行 Attention/MLP；
   - 静态补丁直接引用上一帧缓存，跳过计算；
   - Attention 中动态补丁仍看到静态补丁的 K/V。
3. **复用集合单调递增**
   - 仿照 LLaMA 的 `last_reusable_patches`，防止层间震荡。
4. **双视角支持**
   - 主视角与手腕视角分别缓存、分别复用。
5. **验证**
   - 使用比较脚本确保输出与 TTF 一致；
   - 测试成功率、延迟：确认“加速同时不降准确率”。

# 4. 调试与开发工具
- `tools/compare_vit_cache_vs_ttf.py`：验证复用行为与 TTF 一致。
- 调试日志可记录每层复用比例、补丁数量等，用于进一步确认行为是否符合预期。

# 5. 下一步建议
1. 编写详细实现方案（含伪代码、输入输出规范）。
2. 在 TIMM ViT 中按小步迭代：先缓存 hidden state，再逐层跳过静态补丁计算。
3. 每完成一小步就运行比较脚本和部分评测，便于快速定位问题。
4. 在确认实现无误后，评估是否要加上逐层递增 / 关键帧等稳态策略。

--- 
