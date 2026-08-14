# 探索性 Checkpoint 测试

`测试1106` 记录了早期 10/20/30 Epoch checkpoint 的提示词响应、生成质量与重绘结果，用于观察欠拟合、标签响应和过拟合趋势。它是训练策略探索材料；仓库中的最终复现实验配置仍为 16 Epoch，两者不可混为同一次实验。

## 测试任务

1. 混合标签是否同时被执行，空间关系是否真实；
2. 单一标签是否产生稳定、可辨认的视觉变化；
3. 增加其他提示词后，已训练标签是否仍保持有效；
4. 多轮训练是否出现构图固化、材质失真或标签过拟合。

## 代表性样例

混合提示词 `spatial_folding, street_spillout, low_transparency, limited_entrance`：

| 10 Epoch | 20 Epoch | 30 Epoch |
|---|---|---|
| [查看](../examples/checkpoint-tests/image-20.png) | [查看](../examples/checkpoint-tests/image-24.png) | [查看](../examples/checkpoint-tests/image-41.png) |

单标签 `street_spillout`：

| 10 Epoch | 20 Epoch | 30 Epoch |
|---|---|---|
| [查看](../examples/checkpoint-tests/image-52.png) | [查看](../examples/checkpoint-tests/image-28.png) | [查看](../examples/checkpoint-tests/image-54.png) |

## 评审口径

每组样例从提示词遵循度、空间真实性、色彩与材质合理性、设计可用性四项检查，并记录标签缺失、语义偏移、透视/结构畸变、构图重复与风格过拟合等 Bad Case。checkpoint 选择不应只看“更像训练图”，还应综合标签可控性与跨提示词泛化能力。
