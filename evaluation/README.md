# Evaluation

本目录提供项目评测的可复现框架：

- `test-prompts.csv`：根据 5 x 3 x 3 全因子设计重建的提示词矩阵模板；
- `scoring-rubric.md`：四项 5 分制人工评测口径；
- `evaluation-results-template.csv`：Base/LoRA 配对评分记录模板。

`test-prompts.csv` 保留了原研究的变量组合，但当前英文 Prompt 是依据标签体系整理的复现模板，不应表述为原实验逐字记录。公开前如能找到原始 Prompt、随机种子和逐样本评分，应以原始记录替换模板，并保留版本信息。

项目记录中的“提升 36%”按下式定义：

```text
(LoRA 四项综合均分 - Base 四项综合均分) / Base 四项综合均分 x 100%
```

在原始逐样本评分尚未补齐前，该数字应视为阶段性项目结果，而不是仓库内已经完全复现的实验结论。
