# 低活力图像增强实验

## 目的

验证在尽量保持既有空间形态的条件下，仅调整界面线索，能否提升街道活力感知。实验选取前序评价中均分最低的 6 张图像作为对照样本，统一使用图生图目标条件：

`street_vitality, street_spillout, medium_transparency, moderate_entrance`

该组合对应“街道外摆、中等界面通透度、中等沿街开口密度”，意在增加可停留、可参与的行为暗示，而非依赖人群或高饱和度制造热闹。

## 对照样本

| 组别 | 修改前条件 | 修改前均分 | 修改前 | 修改后 |
|---|---|---:|---|---|
| 1 | 无灰空间 / 中通透 / 开口多 | 3.00 | [查看](../examples/vitality-enhancement/image-10.png) | [查看](../examples/vitality-enhancement/image-3.png) |
| 2 | 无灰空间 / 高通透 / 开口少 | 3.00 | [查看](../examples/vitality-enhancement/image-11.png) | [查看](../examples/vitality-enhancement/image-6.png) |
| 3 | 界面退让 / 低通透 / 开口多 | 2.92 | [查看](../examples/vitality-enhancement/image-7.png) | [查看](../examples/vitality-enhancement/image-5.png) |
| 4 | 界面退让 / 低通透 / 开口少 | 2.62 | [查看](../examples/vitality-enhancement/image-1.png) | [查看](../examples/vitality-enhancement/image-4.png) |
| 5 | 空间折叠 / 低通透 / 开口少 | 2.50 | [查看](../examples/vitality-enhancement/image-9.png) | [查看](../examples/vitality-enhancement/image-2.png) |
| 6 | 无灰空间 / 低通透 / 开口少 | 1.67 | [查看](../examples/vitality-enhancement/image-0.png) | [查看](../examples/vitality-enhancement/image-8.png) |

## 观察结论与口径限制

项目记录显示，6 组修改后图像的活力感评价均有提升，支持“明确的停留线索能够改善活力感知”的方向性判断。当前材料未附修改后的逐图得分、样本量与显著性检验结果，因此公开表述采用“均有提升”，不直接宣称统计显著；补齐原始评分后可进一步报告均值差、置信区间与配对检验。
