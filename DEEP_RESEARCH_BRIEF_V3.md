# Gemini Deep Research 深度攻关委托书：面向高空红外弱小目标的 YOLO26 端到端检测（V3 破局方案）

> **使用方式**：可将本文件直接上传或全文粘贴至 Gemini Deep Research。文中所有源码细节、超参数与 15 组实测数据均来自真实 Colab Pro (NVIDIA L4, `ultralytics 8.4.218`) 日志，无任何估算或虚构。

---

## 0. 给 Deep Research 的核心任务指令

我目前正在准备一篇面向**传统订阅制（非 OA、0 版面费、非预警）SCI 期刊**的独立研究论文：
- **冲高目标（中科院 2/3 区）**：*Infrared Physics & Technology* (Elsevier)
- **稳发主力/保底（中科院 4 区）**：*Journal of Real-Time Image Processing (JRTIP)*、*Signal, Image and Video Processing (SIVP)*、*The Visual Computer* (Springer)

**硬性新颖性边界（必须严格规避 2026 年已发表竞品撞车）**：
- 已知近期同类文章：`CVDI-YOLO26`（SPIE Oct 2026，HIT-UAV + YOLO26，P2→P3 零初始化细节注入）、`YOLO26-P2-P4-SFAD`（SPIE Oct 2026）、`Aero-YOLO26`（Sensors Oct 2026）、*Frontiers in Marine Science* 2026（仅在 BboxLoss 中做尺度门控 MPDIoU-NWD）。
- **我们的差异化主轴**：聚焦 **YOLO26 原生 NMS-free 一对一（O2O）检测范式 + 无 DFL（`reg_max=1`）回归** 在高空密集红外弱小目标下的结构性瓶颈——在官方 HIT-UAV 测试集上，我们已实测出原版 `YOLO26n` 存在高达 **$\Delta_{\text{E2E}} = \text{mAP}_{\text{O2M}} - \text{mAP}_{\text{O2O}} = +2.32\%$ mAP50 / $+1.80\%$ mAP50-95** 的端到端精度折损（O2M+NMS 达 **89.98% / 59.30%**，而原生 O2O 仅 **87.67% / 57.51%**，折损约为 COCO 的 3 倍）。

我们已完成 **Phase 0（基线与源码审计）+ Phase 1 v1（3 组诊断）+ Phase 1 V2（9 组正式消融）共 15 组 200-epoch 完整实验**。V2 彻底解决了数值稳定性问题并验证了部分机制，但**在标准 `640×640` 测试集上，V2 各方案尚未稳定超越基线的总体 `Test mAP50 (87.67%)` 与 `Test mAP50-95 (57.51%)`**。

请你基于以下完整“实验病历”与源码机制，结合 2024–2026 年顶会/顶刊文献（CVPR/ICCV/ECCV/AAAI/TGRS/TCSVT/IPT/PR 等），帮我进行**深度文献调研与 V3 算法破局设计**，重点回答 **第 5 节的 5 个核心攻关问题**。

---

## 1. 实验环境、基准协议与已核实的 YOLO26 (`8.4.218`) 源码真相

### 1.1 数据集与统一评测协议
- **主数据集**：官方 **HIT-UAV**（`suojiashun/HIT-UAV-Infrared-Thermal-Dataset`），640×512 LWIR 长波红外航拍图，严格使用官方划分：**Train 2,029 / Val 290 / Test 579**（绝不重切 8:1:1）。
- **类别口径**：主实验采用标准 **4 有效类**（`Person, Car, Bicycle, OtherVehicle`，剔除标注噪声 `DontCare`；5 类保留作附录）。
  - 注意：`Test` 集共 4,780 个实例，其中 `OtherVehicle` 样本极少（仅约 ~30 个框），单框误检/漏检即可导致该类 AP 波动 ±3–8%，进而拉动 4 类宏平均 mAP50 波动 ±1–2%。
  - 实测视觉检查发现：HIT-UAV 存在显著的**昼间热反转（Daytime Thermal Inversion）**——正午阳光暴晒后的篮球场/沥青路面极亮，行人/骑行者呈现为比背景更暗的“冷斑”，与夜间“亮斑”极性完全相反。
- **硬件与软件**：Google Colab Pro `NVIDIA L4 (22.5 GB)`，`Python 3.13.16`，`PyTorch 2.14.1+cu130`，`ultralytics 8.4.218`。
- **训练默认超参**：`epochs=200, patience=50, batch=32, imgsz=640, close_mosaic=10, amp=True, seed=42, cache=True`（注意：目前沿用了 Ultralytics 默认增强 `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4, fliplr=0.5, flipud=0.0, mosaic=1.0, scale=0.5`，以及 Albumentations 默认 `Blur/MedianBlur/ToGray/CLAHE(p=0.01)`）。

### 1.2 从 `ultralytics 8.4.218` 实测提取的关键源码机制
1. **双头梯度隔离（`Detect.forward`）**：
   ```python
   preds = self.forward_head(x, **self.one2many)
   x_detach = [xi.detach() for xi in x] if self.training else x  # 关键！O2O 梯度不回传至 Backbone & Neck
   one2one = self.forward_head(x_detach, **self.one2one)
   ```
   **推论**：整个 Backbone、Neck（以及所有结构增强模块）**100% 仅由 `one2many` 分支的梯度驱动**；`one2one` 仅更新自身 6 个轻量 head 卷积层！
2. **渐进损失平衡 `E2ELoss`（ProgLoss）**：
   ```python
   self.one2many = v8DetectionLoss(model, tal_topk=10)
   self.one2one  = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
   # o2m 权重随 epoch 从 0.8 线性衰减到 0.1；o2o 权重从 0.2 线性上升到 0.9
   ```
3. **小目标分配器 `TaskAlignedAssigner`（STAL）**：
   - `select_candidates_in_gts`：若 GT 边长 `< stride_val`（P3-P5 模型 `stride_val = stride[1] = 16 px`），将其以中心放大至 `16×16` 扩大候选锚点池。
   - `select_highest_overlaps`：
     - 多 GT 冲突时，用 `overlaps.max(1)`（即纯 `CIoU`）仲裁归属；
     - O2O 分支随后用 `torch.topk(align_metric * mask_pos, topk2=1)` 选出唯一正样本；
   - `target_scores` 软分类标签由 `overlaps`（`pos_overlaps`）校准；而 `BboxLoss` 的回归权重 `weight = target_scores[fg_mask].sum(-1, keepdim=True)` **同样直接乘在 `loss_iou` 上**。
4. **无 DFL 回归 `BboxLoss`（`reg_max=1`）**：
   - `loss[0]` = `CIoU` 损失（`box=7.5`）；
   - `loss[2]` = 图像尺寸归一化后的 `ltrb` L1 损失（`dfl=1.5` 实际加权在 `l1_loss` 上，日志量级仅 ~0.002）。
5. **优化器实测真相**：
   - `optimizer="auto"` 在 HIT-UAV（`2029/64 × 200 = 6,340 < 10,000` iter）下实际选用 **`AdamW(lr=0.00125, momentum=0.9)`**，并非官方宣传的 `MuSGD`。
   - 我们实测了显式 `optimizer="MuSGD" (lr=0.01)` 与 `SGD (lr=0.01)`，两者均明显劣于 `AdamW(auto)`（见下表）。
6. **预训练迁移特性**：
   - 保持原生 `yolo26n.pt` 结构时，可迁移 **`708/708`** 权重，且 Ultralytics 自动按类名重映射 `3/4` 分类头权重（`Person, Car, Bicycle`）。

---

## 2. 全部 15 组实测实验总表（HIT-UAV 官方 4 类，seed=42）

> 注：`Val(rect)` 为 Ultralytics 训练结束对 `best.pt` 跑原生 `rect=True (640×512)` 验证分数；`Test O2O (640)` 和 `Test O2M (640)` 为 `eval_utils.py` 统一在 `rect=False (640×640)` 下测得的端到端（`nms=False`）与传统多对一（`nms=True`）测试集指标；`Rect50` 为测试集原生 `rect=True (640×512)` 的 `O2O mAP50`；`3cls` 为剔除极小样本 `OtherVehicle` 后三大主类（Person/Car/Bicycle）的 AP50 均值。

| # | 实验名 | 模块/设置 | Best Ep / 总Ep | Params(M) / GFLOPs | Val(rect) m50 / m95 | **Test O2O mAP50** | **Test O2O mAP50-95** | **Test O2M mAP50** | **Δ_E2E m50 / m95** | **Rect 640×512 m50** | Test P / R | 3cls 均分 | Person / Car / Bicycle / OtherVeh | 640 Total ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **`p0_yolo26n_musgd`** | **基线 (`AdamW auto`)** | **84 / 134** | **2.535 / 6.04** | 93.2 / 63.2 | **87.67** | **57.51** | **89.98** | **+2.32 / +1.80** | 87.47 | 89.0 / 80.9 | 93.11 | 92.81 / **97.01** / 89.50 / 71.34 | 2.40 |
| 2 | `p0_yolo26n_true_musgd` | 基线 (`MuSGD lr=0.01`) | 90 / 140 | 2.535 / 6.04 | 89.7 / 59.9 | 86.07 | 54.88 | 87.98 | +1.91 / +2.09 | 86.41 | 87.4 / 82.2 | 92.61 | 92.71 / 95.90 / 89.22 / 66.46 | 2.39 |
| 3 | `p0_yolo26n_sgd` | 基线 (`SGD lr=0.01`) | — | 2.535 / 6.04 | 92.2 / 60.9 | 85.04 | 55.06 | 86.93 | +1.89 / +1.42 | 85.90 | 86.8 / 81.4 | 92.64 | 92.96 / 95.09 / 89.87 / 62.22 | 2.51 |
| 4 | `p0_yolo11n` | NMS 基线 (SGD) | — | 2.606 / 6.57 | 91.3 / 60.9 | 86.67 | 57.08 | — | — | 86.73 | **90.9** / 82.2 | 92.35 | 91.62 / 94.98 / 90.44 / 69.65 | 3.28 |
| 5 | `p0_yolov8n` | NMS 基线 (SGD) | — | 3.139 / 8.75 | 89.4 / 59.5 | 85.35 | 55.24 | — | — | 85.44 | 85.7 / 81.4 | 92.85 | 91.38 / 96.73 / 90.44 / 62.87 | 3.03 |
| 6 | `p1_mod_a` (v1负对照) | 裁P5换P2-P4 (355/732) | — | 1.810 / 7.17 | 87.7 / 58.2 | 84.16 | 53.79 | 87.12 | +2.96 / +1.66 | 83.96 | 82.7 / 80.2 | 91.80 | 90.59 / 95.80 / 89.00 / 61.25 | 3.26 |
| 7 | `p1_mod_b` (v1) | Stage0+1 Max/Min/Avg | — | 2.537 / — | 90.4 / 60.1 | 87.07 | 55.79 | 87.36 | +0.29 / +0.17 | 87.43 | 88.3 / 83.6 | 92.58 | 92.55 / 95.36 / 89.81 / 70.55 | 3.44 |
| 8 | `p1_mod_c1` (v1崩盘) | 耦合 NWD 进 `overlaps` | Ep19崩盘→Ep28 NaN | 2.535 / 6.04 | — | — | — | — | — | — | — | — | — | — |
| 9 | `p1v2_c1` | V2 FP32 解耦 `DSG-Assign` | 150 / 200 | 2.535 / 6.04 | 92.7 / 62.5 | 86.17 | 56.04 | 86.77 | +0.60 / +0.25 | 86.64 | 87.2 / **84.2** | 92.59 | 92.72 / 94.47 / **90.58** / 66.94 | **2.38** |
| 10 | `p1v2_c2` | V2 FP32 `DSG-Loss` | 94 / 144 | 2.535 / 6.04 | 93.3 / 63.0 | 86.47 | 55.33 | 87.71 | +1.24 / +1.36 | 87.26 | 86.5 / 83.1 | 92.32 | **93.59** / 94.90 / 88.46 / 68.92 | **2.37** |
| 11 | `p1v2_c12` | V2 `DSG-Assign + Loss` | 132 / 182 | 2.535 / 6.04 | 92.8 / 63.0 | 85.69 | 54.97 | 86.53 | +0.84 / **+0.14** | 87.19 | 85.1 / **84.3** | 92.24 | 92.27 / 94.67 / 89.78 / 66.03 | 2.40 |
| 12 | **`p1v2_bitc`** | V2 Stage1 `BiTC-Stem` | **161 / 200** | 2.536 / 6.07 | **93.2 / 64.1** | **87.45** | **56.66** | 87.19 | **-0.27 / -0.40** | 87.42 | **89.5 / 83.6** | 92.32 | 92.64 / 94.96 / 89.35 / **72.86** | 2.65 |
| 13 | **`p1v2_tcgate`** | V2 Neck `TC-Gate(P3/P4)` | **55 / 105** | 2.536 / 6.04 | 90.5 / 61.9 | 87.13 | 55.54 | **88.07** | +0.94 / +1.58 | **88.12** | 87.3 / 80.9 | 92.31 | 92.34 / 95.89 / 88.71 / 71.58 | 2.62 |
| 14 | `p1v2_bitc_tcgate` | V2 `BiTC + TC-Gate` | 84 / 134 | 2.536 / 6.07 | 91.2 / 61.8 | 86.64 | 55.68 | 87.60 | +0.95 / +0.57 | 87.23 | 85.8 / 82.5 | 92.89 | 92.78 / 96.83 / 89.06 / 67.91 | 2.71 |
| 15 | `p1v2_full_c1` | V2 `BiTC + TC-Gate + C1` | 95 / 145 | 2.536 / 6.07 | **93.6 / 62.8** | 85.02 | 54.73 | 85.76 | +0.74 / +0.98 | 85.62 | 84.5 / 79.4 | 92.25 | 92.29 / 94.80 / 89.67 / 63.34 | 2.75 |
| 16 | `p1v2_full_c12` | V2 `BiTC + TC-Gate + C12` | 90 / 140 | 2.536 / 6.07 | **93.4 / 64.1** | 84.96 | 54.77 | 86.15 | +1.19 / +0.89 | 86.28 | 83.8 / 82.0 | **93.21** | 92.87 / 96.60 / 90.17 / 60.20 | 2.67 |

---

## 3. V2 当前三个模块的精确数学定义与实现

1. **Mod B_v2 (`BiTC-Stem`，包裹 Stage 1 `Conv(16→32, s=2)`)**：
   $$C(x) = |\text{AvgPool}_{3\times3, s=2}(x) - \text{AvgPool}_{5\times5, s=2}(x)|,\quad y = \text{BaseConv}(x) + \tanh(g)\odot \text{SiLU}(\text{BN}(\text{Conv}_{1\times1}(C(x))))$$
   其中 $g\in\mathbb{R}^{32}$ 严格初始化为 $0$。训练后实测 `mean |tanh(g)| = 0.113`，范围 `[-0.288, +0.236]`。
2. **Mod A_v2 (`TC-Gate`，包裹 Neck P3 `#16` 与 P4 `#19`)**：
   $$D = |x - \text{AvgPool}_{3\times3}(x)|,\quad M = \sigma(\text{Conv}_{3\times3}([\text{Mean}_c(D), \text{Max}_c(D)])),\quad y = x + \tanh(g)\odot(x \odot M)$$
   其中 $g$ 初始化为 $0$。训练后实测 P3 `mean |tanh(g)| = 0.047`、P4 `0.087~0.111`（以负值为主，`min = -0.319`，说明在执行空间杂波负向抑制）。
3. **Mod C1/C2 v2 (`DSG-NWD`，全程 FP32，同时注入 `one2many` 与 `one2one`)**：
   - NWD 相似度：$\text{NWD} = \exp\left(-\sqrt{(g_x-p_x)^2 + (g_y-p_y)^2 + \frac{1}{4}((g_w-p_w)^2 + (g_h-p_h)^2)} / C\right)$，当前超参 $C=8.0, A_0=1024\text{ px}^2, \tau=256, w_{\max}=0.25$，门控 $\omega = w_{\max}\cdot\sigma((A_0 - A_{\text{gt}})/\tau)$。
   - **C1 (`DSG-Assign`)**：重写 `TaskAlignedAssigner.get_box_metrics`：`align_metric` 使用 $u_{\text{match}} = (1-\omega)\text{CIoU} + \omega\text{NWD}$；返回给 `target_scores` 的 `overlaps` 保持纯 `CIoU`（根治了 v1 第 19–28 轮 O2O 置信度爆炸崩盘）。
   - **C2 (`DSG-Loss`)**：重写 `BboxLoss.forward`：将 `loss_iou` 的 `CIoU` 替换为 $(1-\omega)\text{CIoU} + \omega\text{NWD}$；保留原始 `reg_max=1` 归一化 L1 分支不变。

---

## 4. 我们已完成的一线深度“尸检”诊断（请基于此进一步深挖与纠偏）

经过对 15 组日志与源码的逐行核查，我们识别出 **5 个直接解释 V2 未超过基线的硬核病灶**：

1. **假性“Δ_E2E 缩小”陷阱（核心警醒！）**：
   - 基线的 `Test O2M` 天花板高达 **`89.98% / 59.30%`**（`O2O = 87.67% / 57.51%`）。
   - V2 各组虽然把 `Δ_E2E` 压到了 `-0.27% ~ +1.24%`，但 **O2M 天花板同步跌到了 `85.76% ~ 88.07%`**！也就是说，我们主要是“拉低了 O2M 天花板”，而没有“保住 90% 骨干特征并把 O2O 真正拉到 89%+”。
   - **直接嫌疑机制**：
     - (a) 我们将 `C1/C2` **同时 patch 进了 `one2many` 和 `one2one`**！由于 `Detect.forward` 对 O2O 输入做了 `x.detach()`，`one2many` 是骨干/颈部唯一的监督老师；改动 `one2many` 直接损伤了 90% 的骨干特征质量！
     - (b) `p1v2_bitc` (Ep161)、`p1v2_c1` (Ep150) 的 `best.pt` 落在很晚的轮次，此时 `E2ELoss.o2m` 已从 `0.8` 衰减到 `0.23–0.27`，导致 O2M 头衰减而对 290 张 Val 过拟合（`Val mAP50-95` 冲到 `64.1%`，Test 反而跌至 `56.66%`）。
2. **`DSG-Assign (C1)` 隐藏的“死锚点权重归零”与 `mAP50-95` 退化**：
   - 在 `TaskAlignedAssigner.select_highest_overlaps` 中，多目标冲突仍用纯 `CIoU (overlaps)` 仲裁，仅 `topk2=1` 用 `align_metric (u_match)`。
   - 当 `C1` 为小目标选出一个 `CIoU ≈ 0` 但 `NWD > 0` 的唯一 O2O 正样本时，因 `overlaps` 保持纯 `CIoU ≈ 0`，其 `target_scores ≈ 0`，导致 `BboxLoss` 里的回归权重 `weight = target_scores.sum(-1)` 也近似为 **`0`**——这个正样本白占了 `topk2=1` 名额却**没有回归梯度**！
   - 而在高 IoU 阶段（`IoU 0.7–0.9`），NWD 偏好中心接近而非边界紧贴，稀释了精确边框排序，导致 `mAP50-95` 普降 `1.5%–2.5%`。
3. **`DSG-Loss (C2)` 误伤 `Car` 与高 IoU 回归**：
   - `p1v2_c2` 将极小目标 `Person` 推至全场最高 **`93.59% (+0.79%)`**，但 `Car` 从 `97.01%` 跌至 `94.90%`、`mAP50-95` 跌至 `55.33%`。
   - 原因：HIT-UAV 高空俯拍下大量 `Car` 面积也在 `600–1400 px²`，`$A_0=1024, \tau=256$` 仍给 `Car` 分配了 10%–20% 的 NWD 损失，削弱了 CIoU 的长宽比与精边界梯度；且 `reg_max=1` 的 L1 分支对大框/小框的梯度量级未按尺度重新校准。
4. **两个局部高通模块叠加的负拮抗 + 灰边（Letterbox）对比度伪影**：
   - 单用 `BiTC-Stem`（`Test 87.45%`, `Val m95=64.1%`, `P=89.5%, R=83.6%, OtherVeh=72.86%`）与单用 `TC-Gate`（**原生 `640×512` 测出全场最高 `88.12% (+0.65%)`**）各自表现很强；
   - 但两者都基于局部均值差分 `|x - AvgPool(x)|`，强行串联（`p1v2_bitc_tcgate`）在低信噪比红外图上造成两次高频噪点放大，Precision 暴跌 `-3.8%`；
   - `TC-Gate` 在 `rect=False`（640×640 上下加灰边）仅 `87.13%`，切换到无灰边 `rect=True (640×512)` 立刻暴涨 `+0.99%` 至 `88.12%`，说明灰边与图像交界处的剧烈灰度跳变欺骗了局部对比度门控。
5. **红外域训练协议错配 & `Val→Test` 漂移与 `OtherVehicle` 方差**：
   - `p1v2_full_c12` 的三大主类均值高达 **`93.21%`（全场 15 组实验第一！）**，但仅因 `OtherVehicle`（测试集 ~30 个框）从 `71.34%` 掉到 `60.20%`，整体 4 类 mAP50 被单类拉低 `-2.78%`。
   - 同时，当前训练仍沿用 RGB 默认增强（`hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4`），以及 `cache='ram'` 的非确定性警告，极可能加剧 2,029 张红外小数据集的过拟合与方差。

---

## 5. 请 Deep Research 重点调研并交付的 5 大核心问题

请基于 **2024–2026 年真实可核验的顶会/顶刊论文与开源实践**（附具体论文标题、年份、会议/期刊及核心公式），给出**反“画大饼”、可直接改代码落地的 V3 破局方案**：

### 问题 1：如何在**不损伤 `one2many` 90% 天花板、不掉 `mAP50-95`** 的前提下，真正把 YOLO26/YOLOv10 的 **O2O 精度拉向 89%–90%**？
- 具体想调研：
  1. 针对双头 NMS-free YOLO（`x.detach()` + `E2ELoss`），学界/开源界在 **O2M→O2O 一致性对齐（Consistent Dual Assignments 改进、O2M-to-O2O 特征/Logit/IoU 蒸馏、Hybrid/Auxiliary Query Groups、只改 `one2one` 不动 `one2many`、改进 `ProgLoss` 权重曲线适配小数据集早停）** 上有哪些经过严格验证的高胜率方法？
  2. 如何解决我们发现的 **`C1` 死锚点问题**（既避免 v1 的 `IoU=0` 置信度爆炸崩盘，又不让选出的小目标唯一正样本在 `BboxLoss` 中因 `target_scores=0` 而丧失回归梯度）？比如：回归权重与分类软标签进一步解耦、或者**训练期退火（Early epochs 用 NWD 救回小目标匹配，后期平滑退火回纯 CIoU 保 `mAP50-95`）**、或者 **Inner-CIoU / Focaler-IoU / Wise-IoU / Shape-IoU / Scale-calibrated L1 (`reg_max=1`)** 在 YOLO26 无 DFL 回归上的最新实证表现如何？

### 问题 2：面向**小样本（~2,000 张）热红外弱小目标 + 完整保留 COCO 预训练权重 (`708/708`)**，哪类轻量化结构改动真正能稳定同时提升 `mAP50` 和 `mAP50-95`（且与 `BiTC-Stem` / `TC-Gate` 正交不拮抗）？
- 我们已实证：
  - 破坏预训练剪 P5 加从零 P2 头是灾难（`-3.51%`）；
  - 单纯叠加两个局部高通差分模块（`BiTC-Stem + TC-Gate`）会互相放大红外噪声；
  - 保留 `708/708` 预训练 + 零初始化残差是正确方向，但需要**互补而非重复的归纳偏置**（例如：一个做浅层双向热对比度保真，另一个应该做什么？是频率域/Haar小波低高频解耦、多尺度空洞/条带上下文抑制杂波、还是 P3 跨层低开销特征融合但不撞车 `CVDI-YOLO26` 的 P2→P3 注入？）。
- 请给出 **2–3 个与 `CVDI-YOLO26` / `YOLO26-P2-P4-SFAD` / `Aero-YOLO26` 100% 区分、参数增量 <5%、延迟增量 <0.3ms、能同时提升 `mAP50` 和 `mAP50-95` 的结构候选**。

### 问题 3：面向 **LWIR 红外航拍小数据集（HIT-UAV 2,029 张）**，有哪些经过验证的**红外专属训练配方（数据增强、正则化、不平衡极小类别 `OtherVehicle` 处理、`rect` 训练/评测对齐）**？
- 例如：
  - 关闭 `hsv_h / hsv_s`、调整 `hsv_v` 与 `erasing=0.0`、引入**热极性反转增强（Thermal Polarity Inversion / 热对比度抖动）**或 **Copy-Paste 针对小目标与稀缺类 `OtherVehicle`**、**EMA / Cosine LR / `patience` / `ProgLoss` 周期调整**，在红外目标检测论文中是否被证明能显著缩小 Val-Test Gap 并提升稳定涨点？（注意：论文中所有训练配方优化必须公平地同时作用于 Baseline 或在消融中清晰剥离，确保审稿人挑不出毛病！）

### 问题 4：真实摸底 **HIT-UAV 官方划分（2029/290/579）** 上的 SOTA 真实水位与目标期刊录用门槛
- 请帮我核实 2024–2026 年发表在 *Infrared Physics & Technology*、*JRTIP*、*SIVP*、*TGRS*、*IEEE GRSL/JSTARS* 上使用 **HIT-UAV 官方划分** 的论文（务必甄别并排除乱切 8:1:1 或数据泄露刷 95%+ 的水文）：
  - 在 4 类（无 DontCare）与 5 类（含 DontCare）官方测试集上，轻量级模型（~2–4M 参数）做到多少 `Test mAP50`、`Test mAP50-95` 和 **相比原版 `YOLO26n` 净提升多少（例如 `+1.5% ~ +2.5%` mAP50、且 3-seed 配对显著、Δ_E2E 真正通过拉高 O2O 缩小）** 足以稳投 *JRTIP / SIVP*，达到多少可以冲刺 *Infrared Physics & Technology*？

### 问题 5：请综合输出一套**精简、高胜率、可直接编码落地的 V3 实验设计方案** + 第二红外数据集选型建议
- **V3 方案约束**：
  - 坚持“少而精、正交互补”原则（最多 2–3 个清晰支柱，每个支柱都直接对应一个已观测到的实证病灶）；
  - 明确每个支柱的**精确数学公式、挂载位置、预期解决的指标（mAP50 / mAP50-95 / Δ_E2E / 速度）以及潜在风险**；
  - 给出一份 6–8 组可控消融的优先级队列（先用 1 组基线 + 关键单模块快速验证，再跑全量 3-seed）。
- **第二红外数据集选型**：对比 `DroneVehicle-IR`（裁 100px 白边转水平框）、`IRSTD-1k`、`RGBT-Tiny`、`LLVIP`、`FLIR` 在验证“高空/远距红外小目标 + 端到端低延迟”故事上的优劣与公开下载可靠性，推荐最优 1 个。
