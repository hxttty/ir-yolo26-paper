# IR-YOLO26n (HIT-UAV) 实验进展与核心记忆存档

> 本文件作为跨 `/compact` 与跨窗口的唯一真实记忆源（Single Source of Truth），记录全部已验证源码细节、实测数据、排雷结论、Google Drive 文件 ID 与后续计划。

## 1. 论文定位与投稿策略
- **用途与硬约束**：评职称，严格投 **传统订阅制、0 元版面费、非预警 SCI 期刊**（绝不投高价 OA 或预警黑名单）。
  - 冲高（中科院 2/3 区）：*Infrared Physics & Technology*（Elsevier）
  - 稳发主力/保底（中科院 4 区）：*Journal of Real-Time Image Processing (JRTIP)*、*Signal, Image and Video Processing (SIVP)*、*The Visual Computer*（Springer）
- **核心差异化（零撞车 2026 年 10 月竞品）**：
  - 避开 `CVDI-YOLO26`（SPIE Oct 2026，HIT-UAV + YOLO26 做 P2→P3 零初始化细节注入）、`YOLO26-P2-P4-SFAD`、`Aero-YOLO26` 与 *Frontiers in Marine Science* 2026。
  - **主轴护城河**：诊断并修复 **YOLO26 原生无 NMS 一对一匹配（O2O）+ 无 DFL（`reg_max=1`）** 在高空密集红外弱小目标下的结构性失效（实测基线端到端折损 $\Delta_{\text{E2E}} = +2.32\%$ mAP50，近 COCO 的 3 倍），并首次揭示及数学修复**朴素 NWD 注入 `TaskAlignedAssigner` 引发的 O2O 置信度爆炸崩盘（Ep 19–28）**。

## 2. 工程环境与云端 ID 速查（随时一键拉取，无需重搜）
- **本地与 GitHub**：`/Users/xiaoyuhe/Documents/Gemini-home/ir-yolo26-paper` ↔ `https://github.com/hxttty/ir-yolo26-paper.git`（macOS 钥匙串已保存凭据，可直接 `git push`）。
- **Colab Pro 环境**：`NVIDIA L4 (22.5 GB VRAM / 53 GB RAM)`，`Python 3.13.16`，`PyTorch 2.14.1+cu130`，`ultralytics 8.4.218`。
- **Google Drive 插件（`gdrive`，账号 `hxy.new@gmail.com`）直达 ID**：
  - 项目根目录 `ir-yolo26-paper/`：`1JCEzpmRCHUnWOEdfaLCE64sSus3oHjwG`
  - 汇总总表 `results/master_results.csv`：`1uG-N1F44Qo9kfVcRAhAK73tZslrFN7LC`
  - 最新日志 `logs/latest_run.log`：`1QIxuRqBwFvn_1B3rdAKzIWk1yCki4lVF`
  - `logs/` 目录：`1KiHjiKR1mAWwEGX9-Q7dvfxMDMoAUjob`
  - `runs/` 目录：`1jpufrptcQ571u1YDCmLtggczJmQzckGj`
  - 源码体检报告 `audit/phase0_yolo26_source_audit.md`：`1W4g-3LKDHtCzkTdcLf8Qz-XQ0RrFyQ43`

## 3. 全量实测结果（15 组正式 200-epoch 实验，HIT-UAV 官方划分 2029/290/579，4 有效类，seed=42）

| 实验 | 实际优化器 | Best/总Ep | Params / GFLOPs | Val(rect) m50/m95 | **Test O2O m50 / m95** | **Test O2M m50** | **$\Delta_{\text{E2E}}$ m50/m95** | **Rect 640×512 m50** | 3主类均分 | `Person`/`Car`/`Bic`/`OthV` | 640 耗时 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`p0_yolo26n_musgd`** | **`AdamW(auto)`** | 84/134 | 2.54M / 6.04 | 93.2 / 63.2 | **87.67 / 57.51** | **89.98** | +2.32 / +1.80 | 87.47 | 93.11 | 92.81/**97.01**/89.50/71.34 | 2.40 ms |
| `p0_yolo26n_true_musgd` | `MuSGD(0.01)` | 90/140 | 2.54M / 6.04 | 89.7 / 59.9 | 86.07 / 54.88 | 87.98 | +1.91 / +2.09 | 86.41 | 92.61 | 92.71/95.90/89.22/66.46 | 2.39 ms |
| `p0_yolo26n_sgd` | `SGD(0.01)` | — | 2.54M / 6.04 | 92.2 / 60.9 | 85.04 / 55.06 | 86.93 | +1.89 / +1.42 | 85.90 | 92.64 | 92.96/95.09/89.87/62.22 | 2.51 ms |
| `p0_yolo11n` (NMS) | `SGD` | — | 2.61M / 6.57 | 91.3 / 60.9 | 86.67 / 57.08 | — | — | 86.73 | 92.35 | 91.62/94.98/90.44/69.65 | 3.28 ms |
| `p0_yolov8n` (NMS) | `SGD` | — | 3.14M / 8.75 | 89.4 / 59.5 | 85.35 / 55.24 | — | — | 85.44 | 92.85 | 91.38/96.73/90.44/62.87 | 3.03 ms |
| `p1_mod_a` (v1 裁P5) | AdamW(auto) | — | 1.81M / 7.17 | 87.7 / 58.2 | 84.16 / 53.79 | 87.12 | +2.96 / +1.66 | 83.96 | 91.80 | 90.59/95.80/89.00/61.25 | 3.26 ms |
| `p1_mod_b` (v1 TCAD) | AdamW(auto) | — | 2.54M / — | 90.4 / 60.1 | 87.07 / 55.79 | 87.36 | +0.29 / +0.17 | 87.43 | 92.58 | 92.55/95.36/89.81/70.55 | 3.44 ms |
| `p1_mod_c1` (v1 耦合) | AdamW(auto) | Ep19崩盘 | 2.54M / 6.04 | — | Ep28 NaN | — | — | — | — | — | — |
| `p1v2_c1` (DSG-Assign) | AdamW(auto) | 150/200 | 2.54M / 6.04 | 92.7 / 62.5 | 86.17 / 56.04 | 86.77 | +0.60 / +0.25 | 86.64 | 92.59 | 92.72/94.47/**90.58**/66.94 | **2.38 ms** |
| `p1v2_c2` (DSG-Loss) | AdamW(auto) | 94/144 | 2.54M / 6.04 | 93.3 / 63.0 | 86.47 / 55.33 | 87.71 | +1.24 / +1.36 | 87.26 | 92.32 | **93.59**/94.90/88.46/68.92 | **2.37 ms** |
| `p1v2_c12` (C1+C2) | AdamW(auto) | 132/182 | 2.54M / 6.04 | 92.8 / 63.0 | 85.69 / 54.97 | 86.53 | +0.84 / **+0.14** | 87.19 | 92.24 | 92.27/94.67/89.78/66.03 | 2.40 ms |
| **`p1v2_bitc`** | AdamW(auto) | 161/200 | 2.54M / 6.07 | **93.2 / 64.1** | 87.45 / 56.66 | 87.19 | **-0.27 / -0.40** | 87.42 | 92.32 | 92.64/94.96/89.35/**72.86** | 2.65 ms |
| **`p1v2_tcgate`** | AdamW(auto) | 55/105 | 2.54M / 6.04 | 90.5 / 61.9 | 87.13 / 55.54 | **88.07** | +0.94 / +1.58 | **88.12** | 92.31 | 92.34/95.89/88.71/71.58 | 2.62 ms |
| `p1v2_bitc_tcgate` | AdamW(auto) | 84/134 | 2.54M / 6.07 | 91.2 / 61.8 | 86.64 / 55.68 | 87.60 | +0.95 / +0.57 | 87.23 | 92.89 | 92.78/96.83/89.06/67.91 | 2.71 ms |
| `p1v2_full_c1` | AdamW(auto) | 95/145 | 2.54M / 6.07 | **93.6 / 62.8** | 85.02 / 54.73 | 85.76 | +0.74 / +0.98 | 85.62 | 92.25 | 92.29/94.80/89.67/63.34 | 2.75 ms |
| `p1v2_full_c12` | AdamW(auto) | 90/140 | 2.54M / 6.07 | **93.4 / 64.1** | 84.96 / 54.77 | 86.15 | +1.19 / +0.89 | 86.28 | **93.21** | 92.87/96.60/90.17/60.20 | 2.67 ms |

## 4. V2 深度“尸检”结论（指导 V3 破局）
1. **优化器实证定案**：在小规模 HIT-UAV 上，`AdamW (auto, lr=0.00125)`（87.67% / 57.51%）显著优于官方宣传的 `MuSGD (lr=0.01)`（86.07% / 54.88%，低 -1.60% / -2.63%）和标准 `SGD`（85.04% / 55.06%）。
2. **戳破“$\Delta_{\text{E2E}}$ 假性缩小”**：V2 虽然把 $\Delta_{\text{E2E}}$ 压至 `-0.27% ~ +1.24%`，但 `Test O2M` 天花板同步从 `89.98%` 掉到 `85.76% ~ 88.07%`。根因之一是 `Detect.forward` 对 O2O 输入做了 `x.detach()`，骨干全靠 O2M 教，而我们将 `C1/C2` 同时 patch 进了 `one2many` 伤了老师；且 `p1v2_bitc`(Ep161)/`p1v2_c1`(Ep150) 在 `o2m` 权重衰减后期过拟合 Val（Val m95 达 64.1% 而 Test 跌）。
3. **`DSG-Assign (C1)` 死锚点权重归零隐患**：解耦保持 `overlaps = CIoU` 虽根治了 Ep 19–28 NaN 崩盘，但当 NWD 为小目标选出 `CIoU≈0` 的唯一正样本时，其 `target_scores≈0` 会使 `BboxLoss` 的 `weight` 同步归零，且高 IoU 排序被 NWD 稀释导致 `mAP50-95` 降 1.5–2.5%。
4. **`BiTC-Stem` 与 `TC-Gate` 的单点亮点与串联拮抗**：单用 `BiTC-Stem` 的 Val m95=64.1%、P=89.5%、R=83.6%、OtherVeh=72.86%；单用 `TC-Gate` 在原生无灰边 `640×512` 测出全场最高 **`88.12% (+0.65%)`**；但两者同为局部高通差分，叠加后放大红外热噪声，且灰边（`rect=False`）会在边缘引入伪热对比度。
5. **已打包 Deep Research 委托材料**：`DEEP_RESEARCH_BRIEF_V3.md`（供调研 V3 破局方案：保住 90% O2M 天花板并把 O2O 拉高、解决 C1 死锚点与 `mAP50-95`、红外专属增强与正交轻量化模块）。
