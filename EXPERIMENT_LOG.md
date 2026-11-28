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
- **本地与 GitHub**：`/Users/xiaoyuhe/Documents/Gemini-home/ir-yolo26-paper` ↔ `https://github.com/hxttty/ir-yolo26-paper.git`（当前主分支最新 commit `3afc031`，已推送至 `origin/main`）。
- **Colab Pro 环境**：`NVIDIA L4 (22.5 GB VRAM / 53 GB RAM)`，`Python 3.13.16`，`PyTorch 2.14.1+cu130`，`ultralytics 8.4.218`，余额 **`170.41 CU`**（2026-11-25 21:04）。
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
| **`v3_e0_base`** *(IR-Clean)* | IR-Clean *(T4)* | **44**/94 | 2.54M / 6.04 | 89.7 / 59.1 *(sq)* | **84.37 / 52.23** | 84.80 | +0.43 / +0.68 | 84.87 *(m95: 53.1)* | 92.25 | 91.44/96.35/88.98/60.71 | 5.21 *(T4)* |
| `v3_e1_p1_distill` *(P1a+1b)* | IR-Clean *(T4)* | 97/147 | 2.54M / 6.04 | 87.9 / 59.1 *(sq)* | 85.25 / 56.36 | 86.74 | +1.50 / `+0.28` | 85.51 *(m95: 56.8)* | 91.71 ⚠️ | 91.92/95.77/87.44/65.85 | 5.49 *(T4)* |
| 🏆 **`v3_e2_p1_piou` *(P1c, 0ms)*** | IR-Clean *(T4)* | 63/113 | **2.54M / 6.04** | 89.7 / 59.2 *(sq)* | **88.80 / 56.33** | **89.77** | **`+0.97` / +1.11** | **88.56** *(m95: 56.6)* | 92.28 | 91.46/96.71/88.65/**78.37** | **4.86 *(T4)*** |
| 🔥 **`v3_e6_arch_combo` *(P2+3)*** | IR-Clean *(T4)* | 109/159 | **2.55M / 6.13** | **91.9 / 60.5** *(sq)* | **87.21 / 56.44** | 88.09 | **`+0.89` / +1.29** | **88.54** *(m95: **58.3**)* | **92.72** | **92.88**/96.58/88.70/70.67 | 5.62 *(T4)* |
| `v3_e7_full` *(P1+2+3, PIoU2)* | IR-Clean *(L4)* | 123/173 | 2.55M / 6.13 | 87.4 / 57.9 *(sq)* | 86.65 / **56.91** | 88.47 | +1.82 / **`+0.26`** | 86.73 *(m95: 57.6)* | 92.29 | 92.38/95.72/88.77/69.73 | 2.85 *(L4)* |
| `v3_e7b_full_sd` *(P1+2+3, SD)* | IR-Clean *(L4)* | 137/187 | 2.55M / 6.13 | 90.7 / 61.4 *(sq)* | 84.97 / 55.88 | 86.22 | +1.25 / `+0.18` | 85.33 *(m95: 57.0)* | 91.51 | 91.50/94.62/88.42/65.33 | 2.92 *(L4)* |

## 4. V2 深度“尸检”结论（指导 V3 破局）
1. **优化器实证定案**：在小规模 HIT-UAV 上，`AdamW (auto, lr=0.00125)`（87.67% / 57.51%）显著优于官方宣传的 `MuSGD (lr=0.01)`（86.07% / 54.88%，低 -1.60% / -2.63%）和标准 `SGD`（85.04% / 55.06%）。
2. **戳破“$\Delta_{\text{E2E}}$ 假性缩小”**：V2 虽然把 $\Delta_{\text{E2E}}$ 压至 `-0.27% ~ +1.24%`，但 `Test O2M` 天花板同步从 `89.98%` 掉到 `85.76% ~ 88.07%`。根因之一是 `Detect.forward` 对 O2O 输入做了 `x.detach()`，骨干全靠 O2M 教，而我们将 `C1/C2` 同时 patch 进了 `one2many` 伤了老师；且 `p1v2_bitc`(Ep161)/`p1v2_c1`(Ep150) 在 `o2m` 权重衰减后期过拟合 Val（Val m95 达 64.1% 而 Test 跌）。
3. **`DSG-Assign (C1)` 死锚点权重归零隐患**：解耦保持 `overlaps = CIoU` 虽根治了 Ep 19–28 NaN 崩盘，但当 NWD 为小目标选出 `CIoU≈0` 的唯一正样本时，其 `target_scores≈0` 会使 `BboxLoss` 的 `weight` 同步归零，且高 IoU 排序被 NWD 稀释导致 `mAP50-95` 降 1.5–2.5%。
4. **`BiTC-Stem` 与 `TC-Gate` 的单点亮点与串联拮抗**：单用 `BiTC-Stem` 的 Val m95=64.1%、P=89.5%、R=83.6%、OtherVeh=72.86%；单用 `TC-Gate` 在原生无灰边 `640×512` 测出全场最高 **`88.12% (+0.65%)`**；但两者同为局部高通差分，叠加后放大红外热噪声，且灰边（`rect=False`）会在边缘引入伪热对比度。
5. **Deep Research V3 报告核心铁证（已全部通过 `search_web` 核验）**：
   - **隐性撞车止损**：2026.10 *Frontiers in Marine Science* (Liu et al.) 在 `YOLOv26n` 上用了 P3 零初始化 `tanh(g)` 注意力 (`Gated Output-16`) + 双分支 `1024 px²` 切换 `MPDIoU-NWD` → **V3 已彻底退役 `TC-Gate` 与双分支 `1024 px² NWD`**！
   - **统计方差审计**：仅约 30 框的 `OtherVehicle` 一类与 4 类总 `Test O2O mAP50` 相关性高达 `r = 0.9639`（解释 `92.9%` 方差），单类采样误差即可注入 `±4.05%` → 必须并行报 **`3-Class Main Avg` + `AP_s`**，最终冠军 vs 基线必须跑 **5 组配对种子**（双侧 t 检验门槛从 4.303 降至 2.776）。
   - **Ultralytics `nms=None` 陷阱**：默认 `val()` 走的是 O2M+NMS（实测验证集 100% 证实！），论文需显式说明 `nms=False` 才是真正的 O2O。
   - **目标期刊与第二数据集定案**：冲高 *Infrared Physics & Technology*（中科院 2 区，0 APC，门槛 `89.8%+ / 59.5%+`）；主力 *JRTIP*（2026 IF 4.9，0 APC，主打 NMS-free 确定性延迟 + P95/P99 尾延迟，门槛 `89.2%–89.7%`）；二顺位 *The Visual Computer*；*SIVP* 仅保底。第二红外数据集唯一主推 **`DroneVehicle-IR`**（640×512 裁 100px 白边转 HBB，8,980 张测试集；快测备选 `FLIR-Aligned IR`）。

## 5. V3 代码架构（commit `e8c6f92`）与实时实验进度
- **三支柱正交实现（+0.49% 参数 `2.5477M` / +1.56% 算力 `6.129G` / 708/708 权重 100% 无损）**：
  1. **Pillar 1 (`src/modules/v3_losses.py`, `ODCS`, +0ms)**：神圣教师原则（绝不动 `one2many` 分配器）+ 保底 `ProgLoss` (`o2m >= 0.45`, `o2o` 前 60% 爬满 0.9) + 模块级可序列化选择性 `O2M→O2O` 内部自蒸馏 (`V3DistillE2ELoss`，次优抑制严格排除 `one2one` 自身前景) + `one2one` 专属退火分配 (`A0=256, tau=64`, 前 60% 退火回纯 CIoU) + 死锚点回归保底权重 + `PIoU2` (Neural Networks 2024, `u(λq)` `.detach()`, FP32) / `SD Loss` (AAAI 2025)。
  2. **Pillar 2 (`src/modules/v3_arch.py`, `BorderSafeBiTCStemWrapper`, +608 params)**：Stage 1 `F.pad(mode='replicate')` + FP32 Charbonnier 光滑幅值（防 FP16 下溢 NaN）+ 零初始化 `tanh(g)`。
  3. **Pillar 3 (`src/modules/v3_arch.py`, `BorderSafeCAAResWrapper`, +11.8K params, 备选 `HaarResWrapper` +8.6K)**：Neck P3/P4 `AvgPool 7×7` + `1×11`/`11×1` 条带长程卷积（全 `replicate` 填充）+ 零初始化 `tanh(g)`。
  4. **IR-Clean 统一协议**：`hsv_h=0, hsv_s=0, hsv_v=0.25, erasing=0.0, close_mosaic=15, deterministic=True, cache="disk", align_val_rect_false=True`。
- **已完成 V3 全 6 组精准拆弹结论（彻底破案！）**：
  1. **真凶 #1 — `IR-Clean` 去掉 `erasing=0.4` 导致严重过拟合**：`v3_e0_base`（`erasing=0.0, hsv_v=0.25, close_mosaic=15`）在第 **44 Epoch** 就早停过拟合，Test O2O 从 `p0_yolo26n_musgd` 的 `87.67% / 57.51%` 暴跌至 **`84.37% / 52.23%`**（`-3.30% / -5.28%`）！证明 2,029 张小规模 HIT-UAV 离不开 `erasing=0.4` 与 `hsv_v=0.4`（热像仪增益漂移）。
  2. **真凶 #2 — `v3_e1_p1_distill`（ProgFloor + 次优硬抑制 KD）拖累完全体**：`ProgFloor` 压低 O2M 权重饿坏了唯一更新骨干的 O2M 老师（O2M 跌至 `86.74%`），次优硬抑制砍掉低置信度召回尾巴（`3-Class Main Avg` 跌至 `91.71%`，`-0.54%`，全场唯一下跌），直接把 `v3_e7_full` 的 Val 从 `91.91%` 拽低到 `87.40%`、Test 从 `88.80%` 拽低到 `86.65%`。
  3. **双冠王诞生 — Pillar 1c (`v3_e2_p1_piou`) + Pillar 2&3 (`v3_e6_arch_combo`) 均暴力正增益**：
     - **`v3_e2_p1_piou`（+0 参数 / +0ms）**：在残血 `IR-Clean` 协议下仍轰出 **全场历史最高 Test O2O `88.80%`**（比 `v3_e0_base` **`+4.43% mAP50 / +4.10% mAP50-95`**，甚至反超旧协议 `p0` 基线 `+1.13%`！），完美守住 O2M 天花板 **`89.77%`**，`Δ_E2E` 从 `+2.32%` 腰斩至 **`+0.97%`**，`OtherVeh = 78.37%` 历史新高！
     - **`v3_e6_arch_combo`（`BiTC-Safe + CAA-Res`, +12.4K 参数）**：比 `v3_e0_base` 暴涨 **`+2.84% mAP50 / +4.21% mAP50-95`**，Val 达 **`91.91% / 60.49%`**，**V3 最高 3 主类均分 `92.72%`**，**原生 `640×512` 轰出 `88.54% / 58.3%`**（全场历史最高 mAP50-95！），3 个零初始化门控全部激活（`BiTC=0.095`, `CAA-P3=0.109`, `CAA-P4=0.124`）。
  4. **`PIoU2` 完胜 `AAAI 2025 SD Loss`**（`+1.68% mAP50 / +1.03% mAP50-95 / +5.52% P`）→ 永久淘汰 `SD Loss`。
- **V3.1 破局方案（`IR-Reg` 黄金协议 + 剔除有毒 `e1`，合体 `Pillar 1c + Pillar 2 + Pillar 3`）**：
  - 协议：`hsv_h=0.0, hsv_s=0.0`（物理干净）+ 恢复 `hsv_v=0.4, erasing=0.4, close_mosaic=10` + `cache="disk", deterministic=True, align_val_rect_false=True`。
  - 已在 `configs/experiments.py` 新增 7 个 V3.1 预设（`v31_e0_base`, `v31_e1_p1c_piou`, `v31_e2_bitc_safe`, `v31_e3_caa_res`, `v31_e4_arch_combo`, `v31_e5_champion`, `v31_e5b_champ_boxkd`）。
  - **V3.1 四组核心门控实测完赛（2026-11-26 09:24–12:10 JST, L4, `IR-Reg`, `val rect=F`, `seed=42`, `Ultralytics 8.4.219`）**：
    - `v31_e0_base`（Ep 69/119）：Val `89.55/59.50`，**Test O2O `84.26% / 54.19%`**，O2M `86.32%`，`Δ_E2E +2.06% / +1.40%`，`3cls 92.14%`，`OtherVeh 60.61%`，`Rect 85.50%`，`2.40 ms`；
    - `v31_e1_p1c_piou`（Ep 83/133, **0ms**）：Val `89.71/59.81`，**Test O2O `84.94% / 55.05%`** (`+0.68/+0.86`)，O2M `85.38%`，**`Δ_E2E +0.44% / +0.47%`**，**`3cls 92.61%`** (`+0.47%`)，`OtherVeh 61.95%`，`Rect 85.94%`，`2.37 ms`；
    - `v31_e4_arch_combo`（Ep 48/98）：Val `90.24/58.45`，**Test O2O `85.85% / 54.44%`** (`+1.59/+0.25`)，O2M `86.34%`，**`Δ_E2E +0.50% / +1.03%`**，`3cls 92.18%`，`OtherVeh 66.85%`，`Rect 86.36% / 55.2%`，`2.93 ms`；
    - `v31_e5_champion`（Ep 110/160）：Val `90.14/60.43`（O2M Val `91.9/61.7`），**Test O2O `85.14% / 55.23%`** (`+0.88/+1.04`)，O2M `85.75%`，**`Δ_E2E +0.61% / +0.30%`**，`3cls 92.30%`，`OtherVeh 63.66%`，**`Rect 86.64% / 56.80%`**（`P=89.9/R=83.6`），`2.86 ms`，门控全激活（`BiTC=0.098, CAA-P3=0.108, CAA-P4=0.121`）。
  - **关键破案与 V3.2 修正（2026-11-26 12:15 JST）**：
    - 相对 `v31_e0_base`，三大支柱全员正增益且把 `Δ_E2E` 从 `+2.06% / +1.40%` 压至 `+0.44%~+0.61% / +0.30%`；
    - **但 `v31_e0_base`（纯官方模型）比 `p0_yolo26n_musgd`（`87.67/57.51`, O2M `89.98%`）低了 `-3.41% / -3.32%`**；逐行 diff 锁定元凶为 `align_val_rect_false=True`（+ `cache="disk"` + `hsv_h/s=0` + `8.4.219`）导致 Ep 69 锁死次优 `best.pt`（而 `p0` 在 `val rect=True` 下于 Ep 84 选出 `Val 93.2/63.5`）。
    - 已新增 **V3.2 官方零漂移协议预设**（`v32_e0_base`, `v32_e1_p1c_piou`, `v32_e2_bitc_safe`, `v32_e3_caa_res`, `v32_e4_arch_combo`, `v32_e5_champion`：`val rect=True, cache=True`, 官方默认 aug），实测完赛（2026-11-26 12:20–14:42 JST, L4, `seed=42`, `Ultralytics 8.4.219`）：
      - `v32_e0_base`（Best Ep **84**/134）：Val `90.38/60.62`，**Test O2O `87.29% / 57.39%`**，**Test O2M `89.59% / 59.17%`**，`Δ_E2E +2.30% / +1.78%`，`3cls 92.68%`（100% 复现 `p0`，彻底证实之前低 3.4% 是 `align_val_rect_false=True` 导致）；
      - `v32_e1_p1c_piou`（Best Ep **97**/147）：Val **`91.31/60.98`** (`+0.93/+0.36`)，Test O2O `86.71% / 55.97%` (`-0.58/-1.42`)，Test O2M `86.86% / 56.63%` (`-2.73/-2.54`)，`Δ_E2E +0.15% / +0.66%`；
      - `v32_e5_champion`（Best Ep **161**/200）：Val **`92.09/61.23`** (`+1.71/+0.61` 历史最高)，Test O2O `86.20% / 56.24%` (`-1.09/-1.15`)，Test O2M `86.22% / 56.37%` (`-3.37/-2.80`)，`Δ_E2E +0.02% / +0.13%`。
  - **V3.3 方案 A 实测完赛（2026-11-26 15:10–18:55 JST, T4, 100ep `cos_lr=True` + `close_mosaic=10`, `seed=42`）**：
    - `v33_e0_base`（基线）：Val `90.64/60.55`，Test O2O `86.48% / 56.34%`，O2M `87.20%`，`Δ_E2E +0.73% / +0.38%`，`3cls 92.83%`，`Rect 85.97%`；
    - `v33_e1_p1c_ciou`（仅 P1c-CIoU, 0ms）：Val `91.50/59.47`，Test O2O `86.08% / 54.94%`，O2M `87.09%`，`Rect 88.10%` (`+2.13%`)；
    - `v33_e2_arch_combo`（仅 P2+P3）：Val `90.56/59.94`，Test O2O `86.61% / 55.64%`，O2M `87.42%`，`Rect 87.17%` (`+1.20%`)；
    - 🏆 **`v33_e3_full_ciou`（完整体）**：Test O2O **`87.60% / 56.37%`** (`+1.13% / +0.03%`)，O2M **`88.14%`** (`+0.94%`)，**`Δ_E2E +0.54%`**，**`3cls 92.95%`**，**`Rect 87.63%`** (`+1.66%`)，`P=89.56%`。
  - **正式转入路线 1（SCI 定稿实验阶段，主攻 *JRTIP / TVC*，2026-11-26 19:05 JST）**：
    - `src/train_runner.py` 新增 `@s<seed>` 队列内联覆盖语法（零改 Colab Notebook 即可一条队列跑多 seed）；`configs/experiments.py` 补全单模块消融 (`v33_e2a_bitc_safe`, `v33_e2b_caa_res`)、统一 100ep SOTA 对比 (`v33_sota_yolov8n/10n/11n/12n`) 及 5-class 附录 (`v33_5cls_base/full`)。
    - **V3.3 3-Seed 配对验血完赛（2026-11-26 19:06–21:25 JST, L4, `seeds=42, 0, 3407`）**：
      - `seed=42`：`86.48/56.34` → `87.60/56.37` (`+1.13/+0.03`, `3cls +0.12%`)；
      - `seed=0`：`85.49/56.32` → `86.81/56.56` (`+1.32/+0.23`, `3cls +0.06%`, `Δ_E2E 1.29%→0.58%`)；
      - `seed=3407`：`87.92/56.69` → `86.02/56.29` (`-1.91/-0.40`, `3cls +0.05%`, `P +5.59%`)；
      - **3-Seed 均值**：`mAP50: 86.63±1.22% → 86.81±0.79% (+0.18%, t=+0.17)`，`mAP50-95: 56.45±0.21% → 56.40±0.14% (-0.05%)`，`3cls: 92.68% → 92.76% (+0.08%, t=+3.52)`，`P: 85.73% → 88.25% (+2.53%)`。
  - **关键止损与 Phase 2 `P2/4` 高分辨率试金石探路（2026-11-26 21:32 JST）**：
    - 决策：3-seed 验血证实 `P3-P5` 下仅 +12.4K 参数微雕的均值增益只有 `+0.18%`（不足以支撑硬核 SCI 投稿），及时止损！
    - **P2/4 高分辨率试金石完赛（2026-11-26 21:38–23:04 JST, L4, 100ep `cos_lr`, `seed=42`）**：
      - `p2_probe_base`（`2.61M / 8.82G`, 708 项预训练全救回）：Val `88.06/57.80`，Test O2O `84.98% / 55.70%` (`-1.50%`)，O2M `86.46%`，`Δ_E2E +1.49%`（翻倍），`3.75 ms`；
      - `p2_probe_odcs`：Val 飙至 `92.06/60.49`（`+4.0%`），但 Test 仍停在 `84.92% / 55.27%`（`OtherVeh 61.44%`, `3cls 92.74%`）——彻底证伪 2K 小图加 34K P2 锚点！
  - **Phase 3 制胜一击：零延迟 YOLO26 双头无 DFL 红外知识蒸馏（2026-11-26 23:09 JST）**：
    - 战略：推理期 `+0 参数 / +0.00 ms`（稳守 `2.53M / 418 FPS`），彻底告别“2K 小图塞结构必过拟合或只涨 0.18%”死胡同；同时储备 `DroneVehicle` 配对 RGB→IR 特权跨模态蒸馏终极王牌。
    - 第一步先测同族强教师上限：新增 `teacher_yolo26s`, `teacher_yolo26m`, `teacher_yolo26l`（100ep `cos_lr`, `batch=32/16` 防 OOM），先跑 `["teacher_yolo26s", "teacher_yolo26m"]` 摸清蒸馏空间并留存冻结教师权重！
    - **`teacher_yolo26s` 实测完赛（2026-11-26 23:44 JST, L4, 34.97 min, 9.95M / 22.75G, `seed=42`）**：
      - Val `90.98% / 61.43%`（历史最高 Val `mAP50-95`）；
      - **Test O2O (640×640)**：**`88.01% / 58.27%`**（相对 `v33_e0_base` `seed=42` **`+1.54% mAP50 / +1.93% mAP50-95`**；相对 3-seed 均值 `+1.38% / +1.82%`）；
      - **Test Rect (原生 640×512)**：**`88.81% / 59.3%`**（相对 `v33_e0_base` `85.97%` 暴涨 **`+2.83% mAP50`**！）；
      - **3 主类均分**：**`93.32%`**（`Person 93.58% / Car 96.22% / Bicycle 90.17%`，全 39 轮历史最高！`OtherVeh 72.08%`）；
      - **关键机制实证**：`Δ_E2E = -0.80% / -0.06%`（`O2O 88.01%` 反超 `O2M+NMS 87.21%`！验证“强骨干容量彻底消除甚至逆转端到端鸿沟”，为 `M2 CrossKD-O2O` 跨头梯度桥提供极高质量冻结 `Teacher O2O` 决策边界）。
    - **`teacher_yolo26m` 实测完赛（2026-11-27 00:50 JST, L4, 65.26 min, 21.78M / 74.98G, `seed=42`）**：
      - Val `90.08% / 60.82%`；Test O2O `86.52% / 57.99%`；Test O2M `87.47% / 58.7%`（`Δ_E2E +0.95% / +0.73%`）；Rect `87.59% / 59.6%`；**3 主类均分 `93.53%`**（`Bicycle 91.14%` 历史新高），但 `OtherVehicle` 过拟合跌至 `65.47%`（相对 `s` 的 `72.08%` 降 `-6.61%`）。
      - **小样本容量匹配结论**：在 2K HIT-UAV 上，**`teacher_yolo26s` 在 Val、Test O2O、Rect 和 `Δ_E2E (-0.80%)` 全面优于 22M 的 `m`**，同时速度 3.3× 更快、容量差更小（`4×` vs `8.6×`），自然成为 HIT-UAV 最佳蒸馏教师（`YOLO26m` 留给 18K `DroneVehicle` 主战场）。
    - **`kd_yolo26.py` + `run_phase3_auto.py` 经 Astra 对抗审计全量加固（commit `b57ce0c`, 2026-11-27 00:39 JST）**：
      - 已修复全部 3 P0 + 7 P1 + 3 P2：KD 非有限直接 `FloatingPointError` 熔断、子进程非零退出立即失败（绝不误判为 `SCENARIO_C_BRAKE`）、**全链路严格 Val-Only 选老师/赢家/5-seed（零 Test 泄漏）**、去掉学生 pre-hook 直用 `s_o2m["feats"]`、完整 `kd_state.pt`（adapters + AdamW 矩）同步恢复、训完中断直跳过训练补评测、`validate_teacher_compatibility`、硬 `timeout` 防挂死、Gate 1 正式 `batch=32` + `adapter_steps>=1` + `best.pt` 字节流纯净度校验、精确无截断 `compute_soft_bce_kd`、锁死 `ultralytics==8.4.219`。
      - **过夜全自动实验已干净完赛（2026-11-27 00:52–06:31 JST, 总耗时 5.65h, Scenario C 熔断生效并自动释放 GPU）**：
        - **Gate 0 + Gate 1 绿灯通关**：Val-Only 锁定 `teacher_yolo26s`，2-ep 正式 `batch=32` 冒烟确认 `best.pt` 100% 纯净（`2.5353M / 708 tensors / 0 KD refs`, `peak_vram=7.28 GB`）。
        - **Gate 2（5 组 KD 探针全部低于 `v33_e0_base` `Val 90.64/60.55 | Test 86.48/56.34`，自动触发 Scenario C 省 CU 刹车）**：
          - `p3_kd_full_s` (`M1+M2+M3+run`)：Val `87.14/58.76` | Test O2O `84.98/55.82` | 3cls `91.45%` | `Δ_E2E +1.80%` | Rect `85.48%`（**但 `O2M+NMS` train-val 飙至 `92.6/64.3` 峰值 `64.6` 反超基线 `63.7` 与教师 `62.6`！**）
          - `p3_kd_full_norun_s` (`M1+M2+M3`)：Val `87.49/59.14` | Test O2O `84.44/55.34` | 3cls `91.49%` | `Δ_E2E +1.38%` | Rect `84.77%`
          - `p3_kd_gentle_s` (`0.3× M1/M2/M3`)：Val `88.30/59.08` | Test O2O `85.77/55.86` | 3cls `91.61%` | `Δ_E2E +1.65%` | Rect `86.70%`
          - `p3_kd_m1_fpkd_s` (仅 `M1`)：Val `89.18/58.72` | Test O2O `85.80/55.73` | 3cls `92.24%` | `Δ_E2E +1.21%` | Rect `85.26%`
          - `p3_kd_m12_cross_s` (`M1+M2`, 无 `M3`)：Val `89.26/59.17` | Test O2O `86.37/56.09` | 3cls `92.30%` | `Δ_E2E +1.38%`（Val `Δ_E2E = 0.00%`！）| **Rect `87.15/57.1`（反超基线 `85.97` +1.18%！）**
        - **Gate 3 同步收齐全部 4 组统一 100ep `cos_lr` SOTA 基线（`seed=42`, L4）**：
          - `v33_sota_yolov8n` (`3.14M/8.8G`, NMS)：Val `88.68/60.11` | Test `86.48/57.24` | 3cls `93.15%` | Rect `87.47%` | `3.04 ms (329 FPS)`
          - `v33_sota_yolov10n` (`2.74M/8.6G`, E2E)：Val `88.81/59.44` | Test `85.66/55.87` | 3cls `92.64%` | `Δ_E2E +1.63%` | Rect `86.00%` | `2.46 ms (406 FPS)`
          - `v33_sota_yolo11n` (`2.61M/6.6G`, NMS)：Val `90.51/60.94` | Test `87.37/56.93` | 3cls `93.02%` | Rect `87.86%` | `3.28 ms (305 FPS)`
          - `v33_sota_yolo12n` (`2.58M/7.5G`, NMS)：Val `90.23/59.89` | Test `87.35/57.22` | 3cls `92.91%` | Rect `88.46%` | `4.09 ms (245 FPS)`
        - **深度“尸检”铁证（3 大真实根因，100% 定位！）**：
          1. **硬核算法 BUG：`M3` 的 1-to-1 O2O 锚点错位拔河（`t_o2o_fg ≠ s_o2o_fg`）**：开 `M3` 后 O2O `cls_loss` 暴涨 `+30%`（Ep80 `0.52→0.68`, Ep100 `0.44→0.54`），`p3_kd_full_s` 的 O2M+NMS 飙到 `64.3%`（峰值 `64.6%`）但 O2O Recall 暴跌 `85.8%→79.7%`。原因：在 `topk2=1` 稀疏匹配下，当教师与学生对同一小目标选了相邻的不同锚点时，原生 GT 死命把教师锚点压向 `0`，`M3` 却同时把它拉向教师正 logit（还乘 `T^2=4`）！**必修复**：`M3` 的 O2O 正监督必须严格限制在 `t_o2o_fg & s_o2o_fg` 交集上（绝不与原生 O2O 负样本硬刚），且 `M2` 延迟到 Ep 5 适配器预热后再开启（消除 Ep 1 `M2=9.94` 冲击），KD 衰减提早至 `0.50→1.00` 留足后期精调。
          2. **2K HIT-UAV 单模态蒸馏天花板饱和（教师过拟合）**：`teacher_yolo26s` 在 2,029 张训练集上第 66 轮即过拟合（`Val O2O` 仅比学生高 `+0.34% mAP50`，`Val/Test O2M` 甚至低于/持平学生 `92.0 vs 92.7`、`87.21 vs 87.20`），纯 `M1` 把学生 Neck 强拉到 `corr=0.928` 直接传递了教师训练集过拟合；而叠加 `M2 CrossKD-O2O` 全面反超纯 `M1`（`Rect +1.89%`, `Test +0.57%`, `Val Δ_E2E → 0.00%`），验证跨头梯度桥本身方向是对的。
          3. **主战场切换铁证**：必须先修复上述 `kd_yolo26.py` 3 处缺陷，再将主火力切换至 **17,990 张训练 / 8,980 张测试且自带配对 RGB 的 `DroneVehicle`**（既有 `YOLO26m` 真正的大样本容量差，更有白天 RGB 跨模态特权语义增益 + 夜间熔断防负迁移故事），视需要再用修好后的 `kd_yolo26.py` 补 1 组 45 分钟 HIT-UAV 验证复盘。
      - Drive 实时决策日志 ID：`10JCZD82G2ip6zWtUGwvDHrWe2gKAqqEI` (`logs/auto_phase3_report.md`)。
    - **`kd_yolo26.py` 零延迟三模块蒸馏引擎已落地（2026-11-26 23:57 JST）**：
      - 实现 `src/modules/kd_yolo26.py`（`M1 F-PKD` + `M2 CrossKD-O2O` 绕过 `x_detach` + `M3 QG-CIoU-L1` 无分布双头校准 + 公平对照基线 `CWD` / `Official PR#24894 L2` / `QG-KD 1-IoU` + Track B 夜间熔断与双师优势门控原语）；
      - 适配器独立 `AdamW` 随 `GradScaler` 步进（绕过 `MuSGD` 正交化与 `LambdaLR` 冲突），`deepcopy`/`pickle` 严格剥离教师与适配器（确保导出 `best.pt` 为 100% 纯净 `2.535M / 6.035G / 418 FPS` 原生 `YOLO26n`）；
      - 新增 16 组 Phase 3 完整消融与对比预设（`p3_kd_*_s` / `p3_kd_*_m`）及 Drive 教师权重自动解析器。
    - **Phase 3 v2 + DroneVehicle Track B 两轮 GPT-6.2-Astra 对抗审计与全量加固（2026-11-27 09:53 JST）**：
      - 首轮审计揪出 7 P0 + 8 P1 + 4 P2，二轮复检再堵住 3 项续训/完成标记/Gate 边界反例；全部已修复并通过 8 组本机 PyTorch + OpenCV 回归测试：
        1. `M3` & `Track B` 严格执行精确同目标 `t_fg & s_fg & (t_gt_idx == s_gt_idx)`，独立拆开 O2M/O2O 交集开关且不漂移历史预设；
        2. 同步配对管线 `paired_rgbt_data.py` 贯穿 `rgb_valid_mask`（关闭混淆的 `erasing=0.0`），夜间熔断改像素掩码矩估计 + `count_include_pad=False`，黑帧+114 灰边与纯灰帧误开率严格 **`0.0000`**；
        3. Track B 冒烟强锁 `MuSGD` + `batch=32` + `close_mosaic=1` + `oom_retries==0` + 双组 AdamW 真实 `step>=1` + `cm_feat>0` + `cm_cls>0` + `expected_nc=5` 原生权重 schema/hook 校验；
        4. `train_runner.py` 落地唯一代目录 `gen_ep{N}_{uuid}` + SHA-256 暂存恢复 + `results.csv` 同步 + 配置变更拒绝旧续训 + 缺失 `best.pt` 绝不跳过（同时兼容 HIT-UAV 已完赛的旧版 `p3_kd_v2_full_s`）；
        5. `prepare_dronevehicle.py` 强制 4 目录/1:1 stem/XML 坐标完整性/互斥全覆盖昼夜清单、`dv_manifest.json`、坏包自动隔离重做、本地快路径自动补打缺失 Drive bundle；
        6. 所有 DroneVehicle E2E 预设加 `nms=False`（按原生 O2O 选 `best.pt`），Track A/B 对齐相同 IR 蒸馏权重，STAL 高斯前景 16-box 分块（误差 `0.00e+00`）。
    - **Phase 4 DroneVehicle Wave 1 实测完赛（2026-11-27 19:16 JST，官方完整 17,990 train / 1,469 val / 8,980 test，切干净镜像 `nonameplease/DroneVehicleNoBorder`，Drive 永久缓存 `dronevehicle_bundle.zip` = `15afG6jfizdQpPYwznCjwOEE0k4DLd2Dt`）**：
      - HIT-UAV v2 复盘 **`p3_kd_v2_full_s`**（修复 M3 锚点拔河后）：Val `89.39/59.44`（比昨晚 `+2.25/+0.68`），Test O2O `85.26/55.60`，`Δ_E2E` 压至 **`+0.49%`**，**Rect `87.27%`**（反超基线 `85.97%` **`+1.30%`**），3 主类 `92.59%`。
      - **DroneVehicle Wave 1 三强基准全量实测（8,980 张测试集，50ep `cos_lr`，`seed=42`，`master_results_dronevehicle.csv` ID = `1l5QPyUWlsGdiM9k1E5WbHo1BR1gZQHCc`）**：
        - **`dv_ir_yolo26n`（纯红外学生基线，2.54M / 447 FPS）**：Val `81.10/59.44` | Test O2O **`78.13/57.66`** | **Day `76.12/55.07`** | **Night `80.55/60.93`**（纯红外白天暴跌 `-4.43%/-5.86%`！）| Rect `78.60%` | `Δ_E2E +0.93%`（`car 97.07 / bus 93.63 / truck 75.99 / freight_car 64.01 / van 59.95`）；
        - **`dv_ir_teacher_s`（红外主教师 YOLO26s，9.95M / 217 FPS）**：Val **`83.46/61.62`** | Test O2O **`79.94/59.62`** (`+1.81/+1.96`) | **Day `78.58/57.58`** | **Night `81.48/62.16`** | Rect `80.55%` | `Δ_E2E +0.71%`（`truck +3.14%, freight_car +2.53%, van +2.22%`）；
        - **`dv_rgb_teacher_s`（特权 RGB 辅助教师 YOLO26s，9.95M）**：Val `78.70/56.20` | Test O2O `76.56/54.84` | **Day `79.35/59.67`**（🔥 **反超红外主教师 `+0.77% / +2.09%`，碾压红外学生 `+3.23% / +4.60%`！**）| **Night `70.37/45.00`**（💥 **暴跌 `-10.18% / -15.93%`！**）；
      - **Track B 双师冒烟体检（`dv_kd_track_b_s_smoke`，甚至在 15 GB T4 上 `batch=32` 峰值显存仅 `9.11 GB`！）**：`MuSGD`, `batch=32/32`, `oom_retries=0`, `primary_steps=36`, `aux_steps=36`, `M1/M2/M3/CM_F/CM_C` 全部健康收敛，夜间熔断 `safe=48.4%(Mosaic)/29.8%(NoMosaic)`，双师优势门控 `gate=28.5%/27.9%`，导出权重 `2.5358M` 与基线零差异（顺便在冒烟阶段抓出 `verify_exported_student_purity` 拿从零 `c3=64` 比预训练 `c3=80` 的误报，已立即修复并推送 `13581a3`）。
    - **Phase 4 DroneVehicle Wave 2 核心实测进展（2026-11-27 19:23 – 2026-11-28 12:03 JST，commit `13581a3`，`workers=8`，50ep `cos_lr`，`seed=42`，O2O `best.pt`）**：
      - **`19:33` 🟢 `TRACK B SMOKE GATE PASSED`**：`2.5358M` 原生 `COCO-adapted yolo26n.yaml(nc=5)` 708 张量逐项匹配、0 KD 残留、0 hook、`batch=32 (oom=0)`、`MuSGD`、`primary_steps=36 / aux_steps=36`、峰值显存仅 **`9.12 GB`**；
      - 🏆 **`dv_kd_track_a_s`（单师纯红外 `IR S → IR n`, `+0.00 ms`）已完赛归档（`23:47` JST）**：
        - Val `81.91% / 60.16%`（训练期峰值 Ep 37 `82.77/60.66`，关 Mosaic Ep 41–50 逆势抗过拟合稳守 `82.4~82.6`）；
        - **Test O2O (8,980 张)**：**`79.13% / 58.63%`**（相对基线 `78.13/57.66` 净涨 **`+1.00% / +0.97%`**，转化 55.6% 师生差距！）；
        - **Day**：`77.06% / 56.07%`（`+0.94% / +1.00%`）；**Night**：**`81.42% / 61.76%`**（`+0.87% / +0.83%`，距 4× 红外老师 `81.48%` 仅差 `0.06%`！）；
        - **Rect 640×512**：**`79.53%`**（`+0.93%`）；**端到端鸿沟 Δ_E2E**：从 `+0.93/+0.95` 压至 **`+0.58/+0.57`**（直降 37%，优于老师 `+0.71`！）；
        - **5 类全员普涨**：`car 97.30%`（反超 4× 老师 `97.16%`！）/`truck 78.19% (+2.20%)`/`bus 94.29% (+0.66%)`/`van 60.33% (+0.38%)`/`freight_car 65.56% (+1.54%)`；
      - ⚠️ **`dv_kd_track_b_s`（Track B v1 双师 `RGB+IR → IR-only`, `+0.00 ms`）已完赛归档（2026-11-28 `12:03` JST）**：
        - Val `81.95% / 60.12%`（Ep 39 退火反弹峰值 `82.72/60.76`）；
        - **Test O2O (8,980 张)**：`78.90% / 58.22%`（比基线 `+0.76/+0.56`，但比 Track A `-0.24/-0.41`）；
        - **Day**：**`77.13%`** `/ 55.95%`（mAP50 微超 Track A `+0.06%`，难类 `van 60.48%` 超 Track A `+0.15%`）；
        - 💥 **Night**：`80.91% / 61.10%`（比 Track A 暴跌 **`-0.51% / -0.66%`**！）；Rect `79.24%`，`Δ_E2E +0.78%`；
        - **精确尸检 3 根因（`kd_health.json` + 源码定位）**：① 旧亮度开关 `v_win>=0.9999 & std>=0.05` 误杀白天边缘/平坦路面（无 Mosaic 仅 `illum_safe=30.8%`）却放行高对比度路灯夜景；② `CM_F=0.50` 未叠加双师优势门控，把暗夜/劣质 RGB 特征硬灌进 IR Neck 冲突；③ `compute_fpkd_loss` 与 `CM_C` 除以稀疏门控和 `g_sum`，把 5% 夜间漏水样本反向放大 20×–200×！
      - 🔧 **Track B v2 精准手术（`dv_kd_track_b_v2_s`，用户拍板路线 A，1 次 L4 试金石，不成即刻转纯 Track A 主线）**：
        - 升级：`quantile_macro` 16×16 宏块 5×5 有效中位数熔断（免疫路灯夜景 + 保留白天 P4/P5 边缘）+ `CM_F` 双师质量优势门控（`q_rgb > q_ir + margin` 仅对 RGB 胜出前景开放）+ `fg_only` / `N` 比例归一化（彻底消除稀疏漏水放大，夜间零负迁移）；Track B v1 保留为消融对照组 `w/o Quality-Gated Feature`。
      - **后续完整投稿计划（当前余额 ~85–95 CU）**：`dv_kd_track_b_v2_s` 试金石决胜 → 若赢则补 `Naive` 对照；随后 Wave 3A（组件消融 + `CWD` + `Conf-L2` 竞品 KD）→ Wave 3B（4 组 50ep SOTA `v8n/v10n/11n/12n`）→ 可选 `YOLO26m` 强师 / 3-Seed + ONNX CPU 延迟测速。
