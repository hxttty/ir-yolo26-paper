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
- **本地与 GitHub**：`/Users/xiaoyuhe/Documents/Gemini-home/ir-yolo26-paper` ↔ `https://github.com/hxttty/ir-yolo26-paper.git`（macOS 钥匙串已保存凭据，可直接 `git push`；当前主分支基准 commit `fb81d68`）。
- **Colab Pro 环境**：`NVIDIA L4 (22.5 GB VRAM / 53 GB RAM)`，`Python 3.13.16`，`PyTorch 2.14.1+cu130`，`ultralytics 8.4.218`，余额 ~190.21 CU（常态 1.94–2.33 CU/h）。
- **Google Drive 插件（`gdrive`，账号 `hxy.new@gmail.com`）直达 ID**：
  - 项目根目录 `ir-yolo26-paper/`：`1JCEzpmRCHUnWOEdfaLCE64sSus3oHjwG`
  - 汇总总表 `results/master_results.csv`：`1uG-N1F44Qo9kfVcRAhAK73tZslrFN7LC`
  - 最新日志 `logs/latest_run.log`：`1QIxuRqBwFvn_1B3rdAKzIWk1yCki4lVF`
  - `logs/` 目录：`1KiHjiKR1mAWwEGX9-Q7dvfxMDMoAUjob`
  - `runs/` 目录：`1jpufrptcQ571u1YDCmLtggczJmQzckGj`
  - 源码体检报告 `audit/phase0_yolo26_source_audit.md`：`1W4g-3LKDHtCzkTdcLf8Qz-XQ0RrFyQ43`

## 3. 已验证的实测结果（HIT-UAV 官方划分 2029/290/579，4 有效类，seed=42）

| 实验 | 实际优化器 | Params / GFLOPs | Val mAP50 | **Test mAP50** | **Test mAP50-95** | `Person` / `Car` / `Bicycle` / `OtherVeh` | **$\Delta_{\text{E2E}}$ mAP50** | GPU 延迟 (`640` / `640×512`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `p0_yolov8n` (NMS) | SGD | 3.14M / 8.75 | 89.41% | 85.35% | 55.24% | 91.38 / 96.73 / 90.44 / 62.87 | — | 3.03 / 2.75 ms |
| `p0_yolo11n` (NMS) | SGD | 2.61M / 6.57 | 91.28% | 86.67% | 57.08% | 91.62 / 94.99 / 90.44 / 69.65 | — | 3.28 / 2.82 ms |
| `p0_yolo26n_sgd` (O2O) | SGD | 2.54M / 6.04 | 92.23% | 85.04% | 55.06% | 92.96 / 95.09 / 89.87 / 62.22 | +1.89% | 2.51 / 2.06 ms |
| **`p0_yolo26n_musgd`** (O2O) | **`AdamW(auto)`** | **2.54M / 6.04** | 90.76% | **87.67%** | **57.51%** | 92.81 / **97.01** / 89.50 / **71.34** | **+2.32%** | **2.40 / 2.12 ms** |
| `p1_mod_a` (v1 裁P5换P2) | AdamW(auto) | 1.81M / 7.17 | 87.66% | 84.16% (-3.51%) | 53.79% | 90.59 / 95.81 / 89.00 / 61.25 | +2.96% | 3.26 / 2.76 ms |
| `p1_mod_b` (v1 Stage0+1 TCAD) | AdamW(auto) | 2.54M / — | 90.36% | 87.07% (`640×512`: 87.43%) | 55.79% | `640×512`: **93.03** / 95.98 / **90.63** / 70.10 | **+0.29% (降8×)** | 3.44 / 2.87 ms |
| `p1_mod_c1` (v1 耦合 NWD) | AdamW(auto) | 2.54M | Ep12 84.9% | Ep19–21 崩盘至 0 → Ep28 NaN | — | — | — | — |

> ⚠️ **论文诚实口径备忘**：`optimizer="auto"` 在 HIT-UAV（按 `nbs=64` 折算 6,340 iter < 10,000）下 Ultralytics 实际选用的是 **`AdamW(lr=0.00125)`** 而非 `MuSGD`！论文中务必按真实日志写 `AdamW (Ultralytics auto)`，并由 `p0_yolo26n_true_musgd` 补测显式 `MuSGD`。

## 4. V2 升级架构（commit `fb81d68`，保留 `708/708` 预训练权重 & 8,400 极速锚点）
1. **`DSG-NWD`（`src/modules/nwd_e2e.py`）**：
   - 全程 **FP32** + 输出严格保 **`float32`**（避免 FP16 下 `eps=1e-9` 截断为 `0.0` 触发 `0/0=NaN`）；
   - **C1 (`DSG-Assign`)**：在 `TaskAlignedAssigner.get_box_metrics` 中**解耦**空间匹配排序（`align_metric` 用尺度门控 NWD，`A0=1024, tau=256, w_max=0.25, C=8.0`）与分类软标签上限（`overlaps` 严格保持纯 `CIoU`，`IoU=0` 绝不给正置信度，根治 Ep 19–28 O2O 崩盘）；
   - **C2 (`DSG-Loss`)**：FP32 尺度门控 NWD 接入 `BboxLoss`（精确乘回 `stride[fg_idx[1]]` 还原像素坐标）；零推理开销。
2. **`BiTC-Stem`（`src/modules/tcad.py`，Mod B_v2）**：
   - 仅包裹 Stage 1（放过 640×640 Stage 0），用带通双向对比度 `|AvgPool3×3 − AvgPool5×5|` + 零初始化 `tanh(g)`，仅 +576 参数 / `<0.15 ms`。
3. **`TC-Gate`（`src/modules/tc_gate.py`，Mod A_v2）**：
   - 仅包裹颈部 `P3 (#16)` / `P4 (#19)`，零初始化局部特征对比度空间门控压制红外热杂波，仅 +230 参数 / `<0.12 ms`，100% 区别于 `CVDI-YOLO26`。
4. **V2 烟雾验收（2 epoch，`p1v2_full_c12_smoke` vs `p0_smoke`）**：
   - Params `2.5362M (+900)` | GFLOPs `6.068 (+0.5%)` | 378 FPS (`640×640`) / 435 FPS (`640×512`)
   - Test mAP50 **`59.12% → 60.99% (+1.87%)`**，mAP50-95 **`33.27% → 34.05% (+0.78%)`**，四类全面上涨且零 NaN！

## 5. 当前正在运行 & 后续路线
- **正在运行**：V2 正式 9 组消融队列（`p1v2_c1`, `p1v2_c2`, `p1v2_c12`, `p1v2_bitc`, `p1v2_tcgate`, `p1v2_bitc_tcgate`, `p1v2_full_c1`, `p1v2_full_c12`, `p0_yolo26n_true_musgd`），约 3.5–4h / ~8–9 CU，跑完自动 `runtime.unassign()`。
- **下一步（V2 队列跑完后）**：
  1. 从 Drive `1uG-N1F44Qo9kfVcRAhAK73tZslrFN7LC` 拉取 `master_results.csv` 锁定冠军架构；
  2. 补齐 SOTA 横向表（`sota_yolov10n`, `sota_yolo12n`, `sota_yolo26n_p2_official`, `sota_rtdetr`）与 5 类附录；
  3. 冠军模型 vs 原版 `YOLO26n` 跑配对 **3 随机种子（0, 42, 3407）** `mean ± std` + ONNX CPU/GPU 端到端测速；
  4. 第二红外数据集泛化（优先 Hugging Face 直拉 `DroneVehicle-IR` 裁剪 100px 白边转水平框，或 `IRSTD-1k`）；
  5. 生成论文级可解释性图表（Ep 19–28 崩盘 vs DSG-NWD 稳定曲线、`TC-Gate` 空间热力图、`|tanh(g)|` 门控分布、诚实 Failure Case）→ 论文写作与投稿。
