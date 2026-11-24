# HIT-UAV × YOLO26 高空红外无人机弱小目标无 NMS 检测实验框架

本项目面向**传统订阅制、非预警 SCI 期刊**（如 *Infrared Physics & Technology*、*JRTIP*、*SIVP*、*The Visual Computer*），围绕 **YOLO26 原生端到端架构（O2O 无 NMS 分配 + 无 DFL `reg_max=1` 直接回归）** 在高空红外弱小目标（HIT-UAV）上的结构性失效，构建可复现、全开关控制的 Colab Pro 实验管线。

---

## 一、目录结构

```text
ir-yolo26-paper/
├── configs/
│   └── experiments.py          # 总控开关板：全部基线、消融预设、SMOKE_TEST/SEED/RESUME 开关
├── src/
│   ├── inspect_yolo26.py       # [Phase 0 第一步] 实装 YOLO26 源码体检（tal.py/loss.py/head.py/yaml）
│   ├── prepare_hituav.py       # HIT-UAV 自动下载、官方划分转换（4类主评+5类附录）、GT可视化与Drive打包
│   ├── eval_utils.py           # 论文级评测：Val/Test + O2O/O2M 端到端折损差 Δ_E2E + 640x512 + ONNX CPU 延迟
│   ├── train_runner.py         # 统一训练入口：定期同步 Drive、自动断点续训、自动追加 master_results.csv
│   └── modules/hooks.py        # Phase 1 改动 A/B/C1/C2 接入钩子（源码体检后落盘）
└── notebooks/
    └── colab_launcher.ipynb    # Colab Pro 一键启动台（GitHub 拉码 + Drive 存档 + 跑完自动断连停表）
```

---

## 二、首次接入 GitHub + Colab Pro 操作（3 分钟搞定）

### Step 1：在 GitHub 建一个私有仓库（Private）
1. 登录 GitHub → 新建仓库 `ir-yolo26-paper`，勾选 **Private**（不要勾选初始化 README）。
2. 在 Mac 终端把本项目推上去：
   ```bash
   cd /Users/xiaoyuhe/Documents/Gemini-home/ir-yolo26-paper
   git remote add origin https://github.com/<你的用户名>/ir-yolo26-paper.git
   git push -u origin main
   ```

### Step 2：生成一个只读该仓库的 Fine-grained Token
1. GitHub 右上角头像 → **Settings** → 最下方 **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**。
2. Repository access 选 **Only select repositories** → 选 `ir-yolo26-paper`；Permissions → **Contents: Read-only** → 生成并复制 Token。

### Step 3：在 Colab 里导入启动 Notebook 并配好 Secret
1. 打开 [Google Colab](https://colab.research.google.com/) → 上传 `notebooks/colab_launcher.ipynb`（或保存一份到你的 Google Drive）。
2. 点击 Colab 左侧 **钥匙图标（Secrets）** → 新增密钥：
   - Name: `GITHUB_TOKEN`
   - Value: 粘贴刚刚的 GitHub Token
   - 打开左侧的「笔记本访问权限」开关。
3. 右上角「更改运行时类型」选 **L4 GPU**。
4. 把 Cell 1 里的 `GITHUB_REPO` 改成 `"<你的用户名>/ir-yolo26-paper"`，按顺序运行即可！

---

## 三、三阶段实战节奏（不乱烧 CU）

1. **Phase 0.1 源码体检 + 数据集初始化（耗时 ~2 分钟，< 0.05 CU）**
   - Cell 1 设置：`RUN_PHASE0_INSPECT = True`，`RUN_PREPARE_DATA = True`，`RUN_TRAINING = False`。
   - 产出：
     - `Drive/ir-yolo26-paper/audit/phase0_yolo26_source_audit.md`（发给哈基米，据此精准编写改动 A/B/C1/C2 源码补丁）
     - `Drive/ir-yolo26-paper/audit/sanity_vis/*.png`（肉眼确认 GT 框精准套在红外行人/车辆上）
     - `Drive/ir-yolo26-paper/datasets/hit_uav_bundle.zip`（后续每次开新 Colab 仅需 3 秒恢复）
2. **Phase 0.2 烟雾测试 + 基线摸底（耗时 ~1.5 小时，约 3 CU）**
   - 先开 `SMOKE_TEST = True` 跑 2 epoch 验证训练+评估+Drive 归档全链路；
   - 确认无误后关掉 `SMOKE_TEST = False`，开 `AUTO_UNASSIGN = True`，一键队列跑完：
     - `p0_yolo26n_musgd`（默认 MuSGD）
     - `p0_yolo26n_sgd`（对比 SGD 锁死最优优化器）
     - `p0_yolo11n`、`p0_yolov8n`（同协议强基线）
   - 所有结果自动汇总至 `Drive/ir-yolo26-paper/results/master_results.csv`。
3. **Phase 1 改动 A / B / C1 / C2 迭代消融 + Phase 2 三随机种子与跨数据集泛化**
   - 根据 Phase 0 真实源码与基线结果，落盘：
     - **A**：连裁 P5 PAN 颈部 + 轻量 DWConv P2–P4（部分加载 `yolo26n.pt` + 原生 `640×512` 导出）
     - **B**：零初始化门控残差双向热对比度下采样（TCAD）
     - **C1 / C2**：面积尺度门控 NWD 拆开关接入 O2M/O2O 分配器与无 DFL 回归损失
   - 同步跑配对 3 种子（0 / 42 / 3407）与第二红外数据集（`DroneVehicle-IR` 或 `IRSTD-1k`）泛化。

---

## 四、本地快速查看全部预设

```bash
cd /Users/xiaoyuhe/Documents/Gemini-home/ir-yolo26-paper
python3 -m src.train_runner --list
```
