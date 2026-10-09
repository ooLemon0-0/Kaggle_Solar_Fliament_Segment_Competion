# Git 源码同步

仓库：https://github.com/ooLemon0-0/Kaggle_Solar_Fliament_Segment_Competion

- `main`：本机 Windows 工程源码；工程目录 `F:/CISDI/Cisdi/Kaggle`。
- `linux-autodl`：AutoDL 4090D Linux 迁移源码、固定依赖、启动/暂停/校验/评估脚本。
- `linux-autodl-4090d-v1`：首次 Linux Git 发布快照。模型逻辑与原迁移版一致；Git 版新增从原始数据重建固定 split 的工具。

Git 只保存源码、配置、协议元数据和官方评测参考。数据、完整标注、预训练/训练权重、运行环境、日志、预测结果和凭据均排除。不使用 Git LFS 保存训练资产。第三方 DINOv3 快照保留原 LICENSE.md 与 SOURCE.json；本仓库不替第三方代码重新授权。

## 服务器正在训练时

不需要停训练即可在另一个目录克隆源码：

```bash
cd /root/autodl-tmp
git clone --branch linux-autodl https://github.com/ooLemon0-0/Kaggle_Solar_Fliament_Segment_Competion.git solar_filament_source
git -C /root/autodl-tmp/solar_filament_source rev-parse HEAD
```

当前运行目录仍为 `/root/autodl-tmp/solar_filament_linux`。不要在其中 pull、checkout、reset 或替换源码：队列会为后续路线启动新的 Python，途中更新会混用版本。新克隆目录没有数据/权重/旧断点，也不会自动启动训练。

## 固定划分与数据

`configs/frozen_split_manifest.json` 只保存观测 ID、数量与 SHA256。Linux 数据下载脚本会从单独下载的原始 Kaggle JSON 重建四个 split，并验证与既有正式断点要求的文件字节完全一致：565 train / 142 val observations，936 / 218 标注组，seed42。同一观测的全部标注者保留在同一侧；不重新随机划分。

Windows 已有数据可单独运行：

```powershell
Set-Location 'F:\CISDI\Cisdi\Kaggle'
& '.\work\routes_env\Scripts\python.exe' scripts/restore_frozen_splits.py
```

Linux 按 `README_LINUX_CN.md` 准备环境与下载数据；`linux/download_data.py` 会自动调用重建工具。已有正确 split 不改写；原始数据或既有 split 不匹配时停止，不覆盖旧划分。官方 evaluator 不改动。

## 后续开发与发布

本机保持 `main`；Linux 开发使用独立 checkout，避免覆盖本机工程：

```powershell
Set-Location 'F:\CISDI\Cisdi\Kaggle'
git worktree add '..\Kaggle_linux_source' linux-autodl
```

在对应 checkout 编辑并核验后：

```bash
git status --short
git add <本次修改的源码路径>
git diff --cached --stat
git diff --cached
git commit -m "Describe the source change"
git push origin HEAD
```

不要使用 `git add -f` 添加 data/output/work 或权重。`.gitignore` 不会移除已经提交的敏感文件；若凭据误提交，立即撤销凭据并清理历史。

当前队列完成后再部署新版本；若需提前切换，先在原运行目录执行 `bash linux/stop.sh`，等待 `linux/status.py` 的真实进程为空且 last.pt 落盘。保存原数据、资产、日志及断点。修改目标流、损失、输入尺寸或重建参数的实验使用新的输出命名空间，不能冒充原协议续训。每轮实验记录源码 commit。

本次 Git 初始化与发布不会操作服务器，也不会恢复本机训练或定时任务。
