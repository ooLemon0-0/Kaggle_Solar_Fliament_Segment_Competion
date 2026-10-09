# Solar Filament Segmentation：Linux / AutoDL 源码

这是 `linux-autodl` 分支的源码发布，适用于单卡 RTX 4090D 24GB。同步规则见 [CODE_SYNC.md](CODE_SYNC.md)，完整准备、训练与评估步骤见 [README_LINUX_CN.md](README_LINUX_CN.md)。

追加的 Flow v2 实验及“旧队列完成后自动启动”命令见 [FLOW_FOLLOWUP.md](FLOW_FOLLOWUP.md)。旧六路线保持原协议；新结果保存到独立 `flow_followup_v1`，不覆盖旧 Flow。

投影路线输入梯度报错的修复、在线更新已固定版本的CPU等待器，以及“剩余特征实验 → 投影补跑 → Flow v2”自动队列，见 [PROJECTION_REPAIR.md](PROJECTION_REPAIR.md)。

Git 版不带原始数据、标注、权重、断点和日志。`linux/download_data.py` 下载数据后自动根据 `configs/frozen_split_manifest.json` 重建原固定 split，校验 SHA256。要继续旧实验，需另外提供原资产和断点；源码克隆本身不会续训。

已有服务器队列运行时，请克隆到独立目录。不要在运行目录更新源码，也不要在新克隆目录再启动一份训练。
