> **Git 发布说明**：本文下面的“已完成 RF / Flow 第24轮 / SAM 第1轮”等数字是最初 ZIP 迁移快照，并非实时进度。当前 Git 克隆只含源码；从 [CODE_SYNC.md](CODE_SYNC.md) 开始。ZIP 源码带完整 split，Git 版改为冻结 ID 清单，并在下载官方数据后重建相同 SHA256 的 split。续训须另外提供原资产与断点。正在跑的服务器目录不更新。

# AutoDL：单卡 RTX 4090D 24GB 迁移版

本包用于继续现有六实验，严格串行：RF → Flow → SAM → 图像投影 → 特征对照 → 特征结构。已经完成的 RF 跳过；Flow 从第24轮待验证断点继续，先重做完整验证；SAM 从第1轮第300/936个样本继续。不会自动重训已有正式实验。监控定时任务和本机训练均保持停止。

## 1. 服务器与上传

选 **1张4090D 24GB**，无需96GB卡。建议主机内存32GB起，64GB更宽裕；数据盘可用100GB起。Flow的目标流生成、数据解码及官方PQ也消耗CPU，优先选择有足够CPU核心的套餐。性能和实时单价以AutoDL页面为准，没有在你的服务器实测速度。

选择带Miniconda的Ubuntu22.04/24.04镜像；宿主驱动必须能运行PyTorch CUDA12.8。系统CUDA目录的版本不能单独证明可用，后面的GPU实测校验才是依据。默认只使用第0张GPU，不做DDP。

上传两个ZIP及其SHA256校验文件到 `/root/autodl-tmp`：源码包包含固定split、官方评测和Linux脚本；续训资产包包含已训练权重/优化器/RNG/日志及三个预训练模型。数据和Windows环境没有打进ZIP。不要把本包解压到另一个正在训练的目录。

```bash
cd /root/autodl-tmp
sha256sum -c SHA256SUMS
unzip solar_filament_linux_source.zip
unzip solar_filament_linux_resume_assets.zip
cd solar_filament_linux
```

这就是你上传的完整源码入口，不需要另找Git仓库。RF-DETR/Cellpose/Transformers安装固定版本；DINOv3使用包内已验证源码快照，避免下载移动的main分支。需要源码仓库的来源和许可见 `vendor/dinov3_snapshot/SOURCE.json`、`LICENSE.md`。包内 `linux/integrity.json` 锁定split、官方评测和实验配置。

## 2. 网络、Conda与独立环境

```bash
command -v conda
conda --version
conda env list
nvidia-smi
df -h /root/autodl-tmp
free -h
# 如果conda不在PATH，通常可先：source /root/miniconda3/etc/profile.d/conda.sh
# 如果已能找到conda：
source "$(conda info --base)/etc/profile.d/conda.sh"
bash linux/check_network.sh | tee logs/network_before_install.log
```

首次网络检查尚无Kaggle CLI/凭据而返回非零是预期：先看HTTPS可达性，再准备下载环境。脚本分别检查Kaggle、PyPI、PyTorch、conda-forge、Hugging Face和GitHub；HTTP401/403证明收到HTTP响应，不能证明比赛有下载权限。签名文件下载域名必须通过实际数据下载再验证。

```bash
bash linux/setup_env.sh tools
# Kaggle Settings -> API生成同一已加入比赛账号的token；下面输入不回显：
read -rsp 'Kaggle API token: ' KAGGLE_API_TOKEN; echo
export KAGGLE_API_TOKEN
bash linux/check_network.sh | tee logs/network_authenticated.log
```

也支持把旧版 `kaggle.json` 放在 `~/.kaggle/kaggle.json` 并 `chmod 600 ~/.kaggle/kaggle.json`。不要把token写进训练日志或发给别人。CLI文件列表失败：401先查凭据；403查同一账号的Join Competition/规则接受与网关；HTTP000查DNS/TLS/代理。AutoDL网络加速按其官方文档操作，不能预先保证你的节点能下载Kaggle。

建立独立训练环境，不覆盖AutoDL的base和其他Conda环境：

```bash
bash linux/setup_env.sh train
PY="$PWD/.envs/solar-routes/bin/python"
"$PY" -m pip check
"$PY" linux/check_environment.py --mode gpu
```

默认训练前缀 `.envs/solar-routes`，下载工具 `.envs/kaggle-tools`，均是Python3.11。可通过 `SOLAR_TRAIN_PREFIX`、`SOLAR_TOOLS_PREFIX` 改位置；后面的下载命令相应传入 `--kaggle "$SOLAR_TOOLS_PREFIX/bin/kaggle"`。PyTorch固定 `2.11.0` / torchvision `0.26.0`，官方cu128 wheel；校验包含CUDA反传、BF16、torchvision CUDA NMS和真实显存。若驱动测试失败，换兼容AutoDL镜像/联系平台，不能靠继续训练解决。安装日志、pip freeze和校验JSON留在work/logs。

## 3. 拉比赛数据、准备资产与预检

```bash
PY="$PWD/.envs/solar-routes/bin/python"
"$PY" linux/download_data.py --official-notebook
"$PY" linux/prepare_assets.py
"$PY" linux/prepare_project.py
"$PY" linux/check_environment.py --mode ready
# 在正式启动前做当前下一组的8步反传预检：
"$PY" linux/check_environment.py --mode ready --preflight flow
```

`download_data.py`通过官方Kaggle CLI下载并校验/解压；原始数据在 `data/raw`，固定split不会重新随机划分。期望707张train图、180张test图，其中train/val是565/142观测、936/218标注组；test仅核对文件是否完整，不用于调参。官方notebook另下载到work供参考，**不覆盖包内官方评测**。

续训资产包已带原始预训练权重，`prepare_assets.py`默认仅校验SHA256。如果只上传源码包，可尝试 `"$PY" linux/prepare_assets.py --download` 下载固定来源；SAM导出若字节指纹不同，会停止并要求使用提供的精确资产，避免悄悄换预训练版本。没有断点则会从头训练，想续训必须上传续训资产包。

预检产物在 `output/benchmark/routes_v3/linux_preflight/<route>`，仅debug，不是正式PQ，不改变正式权重和30轮/4小时预算。其余组可在它们轮到且GPU空闲时做同样预检；正式队列会为未有断点/预检的路线执行原有预检。

## 4. 启动、状态、安全暂停与续训

```bash
bash linux/start.sh
tail -f logs/pipeline_console.log
# 新开终端：
.envs/solar-routes/bin/python linux/status.py
tail -f output/benchmark/routes_v3/flow/train_console.log
```

后台runner逐组自动接续；一组失败记录failure并继续独立可运行项，缺少RF依赖则标blocked。不重复启动同一路线。状态工具核验真实Python命令行与创建时间，并展示日志/checkpoint活动时间；running JSON不是进程存活证据。后台不需要本地Codex定时任务；阶段结果会写进日志和报告，没有自动发Kaggle提交。

```bash
# 暂停，等待checkpoint安全写完再关机：
bash linux/stop.sh
.envs/solar-routes/bin/python linux/status.py
# 确认Actual processes为空、last.pt已更新，再停止租赁实例。
# 继续：
bash linux/start.sh
```

停止请求在梯度累积边界落盘，保存模型、AdamW、RNG、epoch permutation/offset。验证中会先完成该次验证，暂停可能需要等待；不要kill -9。意外掉电只能恢复到最后一次原子checkpoint，未保存尾段可能重放。再次start是明确续训动作，会清除停止请求并跳过completed实验。

## 5. 完整评估和报告

**正式训练停止/完成后**使用，避免GPU争抢；评估与训练共享独占锁：

```bash
PY="$PWD/.envs/solar-routes/bin/python"
"$PY" linux/evaluate_routes.py --route all --checkpoint best
# SAM理想框/扰动框诊断缺失或失败时，补齐：
"$PY" linux/evaluate_routes.py --route sam --checkpoint best --sam-diagnostic
"$PY" linux/report.py          # 当前临时报告
"$PY" linux/report.py --final  # 只有六组全部完成且产物校验通过才成功
```

评估覆盖全部142观测/218标注组，恢复2048原尺寸，调用未改动的官方PQ；额外评估写到 `output/benchmark/routes_v3/re_evaluation`，不覆盖用于选择的best结果。全部报告在 `deliverables/routes_v3`，包括PQ/Dice/IoU、RF框召回、SAM诊断与同图可视化。SAM诊断用GT派生提示，仅诊断，不能冒充正式PQ。此轮路线筛选不新生成test提交。

当前迁移快照：RF已经30轮，最佳PQ **0.411695**（第20轮）；Flow最佳 **0.372842**（第18轮），第24轮训练完成但正式验证未落盘，迁移后重做第24轮验证；SAM仅第1轮第300个样本，无当前正式PQ。详见 `migration_snapshot.json` 和续训日志。

## 6. 实验含义、成本及限制

| ID | 方法 | 当前输入策略 |
|---|---|---|
| rf | RF-DETR-Seg Small，query端到端检测/实例mask | 全图1056 |
| flow | CellposeDINO ViT-B，前景+二维流，经全局拼接重建实例 | 原图尺度384重叠tile |
| sam | RF框定位+SAM2.1Tiny局部精修，微调最后两视觉block/neck/decoder | 原图ROI编码1024，精修512 |
| image_projection | 前置残差投影头+冻结最优RF | 缩放后1056，3通道有界残差 |
| feature_control | RF mask head+细节支路，无结构辅助损失 | 全图1056 |
| feature_structure | 上述支路+前景BCE/Dice及实例pull/push | 全图1056 |

各路线固定seed42，最多30epochs或4小时准备+训练计算时间，先到者停；验证、预检及历史GPU等待不计入4小时，但服务器租赁照样计费。SAM与投影/特征组继承RF成本，报告新增及继承成本，不能声称累计预算完全公平。迁移不重置预算。换GPU/系统后保存状态可恢复，但不能保证bit-for-bit重现。

当前图像投影是在缩放后的1056上处理，**尚未验证原生超高分辨率投影的优势**。这次先续完已有协议；未来2048/原尺寸更强YOLO、分块投影或其他高分辨率实验应另建实验命名空间、匹配输入尺度/预算，不混入本轮结论。

本机已核验断点结构、split/评测指纹、预训练SHA、Python语法、串行队列控制和ZIP完整性。**尚未在你的Linux4090D上安装/训练实测**，必须跑上述网络/GPU/ready/preflight再启动。不存在由本包保证的Kaggle外网访问、速度或预期分数。

源码也保留上一轮YOLO/ResNet U-Net/冻结DINO基线。它们的额外依赖列在 `linux/requirements-baselines-optional.txt`，不参加当前续训队列；如需重跑应另建Conda环境/输出目录。旧DINOv3 ViT-S/16权重可能需要单独的Hugging Face授权，不在这次六实验预训练资产包中。不要把可选依赖装进当前固定路线环境而破坏版本校验。

参考：[PyTorch官方安装命令](https://pytorch.org/get-started/previous-versions/)、[Kaggle官方CLI认证与数据下载](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)、[AutoDL官方文档](https://www.autodl.com/docs/)。
