# Solar Filament Segmentation 实验工程

`main` 保存 Windows 工程；AutoDL 请使用 [`linux-autodl` 分支](https://github.com/ooLemon0-0/Kaggle_Solar_Fliament_Segment_Competion/tree/linux-autodl)。同步与运行中目录的更新规则见 [CODE_SYNC.md](CODE_SYNC.md)。Git 不包含数据、完整标注、环境、权重或运行结果；固定划分以 ID 清单和 SHA256 保存，下载数据后通过 `scripts/restore_frozen_splits.py` 重建。

下面为第一轮三模型 baseline 的历史操作说明；当前六路线 Linux 流程以 Linux 分支的 README 为准。

# Solar Filament Segmentation V0

三种单类 filament 分割 baseline 的可复现工程。三个模型已完成同一完整划分上的30-epoch训练、官方评测和全部test预测；早期debug smoke独立保留。

实际项目：`F:\CISDI\Cisdi\Kaggle`。

## 环境

本机 GPU 为 RTX 5070 Ti 16 GB。运行环境是项目内的 `work/evaluation_probe`，继承 `dbd_dinov3` 的 Python 3.11.16、PyTorch 2.11.0+cu128；所需额外依赖安装在项目虚拟环境内。旧 `yolo11_infer` 的 CUDA 11.8 不支持此 GPU。

在 PowerShell 中：

```powershell
Set-Location 'F:\CISDI\Cisdi\Kaggle'
& '.\work\evaluation_probe\Scripts\Activate.ps1'
python -c "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
```

若 PowerShell 的策略禁止激活，可将下文每个 `python` 改为 `& '.\work\evaluation_probe\Scripts\python.exe'`，无需修改系统执行策略。

在另一台电脑，先用 Python 3.11 创建独立环境，再执行 `python -m pip install -r requirements.txt`。requirements 固定当前实际使用的版本，并指定 CUDA 12.8 wheel index。安装后先验证 GPU 运算。完整环境版本另保存在 `output/environment_packages.json`。

## 官方数据与预训练权重

原始数据在 `data/raw/MAGFiLO_1.0_Kaggle_2026/`，包含 707 张 train JPEG、180 张 test JPEG、1154 条 annotator-image 记录和 8199 个实例。原始 JPEG 与 JSON 不修改；动态 resize 在数据加载或 Ultralytics 内部执行。

本机已经完成 Kaggle 认证和比赛入场。重新下载时使用独立的 CLI：

```powershell
$env:KAGGLE_CONFIG_DIR=Join-Path (Get-Location) 'work\kaggle_config'
& '.\work\kaggle_cli\Scripts\kaggle.exe' competitions download -c filament-segmentation-2026 -p data/raw
& '.\work\kaggle_cli\Scripts\kaggle.exe' kernels pull azimahmadzadeh/self-evaluation-notebook -p official/self_evaluation -m
```

已提取、验证并保留官方 Notebook。重新下载后不要直接覆盖 evaluator；应先核对 Notebook SHA256 与评分变更。来源为 [Official Self-Evaluation Notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)。

YOLO 与 ResNet50 权重已存入 `output/pretrained/`。需要时可执行：

```powershell
python scripts/prepare_weights.py --model all
```

DINOv3 使用本机已有目录：

```text
F:\CISDI\Cisdi\DBD\dbd_compositional_dinov3\pretrained\dinov3-vits16-pretrain-lvd1689m
```

在其他电脑上先提供同版本 DINOv3 ViT-S/16 的 HF config 和 safetensors，并修改 `configs/dino_transformer.yaml` 的 `backbone_weights`。代码仅从指定本地路径加载，不自动切换为随机 backbone。

## 数据检查与 GT 可视化

```powershell
python scripts/analyze_dataset.py
python scripts/visualize_gt.py --count 30
python -m unittest discover -s tests -v
```

审计、instance_statistics.csv、GT masks/bbox/spine 图统一位于 `output/data_analysis/`。完整拆分位于 `output/splits/train.json` 和 `val.json`，固定 seed=42：565 个 train observations/936 个 annotator groups，142 个 val observations/218 个 groups。同一个 `file_name` stem 的所有记录必须在同一侧。重新分析会核对已有 split 一致性，拒绝替换不同 split。

debug_train/debug_val 使用正式 split 中固定的 8/4 个 observations，保留全部标注者，分别为 14/8 个训练/验证 samples。三个模型共享同一批 records。

## 三个 baseline

| 模型 | 实现 | 监督与实例转换 |
|---|---|---|
| YOLO-Seg | YOLO11s-seg，Ultralytics 8.4.15，COCO 预训练 | 每个 annotation 一个单类 polygon；native instance mask |
| Semantic | SMP 0.5.0 U-Net + ResNet50，官方 ImageNet V1 encoder | BCEWithLogits + Dice；8 连通域；默认不做 morphology |
| DINO | Transformers 5.17.0 DINOv3 ViT-S/16 frozen backbone；128 维、3 层 decoder、64 queries | objectness + dense 256x256 query masks；训练 Hungarian 匹配与 mask BCE/Dice |

输入统一 1024，灰度复制为三个相同通道；U-Net/DINO 使用 ImageNet mean/std，YOLO 使用其成熟预处理。DINO backbone 始终 eval/frozen。chirality categories 仅作为原始元数据保留，不进入分类监督。

预测接口为 `{'image_id': observation, 'instances': [{'mask': HxW bool, 'score': float}, ...]}`，统一恢复到 2048 原尺寸。在磁盘上以 COCO compressed RLE 保存等价 mask，避免密集数组存储。所有模型最小 native mask area=32，最多输出64实例（按score排序）。GT 每个 group 实例数为1–26。U-Net 过滤前连通域数量也保留，避免隐藏碎片问题。这个上限与阈值是明确记录的 V0 配置。

## debug 全链条

```powershell
python scripts/train.py --model yolo_seg --debug
python scripts/train.py --model semantic_seg --debug
python scripts/train.py --model dino_transformer --debug
python scripts/evaluate.py --model all --debug
python scripts/make_submission.py --model yolo_seg --debug
python scripts/make_submission.py --model semantic_seg --debug
python scripts/make_submission.py --model dino_transformer --debug
python scripts/compare_models.py --debug
python scripts/check_artifacts.py --debug
```

默认 debug 为 1 epoch。可用 `--epochs 3` 做更长 smoke；公平比较时三个模型应使用相同 epoch。训练每个 epoch 都执行官方 val PQ，以该指标保存 best.pt，并保存 last.pt、config.json、training_log.csv、training_summary.json。YOLO 的 `training/weights/best.pt` 是库自身的 mAP 选择，**统一比较只读取 `checkpoints/best.pt`，它由官方 PQ callback 选择**。

debug 结果在 `output/<model>/debug/`，包括 checkpoints、predictions、visualizations、metrics.json 和 submission.csv。正式训练位于 `output/<model>/`，与 debug 隔离。

## 正式训练与统一比较

单卡后台队列入口为 `python scripts/run_full_baselines.py`，依次训练三个模型各30 epochs、生成test CSV、比较并检查产物。状态保存在 `output/benchmark/full/pipeline_status.json`，训练控制台日志位于 `output/<model>/train_console.log`。入口拒绝覆盖已有正式checkpoint，也拒绝重复启动标记为running的队列。

```powershell
python scripts/train.py --model yolo_seg
python scripts/train.py --model semantic_seg
python scripts/train.py --model dino_transformer
python scripts/evaluate.py --model all
python scripts/make_submission.py --model yolo_seg
python scripts/make_submission.py --model semantic_seg
python scripts/make_submission.py --model dino_transformer
python scripts/compare_models.py
```

正式配置为全量 train split、30 epochs、batch=1、AdamW。改变预算时三个模型统一设置并记录。推理重用保存的训练配置，保证 checkpoint 结构、尺寸与 epoch 信息正确。后续 1536/2048 实验前先归档当前 output 中对应的 run，再统一改三个 configs 的 image_size/inference_size；当前没有 tile。

单独运行推理或使用指定 checkpoint：

```powershell
python scripts/infer.py --model yolo_seg --debug --split val
python scripts/infer.py --model semantic_seg --debug --split test
python scripts/make_submission.py --model dino_transformer --debug --checkpoint output/dino_transformer/debug/checkpoints/last.pt
```

YOLO 转换可单独调用 `python scripts/convert_to_yolo_seg.py --debug`。副本与 labels 位于 output/data_analysis/yolo/，不 resize 或改写原始图片。

## 指标与提交

`src/evaluation/official_pq.py` 原样提取官方 cell 2、4、5（包含 Markdown、从1计数），函数 AST 与原文一致。官方 PQ：严格 IoU>0.5，计入所有 qualifying pairs，累计全部 annotator-image 的 TP IoU/FP/FN 后计算总分；不用自行替换为 Hungarian 或 per-image PQ 平均。相同 prediction 分别对照不同 annotator GT，不能把其 masks 合并为一份实例 GT。

adapter 为内存受控，分块调用同一官方 overlap 函数并在支持时使用 GPU float32。回归测试验证与原始 CPU Notebook 的矩阵和 PQ 相同。所有 val observation 必须有 prediction record，即使 instances 为空；缺失、重复或错配图像会报错。

Dice/IoU 是每个 annotator group 的 foreground union 像素分数平均，另保存官方正重叠 pair 的均值。fragmentation 是 IoU>0 下一个GT重叠多预测的数量；over_merging 是一个预测重叠多GT的数量。两者计数遵循官方 relationship 图语义。

推理统一 FP16/AMP，每模型一次 GPU warmup、同步计时。时间包括读图、预处理、forward 和实例后处理，排除加载模型、画图、RLE编码与官方评测。显存同时保存训练peak和推理peak（PyTorch allocated）；参数量包含frozen backbone。debug 只测4张图，速度不能当作稳定全量性能。

`output/benchmark/results.csv`、comparison.md 是最近一次比较；debug/full 子目录保留对应结果。comparison 图包含 Original、GT、三个模型及放大视图；放大仅用于观察，不是模型tile。每张图标注展示的annotator，完整指标使用所有annotators。

submission.csv 只有 filament_id、segmentation_rle。使用 pycocotools compressed counts，不添加 bbox、spine、category/chirality。所有180张test均执行推理，没有预测的图像不创建伪实例。submission_summary.json 记录覆盖数、行数和零检测图。**当前debug DINO无检测，CSV仅表头；需要继续训练后才能形成有意义的比赛预测。** 工程没有自动上传Kaggle。

## 当前验证范围

数据读取、30张GT可视化、按observation拆分、官方PQ回归、三个模型的loss/backward/checkpoint、val推理、统一评测、全部test推理与CSV生成已实际执行。完整30-epoch实验已完成：YOLO官方PQ=0.371927、U-Net=0.343608、DINO=0.026902，分别选择第7、7、26轮的最佳权重。全部142张val的同图比较和三个模型产物检查已通过。正式结果见 `output/benchmark/full/` 和 `output/benchmark/comparison.md`，早期debug结果独立保留。

7 项回归测试已通过。`scripts/check_artifacts.py --debug` 另检查三个模型的 checkpoint、配置、日志、相同 split、全部180张test覆盖、CSV字段及抽样RLE解码；DINO直接验证64个query的dense mask输出、mask head反向传播和backbone冻结。检查记录在 `output/benchmark/debug/artifact_checks.json`。
