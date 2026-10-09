import _bootstrap
import importlib.metadata
from pathlib import Path
from src.utils.common import ROOT,digest,split_path
from src.routes_v3.engine import BASE,atomic_json

dest=ROOT/'deliverables/routes_v3'
common=dict(size=1056,epochs=30,hours=4,seed=42,accumulation=4,eval_every=2,lr=1e-4,encoder_lr=1e-5)
runs=[dict(common,id='rf',kind='rf',dependency=None),
      dict(common,id='flow',kind='flow',size=384,lr=1e-5,encoder_lr=1e-5,dependency=None),
      dict(common,id='sam',kind='sam',dependency='rf',lr=5e-5,encoder_lr=1e-6),
      dict(common,id='image_projection',kind='image_projection',dependency='rf'),
      dict(common,id='feature_control',kind='feature_control',dependency='rf'),
      dict(common,id='feature_structure',kind='feature_structure',dependency='rf')]
manifest=dict(protocol_version=3,runs=runs,deliverable_dir=str(dest),train_split_sha256=digest(split_path('train')),
              val_split_sha256=digest(split_path('val')),train_observations=565,train_annotation_groups=936,
              val_observations=142,val_annotation_groups=218,
              versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','rfdetr','cellpose','dinov3','transformers']},
              common_postprocess=dict(confidence=.25,max_instances=64,mask_logit_threshold=0.,native_mask_size=2048),
              test_policy='No test data used for training, selection, thresholds or route ranking.',
              budget='30 epochs OR 4 training hours per run, first reached; preflight, asset preparation and validation excluded.',
              inheritance='All dependent runs inherit best RF checkpoint; report both incremental and inherited training cost.',
              resume='AdamW, model, RNG and epoch permutation/offset saved atomically after completed gradient accumulation.')
atomic_json(BASE/'experiment_manifest.json',manifest);dest.mkdir(parents=True,exist_ok=True);atomic_json(dest/'experiment_manifest.json',manifest)
plan='''# 本机 16GB：五类路线，六组实验

比较目标：选择最有潜力的太阳暗条实例分割方案。每个目标需要独立 mask，细长结构的完整性和目标间分离同样重要。旧 baseline 已完成，保留作参考。

| 路线 | 本轮技术组合 | 要验证的判断 |
|---|---|---|
| 端到端 Transformer | RF-DETR-Seg Small 实例预训练；多层 DINOv2 特征；可训练编码器；官方集合匹配和稀疏 mask loss | 旧冻结 DINO 的失败是否来自预训练与 decoder/微调不足 |
| 稠密流场实例分割 | CellposeDINO ViT-B；384 原尺寸块；前景偏置采样；前景+二维流；重叠块先融合流，再全图重建实例 | 不经检测框，能否更好保存细长目标并分离邻近目标 |
| 定位→分割 | 最优 RF 模型预测框→SAM2.1 Hiera Tiny；微调最后两视觉 blocks 和 neck（lr1e-6）；prompt/mask decoder（lr5e-5）；原图局部细节残差精修到512 | 局部放大与形状先验能否提升定位后的 mask IoU |
| 图像空间投影 | 有界残差投影头→冻结最优 RF；梯度穿过完整实例分割器 | 学习适合任务的图像变换，是否比直接微调更有用 |
| 特征空间结构化投影 | 最优 RF 冻结 backbone/定位 decoder；可训练成熟 mask head；原图 /4 细节融合 RF 上下文；前景辅助损失+实例内聚/实例间分离 | 学习“干净空间”能否同时保留轮廓并区分实例 |
| 必要对照 | 同一特征投影网络，关闭辅助结构损失，其余设置一致 | 防止把新增参数和细节分支的收益误认为投影损失收益 |

执行顺序：RF → CellposeDINO → SAM2 → 图像投影 → 特征对照 → 特征结构投影。依赖项等待 RF 的最优 checkpoint；独立路线不会因另一独立路线失败而永久停止。

固定：seed42；565 train observations / 936 独立标注组；142 val observations / 218 标注组；原尺寸 2048 官方 PQ 不改；另报前景并集 Dice/IoU、TP/FP/FN、碎裂与粘连。整个实验不使用 test 调参。

每组最多 30 epochs 或 4 小时训练计算，先达到者停止，最多 24 GPU 训练小时，加上验证和准备时间。单卡顺序运行。满图 RF 输入优先 1056，预检 OOM 时按固定顺序 960、768 回退；正式开始后所有 RF 依赖组使用同一尺寸。Cellpose 使用 384 原尺寸 tile，SAM 1024 局部输入。这是本机预算约束下的路线筛选，不声称等累计预算：依赖组还继承 RF 的训练成本，报告须单列。

验证每 2 epochs 和结束时，最优 checkpoint 由完整官方 val PQ 选择。物理 batch1，累积4次；记录实际 epoch、部分 epoch 样本、优化步数与计算时间。CPU/RAM 不预装全部 native masks。训练前先做最大实例数样本的反传和短优化检查；debug 检查不计入正式结果。

投影不需要“干净图片”真值。图像投影由实例损失反向监督；特征投影同时学习前景与实例分离。特征空间不是把所有目标压成一个二值通道，而是让同一实例的表示相近、不同实例分开。

Cellpose 的一个像素只能属于一个实例：重叠标注在训练流损失中排除，前景保留；官方验证仍使用完整原始独立 masks。跨标注者记录不合并。SAM 正式成绩使用真实预测框；理想框诊断只能另列，不能作为正式 PQ。RF 框漏失会限制两阶段上限。

失败处理：保留 checkpoint 与日志；checkpoint 包括 optimizer、RNG、epoch 排列和样本位置；不得盲目覆盖重训。OOM 只允许正式启动前预设回退，正式失败要诊断并记录变更。监控交叉核验进程命令/创建时间、日志和 checkpoint，不凭 running 状态判断存活。

最终比较：官方完整 val、同图可视化、按目标大小/细长程度的分层诊断、定位召回/理想框与真实框差距、实际预算。只有完整验证后的结果可用于排名；短训练领先只说明有潜力。

研究依据：
- [RF-DETR 官方](https://github.com/roboflow/rf-detr)
- [CellposeDINO 官方模型](https://cellpose.readthedocs.io/en/latest/models.html)
- [SAM2 官方](https://github.com/facebookresearch/sam2)
- [Cellpose3：由分割任务监督图像恢复](https://www.nature.com/articles/s41592-025-02595-5)
- [CellSAM：定位结合 SAM](https://www.nature.com/articles/s41592-025-02879-w)

这些来源支持技术方向；没有论文能证明上述组合在本比赛必胜，需要本轮实验验证。
'''
(BASE/'plan.md').write_text(plan,encoding='utf-8');(dest/'plan.md').write_text(plan,encoding='utf-8')
print(str(BASE/'experiment_manifest.json'))
