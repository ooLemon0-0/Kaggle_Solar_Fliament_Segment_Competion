# Flow v2：独立追加实验

此版本不修改旧六路线的配置或模型行为。不要在正在训练的 `solar_filament_linux` 中更新代码。新源码在 `solar_filament_source_clean` 中等待旧队列结束，然后独立串行运行。

## 技术组合

保留 CellposeDINO ViT-B 原预训练。训练输入从原尺寸384 crop扩大到768 crop；全图仍为2048，没有把整图缩小到768。选择目标完整包围框可容纳的随机crop，而非随机前景像素中心。完整标注组先生成固定中心流场，再裁剪；90度旋转/水平翻转同时变换dy/dx方向，避免裁剪位置改变目标中心。对DINO blocks做梯度检查点以降低显存。

损失改为前景平均流MSE＋低权重背景流MSE＋前景加权BCE＋Dice，重叠像素不参与监督。负样本crop仍保留20%的随机采样。固定目标按训练split SHA缓存，缓存生成发生在训练loss内并计入训练时长。目标缓存用float16，首次生成与后续读取使用相同量化；推理原始流字段保持float32。

v2从原预训练重新开始，**不续训旧Flow**。最多30轮或8个准备/计算小时，先到者停止；每2轮及最终完整验证，按固定2000积分步/.4流一致性阈值选择best checkpoint。GPU预检必须通过，否则停止，不暗中降低输入尺寸。768并未解决所有上下文歧义，中心流表示对长细/分支目标的适配仍是实验假设。

随后各缓存一次旧Flow最佳checkpoint与v2最佳checkpoint的完整验证原始流；每组比较200/.4、1000/.4、2000/.4、2000/.8、2000/关闭过滤五个解码组合。固定cellprob=0、score=.25、min_size16、max_instances64及官方PQ，不使用test。字段缓存重用减少重复DINO前向，但长积分和十次完整官方评测仍消耗额外租赁时间。

从原Flow、v2固定解码结果和十个解码候选中选择完整val官方PQ最高者为`flow_selected`。旧Flow永久保留；不得因为v2较新而强行选它。报告同时列出所有候选、旧Flow训练成本、v2训练成本及继承成本。v2预算扩大、解码搜索追加，**不声称等预算公平**；选择与报告使用同一val，最佳分数存在选择乐观偏差。未见test成绩前不预测排行榜提升。

背景与最终mask的前景召回诊断用于区分网络漏检和流重建损失。可交付报告包含PQ/Dice/IoU、所有候选表、同场景同ROI可视化和产物完整性检查。

## 在 AutoDL 上追加，不打断当前训练

先在新源码目录拉取本次Git发布（旧运行目录不动）：

```bash
cd /root/autodl-tmp/solar_filament_source_clean
env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 \
  git -c http.proxy= -c credential.helper= -c http.version=HTTP/1.1 pull --ff-only origin linux-autodl
bash linux/start_flow_followup.sh /root/autodl-tmp/solar_filament_linux
tail -f logs/flow_followup_console.log
```

等待进程不创建CUDA上下文、不占GPU。它固定启动时的commit及源码文件指纹；等待期间不要再pull/编辑这个checkout。重复启动由文件锁拒绝。原runner的真实PID/命令行与工作目录仍存活时一直等待；只有六组全部completed、没有真实runner/trainer、停止请求不存在且旧GPU/runner锁均可独占时才运行。旧队列失败或暂停则标blocked并退出，需先处理旧队列。

只共享只读输入目录：`data`、`output/pretrained`、`output/splits`。所有新训练、缓存、状态、日志和报告写入新checkout的`output/benchmark/flow_followup_v1`，绝不链接整个旧output。复用原`.envs/solar-routes/bin/python`，没有额外pip依赖或资产下载。磁盘需至少16GiB空闲以容纳两组完整原始字段缓存和新checkpoint；CPU/RAM也参与生成流目标与缓存。

## 状态、停止、恢复

```bash
cd /root/autodl-tmp/solar_filament_source_clean
cat output/benchmark/flow_followup_v1/queue_status.json
cat output/benchmark/flow_followup_v1/pipeline_status.json
tail -f output/benchmark/flow_followup_v1/flow_v2/train_console.log
```

等待期间只有queue_status；正式启动后pipeline_status和train_console.log才出现。停止新追加器与新训练：

```bash
touch output/benchmark/flow_followup_v1/stop_requested.json
```

训练会在梯度累积边界保存原子checkpoint（optimizer、RNG、epoch排列/offset）；验证中先完成本次验证。检查进程退出及last.pt更新时间后才关机。不得用旧`linux/stop.sh`停止新实验，也不得把新checkpoint复制到旧Flow目录。确认安全停止后，保留相同源码commit，删除新namespace的stop_requested.json再执行start_flow_followup.sh。解码搜索在当前一个完整观测/候选完成后可重新执行并复用已缓存产物；旧队列不受新停止请求影响。

完成后查看`deliverables/flow_followup_v1/comparison.md`、`results.csv`、`all_candidates.csv`和`selected_flow_baseline.json`。最终flow_selected可能仍是旧Flow；以实测结果为准。

参考：[Cellpose官方参数与实现](https://cellpose.readthedocs.io/en/latest/_modules/cellpose/models.html)、[Cellpose3论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11903308/)。论文中的长细细胞使用更长积分支持这里的候选设计，不能直接证明太阳暗条效果。
