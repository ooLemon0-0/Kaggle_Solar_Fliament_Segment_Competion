# P0：原生图投影，作为DINO双头的共同起点

只运行一组P0，不继续铺开模型。RF-DETR1536最佳正式断点提供全图queries、类别、框；这些权重全部冻结。新增轻量卷积投影直接读取原生2048灰度图，stride4生成512×512特征，与RF上下文融合，用原实例mask head生成独立实例掩码。训练原mask head与新投影；不加入DINO、分离/连接辅助损失或二值连通域。最终512网格logits插值到2048评测，不能称为2048全分辨率logits。

原图/原2048实例标注同步旋转和翻转；只RF上下文支路缩放到1536。训练mask采样保持RF原坐标和nearest采样，通过每次最多8个匹配分块读原标注，避免group匹配把同一2048标注复制数百份。只在P0进程内替换训练采样函数，不修改安装包、原始数据或官方PQ。native投影可微，融合残差零初始化；训练头输出网格明确使用native2048尺寸，避免被RF1536下游重新降为384。

固定seed42、原565train观测/936标注组、142val观测/218组、官方2048PQ、阈值.25/最多64实例。新增预算12epochs或2个样本准备+计算小时先到者停，积累4；满12轮2808updates。累计4小时软墙钟，进行中预检/验证可以结束；验证时间不算训练小时。继承RF1536的实际训练时间另报。每2轮与最终全val；best取PQ，自动产物检查、PQ/Dice/IoU、尺寸分层best-mask-IoU诊断、RF1536/GT/P0同图和成本报告。**没有自动启动下一组**。

P0比RF1536的差值包含额外mask适配、掩码网格提高和投影分支，不能单独归因为投影。原M0是1056/30轮，其结果也不能代替此组的同预算1536控制。后续若值得推进，按同初始化/预算加入DINO(D0)，再分别试分离、连接头；先取得P0完整结果和失败图再决定。阈值.005只是开支门槛，不是单seed显著性。纯自监督双头尚待旧任务样本定义；默认新双头采用train实例监督，不声称无监督。

## 安全启动：线上Git、独立工作树

服务器M0真实状态未知。以下启动器获取原GPU/runner、Flow队列锁并检查真实进程；若M0正在运行就拒绝，不停止它，不删除lock文件。不得用原运行目录直接pull/merge。先在线fetch，提取下面脚本到/tmp执行；新工作树固定该提交。已存在不同提交的新目录会拒绝覆盖。

```bash
(
cd /root/autodl-tmp/solar_filament_source_clean || exit 1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
timeout 90 env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 \
  git -c http.proxy= -c credential.helper= -c http.version=HTTP/1.1 \
  -c http.lowSpeedLimit=1 -c http.lowSpeedTime=20 fetch --progress \
  https://github.com/ooLemon0-0/Kaggle_Solar_Fliament_Segment_Competion.git \
  refs/heads/linux-autodl:refs/remotes/origin/linux-autodl &&
git show origin/linux-autodl:linux/start_native_projection_online.sh > /tmp/solar-native-projection.sh &&
bash /tmp/solar-native-projection.sh "$PWD" \
  /root/autodl-tmp/solar_filament_linux /root/autodl-tmp/solar_filament_rf1536
)
```

需要rf_resolution_v1所有检查通过、30epochs/7020updates完成、原split/config一致。读取实际RF1536 best.pt哈希并固定在manifest；加载时核验checkpoint config/epoch/PQ/split，随后在实际1536/2048、最大26目标上做8步反传和预测/RLE预检，预检权重丢弃再seed42正式构建。预检失败不开始正式训练，不暗降尺寸。

```bash
/root/autodl-tmp/solar_filament_linux/.envs/solar-routes/bin/python \
  /root/autodl-tmp/solar_filament_native_projection/linux/status_native_projection.py
tail -n 0 -F /root/autodl-tmp/solar_filament_native_projection/logs/native_projection_console.log
```

Ctrl+C退出tail不会停止训练。报告在新工作树output/benchmark/native_projection_v1/comparison.json、comparison.md、paired_mask_diagnostics.json、same_scene_comparison.png；可交付文件另复制deliverables/native_projection_v1。

## 停止与恢复

向新namespace写stop请求，会在积累边界原子保存last.pt后暂停；不中断旧模型。

```bash
printf '{"reason":"user_pause"}\n' > \
 /root/autodl-tmp/solar_filament_native_projection/output/benchmark/native_projection_v1/stop_requested.json
```

确认真实进程退出后，在**原提交**恢复optimizer、RNG、epoch排列/offset和累计预算。不得覆盖从头训练；缺last.pt但有正式结果会阻断。

```bash
cd /root/autodl-tmp/solar_filament_native_projection
bash linux/start_native_projection.sh /root/autodl-tmp/solar_filament_linux \
  /root/autodl-tmp/solar_filament_rf1536 --resume
```

完成训练但报告异常可同目录原Python运行scripts/run_native_projection.py --active-project ... --rf-project ... --report-only，只读缓存重建报告，不再训练。安全锁/实际进程检查仍生效。

## 本地验证记录（debug，不是正式PQ）

2026-10-10，38项CPU回归通过，两个启动脚本语法通过。RTX5070Ti16GB，本地仅有RF1056最佳权重，因此context1056/native2048代理预检：26目标、8步、loss14.594047→13.670963，峰值7923.900MiB；原图投影有梯度且pixel stem更新，冻结定位参数逐项完全不变；512mask grid、64个非空2048 RLE预测。初版原RF采样重复展开标注OOM，分块等价采样后通过。原生路径可以在16GB运行不等于服务器1536完整组合已通过；必须由启动器重做真实RF1536预检。没有新的正式PQ或服务器启动证明。Codex定时监控保持停止。
