# 投影输入梯度修复与串行补跑

`image_projection`失败于RF-DETR的`nested_tensor_from_tensor_list`：把带梯度的投影输入复制到多视图切片时触发PyTorch的in-place/view限制。固定1056输入直接封装为`NestedTensor(image, 全False padding mask)`，不detach、不no_grad；冻结RF参数但保留损失到投影头的梯度。冻结骨干为eval，内部梯度检查点因此关闭；在冻结RF前向外使用非重入checkpoint节省激活显存，并保留随机状态。RF/Feature/Flow/SAM其他路线行为不变。

补跑须通过最大标注目标数样本8步debug反向，明确检查投影梯度非零且有限、RF无参数梯度，随后按原30epochs/4准备计算小时预算、1056输入、seed42和原split训练与完整142观测218组val官方2048PQ。预检不是正式PQ，不增加正式训练预算。此实验仍在缩放到1056后投影，没有验证原尺寸/超高分辨率投影。

## 在线更新并重新排队（当前训练无需停）

已固定版本的旧Flow等待器不能直接pull覆盖。先fetch（只下载Git对象，不替换源码），从刚下载的提交提取CPU取消脚本，确认只存在等待器且Flow尚未开始，发送其专属取消文件并等待真实进程退出，再快进合并，重排队。取消脚本不终止或修改原runner/训练子进程。若Flow已有训练产物或运行子进程，自动拒绝源码升级。任何失败则停止后续步骤。

在AutoDL粘贴：

```bash
cd /root/autodl-tmp/solar_filament_source_clean
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
timeout 90 env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 \
  git -c http.proxy= -c credential.helper= -c http.version=HTTP/1.1 \
  -c http.lowSpeedLimit=1 -c http.lowSpeedTime=20 fetch --progress \
  https://github.com/ooLemon0-0/Kaggle_Solar_Fliament_Segment_Competion.git \
  refs/heads/linux-autodl:refs/remotes/origin/linux-autodl &&
git show origin/linux-autodl:linux/requeue_projection_online.sh > /tmp/solar-requeue-projection.sh &&
bash /tmp/solar-requeue-projection.sh "$PWD" /root/autodl-tmp/solar_filament_linux
```

正常打印`WAITING_FOLLOWUP_CANCELLED_SAFE_TO_MERGE`、新commit、`FOLLOWUP_INPUTS_READY ... repair_projection=True`和新等待器PID。`WAIT_ORIGINAL feature_control`或`feature_structure`表示已排队，尚未占GPU。没有SSH接入时，本地发布代码不代表服务器已经排上，以上输出才是确认依据。

执行顺序：剩余feature_control → feature_structure → image_projection修复补跑 → Flow v2及解码搜索。

## 保留结果和安全边界

等待原队列真实进程全部退出。只有所有其他路线completed、image_projection单独failed、原stage=needs_attention且无停止请求时才启动修复。其他路线失败、paused、queued均阻断，不伪造completed。原runner/GPU锁由等待器全程独占，修复子进程必须有带创建时间的活守护进程；重复等待器被文件锁拒绝。

补跑从新Git目录加载源码，共享原输入资产；**原训练源码不覆盖**。投影输出仍在原`output/benchmark/routes_v3/image_projection`，所以原监控命令继续有效。先完整备份失败目录（含last/best/checkpoint/log/config）及队列状态/报告至原`output/benchmark/routes_v3/repair_history/image_projection_<timestamp>`，日志追加。

有完整last.pt时恢复optimizer/RNG/排列offset/累计预算，不重置训练计时；没有last.pt且没有best.pt、best_metrics或training_log才允许新训练。存在不可恢复训练产物时拒绝覆盖，报blocked。修复预检写新记录，旧记录已归档。完成后更新原六路线状态和报告；失败或暂停不启动Flow。`repair_record.json`记录源码commit、备份路径与是否续训，`repair_failure.json`记录新失败；原failure.json保留作历史证据。

新队列运行/等待期间继续固定commit，不再次pull。取消等待器只有在尚未开始追加训练时可执行；补跑期间需要停止时用原`linux/stop.sh`，会在累积边界保留last.pt并阻断Flow，恢复须检查实际进程及相同源码。

查看整个后续队列（目录尚未创建的文件由-F重试）：

```bash
tail -n 5 -F \
  /root/autodl-tmp/solar_filament_source_clean/logs/flow_followup_console.log \
  /root/autodl-tmp/solar_filament_linux/output/benchmark/routes_v3/{feature_control,feature_structure,image_projection}/train_console.log \
  /root/autodl-tmp/solar_filament_source_clean/output/benchmark/flow_followup_v1/flow_v2/train_console.log
```

补跑器控制台会打印`START_PROJECTION_REPAIR`、`PROJECTION_REPAIR_OFFICIAL`及`PROJECTION_REPAIR_COMPLETED; next Flow v2`。原status.py正常显示新投影训练与最佳PQ。Ctrl+C只退出查看。无额外pip安装、无test调参、无Kaggle提交、无Codex定时通知。
