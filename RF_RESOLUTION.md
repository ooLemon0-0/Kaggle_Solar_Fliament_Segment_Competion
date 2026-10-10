# 第一步：只验证 RF 输入分辨率

2026-10-10：现有七组已结束。RF 在 epoch20 最好 .411695，epoch30降至 .400603；细节分支最近六次验证 .421315～.422714，稳定领先。当前不支持先把所有组延长训练。Flow v2 原前景召回 .822004、重建召回 .766074，但同 n2000_f04 解码 PQ .387639 低于旧 Flow .399941；前景更完整不等同实例质量更好。SAM 理想框 mean IoU .706224、15%扰动框 .658151，仅用于诊断，不能当正式PQ或全验证上限。

## 唯一正式实验

- RF-DETR-Seg Small，RF-DETR固定1.11.2；同一原始实例预训练权重，**不继承已训练RF checkpoint**。
- 只改变输入1056→1536；原encoder LR1e-5、head LR1e-4、AdamW、增强、batch1/累积4、seed42保持。
- 保留565train观测/936标注组、142val观测/218组、原尺寸2048官方PQ；阈值与64个最大实例保持原协议。不使用test。
- 最大目标数样本先做8步反传和native预测预检。属于debug，不计正式PQ；OOM拒绝暗降尺寸。
- 正式最多30epochs或4小时样本准备+模型计算；每2epochs和最终完整验证。
- 预检、验证、报告与暂停续跑计入累计6小时软墙钟上限。超时请求在梯度累积边界安全暂停；进行中的预检/完整验证可以完成，故不是硬切断保证。不覆盖旧结果。
- 原子断点复用已有engine，保存optimizer/RNG/epoch排列/offset。缺失last.pt但有正式训练证据时拒绝重启。异常退出的未闭合墙钟区间保守计至恢复时，包含停机时间；因此可能需要先重新审核预算。
- 整个实验持有原GPU/runner锁与追加队列锁，并重新检查实际训练/等待进程；重复启动拒绝。仅本实验，不自动接续细节/SAM/Flow/YOLO。

## 线上获取并启动

采用此前成功的无代理、隔离Git配置、HTTP/1.1、超时fetch。**不向原训练目录或旧Flow源码目录merge新文件**；独立detach worktree固定提交。

```bash
cd /root/autodl-tmp/solar_filament_source_clean
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
solar_git() {
  timeout 90 env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 \
    git -c http.proxy= -c credential.helper= -c http.version=HTTP/1.1 \
    -c http.lowSpeedLimit=1 -c http.lowSpeedTime=20 "$@"
}
solar_downloaded=0
for solar_attempt in 1 2 3; do
  if solar_git fetch --progress \
    https://github.com/ooLemon0-0/Kaggle_Solar_Fliament_Segment_Competion.git \
    refs/heads/linux-autodl:refs/remotes/origin/linux-autodl; then
    solar_downloaded=1; break
  fi
done
if [ "$solar_downloaded" -eq 1 ]; then
  git show origin/linux-autodl:linux/start_rf_resolution_online.sh > /tmp/solar-rf-resolution.sh &&
  bash /tmp/solar-rf-resolution.sh "$PWD" /root/autodl-tmp/solar_filament_linux
else
  echo 'Online fetch failed; no training started.'
fi
```

启动器先做CPU完整性/实际进程检查，再以nohup启动单个RF进程；前台窗口关闭不停止训练。原目录和旧Flow实验文件保持原状。独立目录 `/root/autodl-tmp/solar_filament_rf1536` 在运行/暂停期间不得pull/merge/切换commit。启动输出的PID本身不是已经通过GPU预检的证据，以实时日志为准。

## 查看一次状态、实时日志、停止/续跑

```bash
/root/autodl-tmp/solar_filament_linux/.envs/solar-routes/bin/python \
  /root/autodl-tmp/solar_filament_rf1536/linux/status_rf_resolution.py
```

```bash
tail -n 0 -F /root/autodl-tmp/solar_filament_rf1536/logs/rf_resolution_console.log
```

仅看新增日志，Ctrl+C只退出查看。完整验证产生OFFICIAL PQ；结束自动写对照结果，不自动启动第二组。

安全停止：创建该实验自己的stop文件，engine在梯度累积边界保存断点。不要使用原队列的stop.sh。

```bash
touch /root/autodl-tmp/solar_filament_rf1536/output/benchmark/rf_resolution_v1/stop_requested.json
```

确认实际进程退出、last.pt完整，且累计预算尚未耗尽，再从**同一固定源码**续跑：

```bash
bash /root/autodl-tmp/solar_filament_rf1536/linux/start_rf_resolution.sh \
  /root/autodl-tmp/solar_filament_linux --resume
```

若仅想先做GPU预检，创建worktree后给start_rf_resolution.sh追加 `--preflight-only`；它不正式训练。后续去掉该参数启动会使用已有成功预检记录。已开始正式训练则必须--resume，不能用预检开关绕过断点保护。

## 收尾与晋级

产物在独立checkout的 `output/benchmark/rf_resolution_v1`，单组checkpoint位于 `rf_1536`。

```bash
cat /root/autodl-tmp/solar_filament_rf1536/output/benchmark/rf_resolution_v1/comparison.json
```

自动核验完整142观测覆盖、原split、native RLE与正式预算记录；输出best PQ/Dice/IoU、epoch、更新数、训练耗时、相对RF1056的PQ差值。报告失败时检查failure.json及artifact_checks，不能仅凭训练completed宣称评测交付完整。

若训练completed而报告异常，只重新生成缓存报告，不重复训练：

```bash
/root/autodl-tmp/solar_filament_linux/.envs/solar-routes/bin/python \
  /root/autodl-tmp/solar_filament_rf1536/scripts/run_rf_resolution.py \
  --active-project /root/autodl-tmp/solar_filament_linux --report-only
```

晋级要求：30epochs完成、检查通过、PQ至少提高.005，并由后续成对尺寸分层诊断确认细目标召回改善。脚本只能判断PQ/预算，不会声称已经算出细目标召回。预算截停未满30epochs明确标无法作等epoch分辨率结论。该门槛是投入判断，不是统计显著性。通过后再设计1536细节分支；不通过先看错误，不立即推进2048或追加所有路线。

源码与CPU保护测试通过不代表AutoDL24GB的1536GPU预检通过；后者必须由服务器真实日志确认。原对照跨Windows/Linux续训，因此不保证逐位可复现，实际版本、预训练哈希、成本全部保留。

尺寸规则参考：[RF-DETR官方训练参数](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/)。
