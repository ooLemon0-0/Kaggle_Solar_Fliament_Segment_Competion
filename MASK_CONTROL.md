# M0：原 mask head 的匹配训练对照

目的：现feature_control为RF最佳checkpoint＋原mask head＋细节分支，再训练30轮，PQ .422718；缺少“只训练原mask head同样30轮”的对照。M0检验独立分支收益，不改变split/PQ或重新训练骨干。

## 固定实验

- 原RF best.pt（原best epoch20、PQ .411695）作为同一warm start；校验checkpoint配置、split、best epoch、非debug及SHA256。
- 输入1056，seed42，batch1/累积4，head LR1e-4，原AdamW/增强/调度保持。
- 骨干、query decoder、分类和框回归头冻结；只有原segmentation_head可训练。
- 无DetailHead，无额外图像投影，无前景/实例拉推辅助损失。
- 原565train观测/936组、142val观测/218组，2048官方PQ、confidence .25、max64、mask logit0不变；不用test。
- 新增30epochs或4小时样本准备+计算先到者停；继承RF 2.575h单列。预检/验证额外记录，累计6h软墙钟上限，进行中的预检/验证可以完成。超时/停止请求在积累边界保存last.pt；不覆盖旧结果。
- 开始前最大目标数样本8步反传：核验只有原mask head收到有效梯度和更新，冻结权重逐张量保持；native预测格式也检查。预检属于debug，不计正式分数。
- 每2epochs和最终自动完整val；结束写与原RF/细节分支的对照、继承/新增成本、完整性检查和下一步判断，不自动启动D-native。

## 发布验证记录

本机短GPU预检（RTX5070Ti16GB，仅debug）：26目标，8次loss 13.646849→12.983643，峰值5057.335MiB；所有冻结参数未变、原mask head有效更新，native预测64个有效实例。此结果不代表AutoDL训练已启动或获得新PQ。

CPU测试覆盖实际可训练范围/同一checkpoint初始输出/冻结权重/梯度、与旧细节组的配置预算对齐、继承成本、预算不足不能作匹配结论，以及既有RF1536/续训/队列/投影保护。服务器仍会重新做真实预检。

## 线上获取并启动

原训练目录 `/root/autodl-tmp/solar_filament_linux`；既有Git目录 `/root/autodl-tmp/solar_filament_source_clean`；训练Python用原目录 `.envs/solar-routes/bin/python`。

不merge正在运行的checkout。线上fetch后提取启动器，它创建独立固定提交工作树 `/root/autodl-tmp/solar_filament_mask_control`。共享原data/pretrained/splits，只在新namespace写结果。原GPU/runner锁、追加锁、真实Python命令/创建时间交叉检查，存在RF1536等任务时拒绝抢占和重复启动。

```bash
(
cd /root/autodl-tmp/solar_filament_source_clean || exit 1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
solar_git() {
  timeout 90 env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 \
    git -c http.proxy= -c credential.helper= -c http.version=HTTP/1.1 \
    -c http.lowSpeedLimit=1 -c http.lowSpeedTime=20 "$@"
}
solar_ok=0
for solar_attempt in 1 2 3; do
  if solar_git fetch --progress \
    https://github.com/ooLemon0-0/Kaggle_Solar_Fliament_Segment_Competion.git \
    refs/heads/linux-autodl:refs/remotes/origin/linux-autodl; then
    solar_ok=1; break
  fi
done
[ "$solar_ok" -eq 1 ] || exit 1
git show origin/linux-autodl:linux/start_mask_control_online.sh > /tmp/solar-mask-control.sh &&
bash /tmp/solar-mask-control.sh "$PWD" /root/autodl-tmp/solar_filament_linux
)
```

启动器先CPU核验，再nohup启动一个M0进程。出现PID只表示启动过程，不表示GPU预检已经通过；看PREFLIGHT/训练/验证日志确认。没有CPU等待器自动抢占其他实验，若原GPU忙或锁占用会明确拒绝；不杀既有RF1536。没有重开Codex定时任务。

## 一次查询及日志

```bash
/root/autodl-tmp/solar_filament_linux/.envs/solar-routes/bin/python \
  /root/autodl-tmp/solar_filament_mask_control/linux/status_mask_control.py
```

```bash
tail -n 0 -F /root/autodl-tmp/solar_filament_mask_control/logs/mask_control_console.log
```

status核验PID/创建时间/cwd/脚本名，输出当前epoch、sample/validation image、最佳PQ/Dice/IoU、最近日志/checkpoint活动。Ctrl+C只退出查看。无需再次start来查看。

## 正式结果、停止与恢复

```bash
cat /root/autodl-tmp/solar_filament_mask_control/output/benchmark/mask_control_v1/comparison.json
```

重点看mask_control.pq、detail_reference.pq、delta_detail_minus_mask_control、matched_epochs_and_updates和继承/新增成本。若预算先到且少于30轮/更新数不同，报告明确拒绝归因。若同训练量且差距≤.002，单seed未分清；细节明显胜出才继续将细节/投影读native2048，M0更好则改用更简洁主线。单seed和单次val选择不是显著性/泛化证明；下一步仍需成对误差和尺寸分层诊断。

结果RLE/142覆盖/split/预算由artifact_checks核验，PQ仍为原官方全验证。M0报告不改变原六组排名文件。

停止：

```bash
touch /root/autodl-tmp/solar_filament_mask_control/output/benchmark/mask_control_v1/stop_requested.json
```

核实真实进程退出且last.pt存在、累计预算尚未耗尽，再在同固定源码恢复：

```bash
bash /root/autodl-tmp/solar_filament_mask_control/linux/start_mask_control.sh \
  /root/autodl-tmp/solar_filament_linux --resume
```

断点必须包括optimizer/RNG/order/offset，缺失但有正式训练证据则拒绝从头覆盖。异常退出的未闭合墙钟区间保守计到恢复时，可能包含停机时间；预算用尽需重新决策，不能静默清零。

训练完成但报告异常，只重建缓存报告：

```bash
/root/autodl-tmp/solar_filament_linux/.envs/solar-routes/bin/python \
  /root/autodl-tmp/solar_filament_mask_control/scripts/run_mask_control.py \
  --active-project /root/autodl-tmp/solar_filament_linux --report-only
```

只做GPU预检：在独立工作树给start_mask_control.sh追加--preflight-only。已开始正式训练后必须--resume，不能用预检开关绕过断点保护。
