# Panthera 随机单次回球训练

当前任务是 `PantheraHorizontalReturn`，owner 配置为 `panthera_horizontal/mujoco`。训练复用 UniLab Manager-Based 环境和 RSL-RL PPO，只注册 MuJoCo。随机发球仍来自原 `ServeLauncherCfg`，没有缩小为固定球。

## 当前控制分工

确定性准备同步控制导轨、joint1 和 joint234，joint5 固定。实测拍心、关节位置/速度、拍面、预测新鲜度和持续稳定窗口共同确认准备完成。

准备完成后，PPO 的六维动作控制：

| 动作 | 含义与当前范围 |
|---|---|
| 0 | 参考轨迹时间推进；逐渐暂停、延迟或加速，速率范围 0–1.02 |
| 1 | joint6 拍面倾角修正，±0.03 rad |
| 2 | joint1 位置残差，±0.015 rad |
| 3–5 | joint2/joint3/joint4 位置残差，各 ±0.015 rad |

导轨保持确定性参考，不由这版 PPO 做残差调整。残差锚定名义轨迹，不随动作累积漂移。零动作推进参考时钟、使用零残差；初始 actor 均值为零，网络和探索参数仍可学习。轨迹速度、加速度按实际时钟速率转换，实际运动持续接受当前 URDF 的位置、速度与力矩检查。

观测包含关节、球、球拍、预测击球点和时间、准备侧、阶段、参考状态、时钟与残差状态，以及五帧球状态历史。删除观测列或改变六维动作含义会破坏已有 checkpoint 合同。

物理碰撞与球路按 0.25 ms 子步采样，策略周期为 10 ms。首次触拍后继续观察；接触结束后停止策略修正并制动，直到实际首次落台或失败。过网需完整球体离网，成功需首次落在对方台内；二次触拍、机械臂接触、落地等由真实接触判定。

## Reward

新任务继承 `panthera_return/base.yaml` 的事件及预测引导；`panthera_horizontal/base.yaml` 覆盖残差成本并加入动作质量项。

| 项目 | 权重与触发 |
|---|---|
| 首次有效触拍 | +0.1，一次 |
| 实际过网 | +0.5，一次 |
| 实际合法落台 | +10，一次 |
| 失败 | −5，一次 |
| 触拍前接近候选点 | 最高 +0.30/s；2 cm 容差，临近击球衰减 |
| 触拍前主动挥拍 | 最高 +0.20/s；实际球拍速度、接近程度及己方反弹条件 |
| 离拍方向、预测过网、预测落点 | 最高 +0.20、+0.50、+1.30，一次 |
| PPO 修正幅度、指令变化成本 | −0.02/s、−0.01/s |
| 实测动作形式成本 | 最高 −0.50/s |
| 成功后的连续动作质量 | 最高 +2，一次；失败不给此项 |

事件项不按时间步缩放；持续项内部乘控制周期。失败回合的正向辅助奖励上界不足抵消失败惩罚，但这不保证没有局部最优。

离拍评分使用完整接触结束后的首个无接触样本，一次性计算。方向评分在 [−1, 1]，其余在 [0, 1]；自由飞行预测不能预先涵盖后续触网或机器人碰撞。重力和评分尺度归 owner YAML 的 `shaping_cfg` 所有。实际接触和落台决定成功，预测提供引导。

## 运行入口

以下命令在 UniLab 目录。当前环境引用相邻 UniSim editable 源码，新的公开跟踪与观测接口尚未发布为正式包。需要同步时保留 `table-tennis` group 和现有后端 extras：

```bash
uv sync --locked --group table-tennis --extra mujoco --extra motrix --extra uni_rl

uv run --no-sync python -m unilab.scripts.train_rsl_rl \
  task=panthera_horizontal/mujoco algo.num_envs=16 \
  algo.num_steps_per_env=64 algo.max_iterations=50 algo.save_interval=10 \
  training.no_play=true training.log_dir=../runs/ppo_horizontal_new_run
```

评估必须保留 checkpoint 同目录的 `run_config.json`，恢复 actor、观测归一化和合同。初版模型评估：

```bash
uv run --no-sync python scripts/tools/evaluate_panthera_return.py \
  --task panthera_horizontal/mujoco --mode policy \
  --checkpoint ../runs/panthera_horizontal/model_49.pt \
  --episodes 1000 --num-envs 32 --seed 20007 --paired-serves \
  --output-dir ../runs/eval_model49_seed20007
```

零修正对照使用相同任务、checkpoint 快照、seed 和球数，将 `--mode policy` 改为 `--mode zero`，另选输出目录；零动作是名义参考轨迹。严格恢复检查结构和归一化；比较模型仍需核对源码和场景指纹。

macOS 原生可视化：

```bash
uv run --no-sync mjpython -m unilab.scripts.play_interactive \
  --algo ppo --task panthera_horizontal --sim mujoco \
  interactive.action_mode=policy \
  algo.load_run=../runs/panthera_horizontal \
  algo.checkpoint=49 interactive.speed=0.5 \
  interactive.camera_follow_body=false interactive.camera_distance=8 \
  interactive.camera_elevation=-20 interactive.camera_azimuth=90
```

空格暂停/继续，`N` 单步，`+`/`-` 调整速度；关闭窗口退出。

## 评估含义与当前证据

`--paired-serves` 按回合编号产生并核对球库，发球不随策略结束时间变化。统计包含全部分配回合和准备失败，输出左右侧、横向位置及速度分组、成功率和置信区间。每次运行应另选输出目录。

`success_rate` 是实际合法率；`horizontal_legal_rate` 同时要求合法、URDF 边界及数值动作门槛通过。新任务的 `acceptance_rate_met` 使用后者。数值门槛是探索性筛选，尚未按视频标定，不能代替视频动作判断。至少 1,000 个独立样本并达到 95% 是当前统计目标；有限样本 100% 不证明整个连续分布 100%。

2026-10-07：初版完成 51,200 transitions。配对种子 10007 的 200 球为 198/200 合法，零残差参考为 200/200；独立种子 20007 的 1,000 球正式结果为 972/1000，95% Wilson 区间约 [95.98%, 98.06%]。1,000 球全部在触拍前完成准备，实际 URDF 边界检查通过。数值动作共同筛选为 0/1000，主要是随挥长度略超过 15 cm 等门槛。

28 条正式失败均为 `below_net`；延续同一策略、同一来球的实际物理后，均首次落在对方台内。完整离网后重复检查整个球的最低点高于网顶，造成误判。修正尚未实施，正式成绩保持 972/1000；延续诊断不能改写成重新评估所得的 1000/1000。历史原始评估及诊断记录已清理，本段保留当时的结果说明。

RSL rollout 日志的批次均值不等于独立回合成功率；部分后出现的终局指标还可能未被上游日志记录。验收使用独立评估的逐回合统计。

## 旧任务和诊断入口

`PantheraReturn` / `panthera_return/mujoco` 是旧直接速度控制任务，动作包括 joint1/joint6 速度和导轨/joint234 准备姿态残差；它与新任务观测和动作语义不同，同为六维也不能交换 checkpoint。

`return_feasibility.py` 是旧名称兼容重导出；共享准备和 Hermite 控制在 `stroke_control.py`。搜索、重放及旧回归测试仍依赖这些入口。

2026-10-06 的旧任务短程训练为零合法回球，曾比较离拍辅助奖励对前向出球速度的影响；不能代表当前横拍策略。相关历史实验产物已清理。正式球拍接触参数未因当时低有效回弹推断而调参。

## 本地源码回归

```bash
uv run --no-sync pytest -q ../tests tests/tasks/manipulation/table_tennis tests/tasks/test_panthera_return_training.py
UNILAB_LOCAL_UNISIM="$(cd ../UniSim && pwd)" make check
UNILAB_LOCAL_UNISIM="$(cd ../UniSim && pwd)" make test
```

`UNILAB_LOCAL_UNISIM` 使用已有本地依赖来源校验，不修改来源断言。通用框架测试可能还需要 Allegro 等其他机器人资源；缺失资源要单独报告。
