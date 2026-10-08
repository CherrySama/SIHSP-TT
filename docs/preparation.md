# 预测与准备阶段记录

本文保留截至 2026-10-05 的预测、导轨、转腰和高度准备说明。文中的“当前”“尚未”均描述当时阶段；现有横拍 PPO 入口、评估与限制请看 [项目首页](../README.md) 和 [回球训练说明](../UniLab/src/unilab/tasks/manipulation/table_tennis/RETURN_TRAINING.md)。

# SIHSP-TT
Skill-Initialized Hierarchical Self-Play for Two-Robot Table Tennis

## 球路预测第一阶段

核心直接放在 `src/`，没有额外的 `sihsp_tt/` 包层：

```text
src/
  ball_state.py        # 带采集时间戳的球桌坐标系状态
  ball_prediction.py   # 与仿真、机器人框架无关的预测核心
  rail_alignment.py    # 左右对准目标、限速制动与预定位判定
tests/
  test_ball_prediction.py
UniLab/
  src/unilab/tasks/manipulation/table_tennis/prediction_adapter.py
  scripts/tools/preview_panthera_ball_launcher.py
```

根项目可独立安装，只依赖 NumPy：`pip install -e .`。导入使用
`from ball_state import BallState` 和 `from ball_prediction import predict_ball`。
UniLab 通过可选的 `table-tennis` 依赖组引用核心，不需要改搜索路径。

### 查看预测

在 macOS 的 UniLab 目录运行：

```sh
uv run --group table-tennis --extra mujoco mjpython scripts/tools/preview_panthera_ball_launcher.py --predict --seed 7
```

青色表示已观测轨迹，橙色表示未来预测，淡紫色是预计对方反弹、紫色是预计己方反弹，绿色是
`x=-1.39 m` 截面的候选接球点。每次发球重置观测历史与反弹事件状态。
默认仅在此预览中关闭机器人碰撞；主环境保留碰撞。
`--robot-collisions` 可显示完整场景中的被动碰撞，但球路预测尚不包含机器人接触。

### 输入和有效性

球桌坐标原点位于桌面中心的地面投影：X 朝对方、Y 朝机械臂看向对方时的左侧、Z 向上。
位置与速度描述球心，单位分别为 m 和 m/s。时间戳使用采集时刻，规划时刻
`now` 必须在同一时钟上；核心会传播采集至规划之间的延迟，不能用接收时刻替换采集时刻。
真实系统应先完成球桌坐标变换与状态估计，再提供 `BallState`。

`predict_ball(state, cfg, has_bounced=False, now=...)` 输出预计过网、己方首弹与候选接球位置和
绝对时刻，以及从当前规划时刻开始的轨迹。配置 `allow_opponent_bounce=True` 后，还会
计算此前的对方半台反弹并输出 `opponent_bounce`。`has_bounced` 专指己方已发生反弹，
对方反弹不改变这个标志。它是状态估计器提供的过去事件，
单个向上运动快照不能区分发射与反弹。UniLab 适配器用连续观测的垂直速度负转正检测反弹，
它使用仿真真值，尚不是带噪声的视觉滤波器。环境重置后必须同时重置对应适配器。
适配器要求采集时间戳严格递增；重复或乱序帧被拒绝，无效帧保留此前有效的反弹线索，
已观测到己方第二次反弹或重复对方反弹后，候选点持续失效直到重置。

无效测量、未来时间戳、已过候选截面、碰网、不合法首次落点以及台面接触过渡期，
不会输出有效候选点。预测结果中的事件保留其绝对时刻，延迟情况下应检查事件是否已过去。
`valid=True` 表示通过球路几何检查，尚不代表机械臂可达或能够回球。

### 模型和验证边界

当前使用重力自由飞行与有效反弹模型，尚未加入空气阻力、显式旋转状态及视觉误差。
发射不施加初始旋转，桌面摩擦仍会自然产生旋转。
核心要求应用显式提供反弹参数；Panthera 适配器采用法向恢复系数 `0.90`、
对方弹切向保留率 `0.94`、己方弹切向保留率 `0.947`，并计入对方接触约 `5 ms`
的间隔与横向位移。参数是当前 0.25 ms MuJoCo 模型、小范围慢速发球的有效标定，
不能直接用于实机。接触期间不输出有效自由飞行预测。
扩展来球种类或更换接触模型后，需要重新标定和核验。

测试对照实际仿真接触和截面穿越，不用发球器未来落点作预测输入。验收上限为首次落点
三维误差 2 cm、候选点三维误差 3 cm、候选时刻误差 20 ms。预测以 100 Hz 更新，
物理仿真以 4,000 Hz 运行。球路验证隔离机器人碰撞。这一阶段的预测精度验证不包含
导轨和手臂运动；下方单独说明新加入的导轨横向预定位。

旧 2 ms 接触配置的历史验证（不代表当前配置）：2026-10-03，种子 `431003` 的 1,536 条随机来球，加上左右位置 9 档、速度 21 档、
反弹时间 3 档的 567 条网格来球，共 2,103 条。对全部 100 Hz 自由飞行观测更新进行比较，
最大首次落点三维误差 13.16 mm、候选点误差 4.80 mm、候选时刻误差 0.44 ms。
接触过渡期单独要求预测失效；首弹后要求不再重复预测首次反弹。
这些误差仅描述当前仿真窗口，不是实机精度或机械臂可达性结论。

### 慢速端线发球

默认发球起点 X=1.42 m，位于对方端线外 5 cm。球先在对方半台反弹，过网后在己方
距网 0.9–1.1 m 内反弹，再到达接球截面。这模拟两次反弹的发球球路，未模拟抛球或对方挥拍。
发射水平速度为 2.8–3.0 m/s，第一次反弹后约为 2.6–2.8 m/s；左右范围仍为 ±0.5 m。
采样己方落点及反弹最高点后，关联反算对方落点、发射高度和竖直速度。
己方反弹后最高点设计值为高出台面 30–34 cm；当前拍心截面的球心高度验收为 20–30 cm。
这些窗口不是任意独立组合，实际碰撞、过网和截面高度均需通过物理验证。

可先重复一个固定基准球，再查看默认随机球：

```sh
uv run --group table-tennis --extra mujoco mjpython scripts/tools/preview_panthera_ball_launcher.py --rail-align --fixed-serve
uv run --group table-tennis --extra mujoco mjpython scripts/tools/preview_panthera_ball_launcher.py --rail-align --seed 7
```

球桌接触使用直接刚度/阻尼参数与固定阻抗；物理时间步由 2 ms 降为 0.25 ms，机械臂目标
更新和球路预测仍为 100 Hz。预览在每个显示周期内推进 40 个物理步，避免每个物理步
同步界面。测试包含落球反弹、碰撞速度及碰撞时刻相位扫描，以及减半时间步对照。
球与桌面的数值接触穿入仍允许到 6 mm，此数值不是实测球体压缩量；不能称完全收敛或实机标定。

```sh
# 在 UniLab 目录，先安装所需依赖；保留原有 MuJoCo/Motrix 后端
uv sync --group table-tennis --extra mujoco --extra motrix
uv run --no-sync pytest -q ../tests/test_ball_prediction.py tests/tasks/manipulation/table_tennis/test_prediction_adapter.py
```

## 导轨横向预定位

在 UniLab 目录运行：

```sh
uv run --group table-tennis --extra mujoco mjpython scripts/tools/preview_panthera_ball_launcher.py --rail-align --seed 7
```

这个模式自动开启预测。浅蓝小球表示实际拍心，黄色小球和箭头表示与拍心同高度、
同前后位置的横向目标；绿色小球仍是球的预计位置，可能与黄色目标存在高度差。
系统以实际拍心的 X 坐标作为预测截面，每 10 ms 用球的当前状态更新预测。
导轨目标 = 当前导轨位置 + 预测球心 Y − 实际拍心 Y；手臂维持 home 控制目标，
不会主动调整 link1 或高度。导轨加速带来的轻微关节跟随误差会反馈到横向目标。

独立的 `src/rail_alignment.py` 不依赖 MuJoCo，只计算横向目标、速度/加速度限制下
的制动和预定位状态。使用到位保持区间，避免对毫米级目标变化反复加速；短暂台面
接触期间保留上一条未来目标，目标过期后减速停止。超出行程的目标会被标记，不能判为到位。

场景新增导轨力驱动，默认仍由 `rail_lock` 锁定；`--rail-align` 和下面的
`--waist-prepare` 会释放它。
场景为 joint5 加入固定约束，保留原有六关节控制接口。新导轨模式在加载时进一步将
link5 编译为角度 0 的固定连接，移除 joint5 自由度及其驱动，确保它不能相对 link4 转动。
该模式是独立的原生预览，尚未接入 Manager-Based 环境的自动预备流程。

导轨与基座安装中心位于 X=-1.53 m。滑块、导轨梁与支架中的最前缘是滑块前缘，
与己方桌沿 X=-1.37 m 的间隙为 5 cm；全行程保持这个间隙。导轨梁和两个支架
同步前移，使用现有尺寸与安装高度。

导轨暂定上限为速度 1.5 m/s、加速度 10 m/s²、驱动力 100 N，行程 ±0.75 m。
前移使球更早经过拍心所在截面，因此加速度上限由 8 调整为 10 m/s²，仍需按下面
的到位时刻与实际运动约束验收。
原生预览的物理步内环利用 MuJoCo 模型计算力前馈，再由驱动施力；没有在运动过程中
改写导轨位置或速度。该前馈属于仿真适配层，实机需替换为标定后的导轨控制接口，
这些数值也不是硬件规格。

验收：在球实际经过拍心所在前后位置之前至少 50 ms，横向误差 ≤1 cm、实际导轨速度
≤0.05 m/s；球经过该位置时也检查这两个条件。每个物理步检查实际速度、加速度、
驱动力、行程和手臂关节范围/速度，并检查 link5 的相对姿态固定。
慢速发球单独要求到位后至少还有 150 ms 的实际到达余量，测试包含固定基准球、
种子 `20261004` 的 64 条随机发球及左右、速度、己方落点、反弹高度的 16 条边界发球。
保留旧直接来球的控制回归测试，
还覆盖目标超行程、端点制动、目标过期和重置。默认隔离机器人碰撞。

原 `--rail-align` 模式仍只完成左右方向预定位。
此模式尚未实现高度对准、主动挥拍、合法回球判定、奖励或 PPO。

```sh
uv run --no-sync pytest -q ../tests tests/tasks/manipulation/table_tennis
```

## 导轨与转腰同步准备

在 UniLab 目录运行固定右侧来球；去掉 `--fixed-serve` 可看随机来球：

```sh
uv run --no-sync mjpython scripts/tools/preview_panthera_ball_launcher.py --waist-prepare --fixed-serve
# 保留完整碰撞，可观察球与静止准备拍面的被动接触
uv run --no-sync mjpython scripts/tools/preview_panthera_ball_launcher.py --waist-prepare --fixed-serve --robot-collisions
```

这个独立原生预览模式根据来球侧，将 link1 从 home 转到 ±90°，让球拍的一面基本
朝向对方球台；中央 ±3 cm 死区默认用右侧准备。导轨与转腰在同一个物理步开始，
导轨按最终转腰姿态的拍心 Y 补偿，预测截面固定在最终拍心 X=-1.53 m。
其他关节保持 home 控制目标，link5 保持固定连接。

转腰目标轨迹限速 3 rad/s、加速度 12 rad/s²，可用 `--waist-speed` 调整目标速度。
原生位置驱动加入速度及惯性耦合前馈，仍通过驱动施力，运动时不改写实际位置或速度。
这些值是仿真试验参数；实机前馈需要标定。Manager/RL 的关节速度配置保持原值。

准备验收使用实际拍心、实际关节位置/速度和实际拍面朝向，不用几何目标替代反馈。
要求横向误差 ≤1 cm、导轨和全部活动臂关节速度 ≤0.05、关节目标误差 ≤0.015 rad、
拍面轴与球台前后方向偏差 <8°，连续满足 30 ms；导轨速度单位为 m/s，关节为 rad/s。
预测超过 50 ms 未得到有效更新或明确判为无效球路时，不再报告当前准备到位。
到位时需预计还剩至少 150 ms，并在隔离机器人碰撞的实际球路中验证到达余量。

2026-10-05 的固定左右/中央 3 球、64 条随机球和 16 条边界球共 83 球全部通过；
最短实际提前到位余量 238.5 ms，link1 实测峰值 3.00318 rad/s（目标限速的跟随误差），
最大横向误差 0.00681 mm，最大导轨力 86.81 N。固定右侧球约 0.800 s 准备完成。
另跑完整碰撞的中央、左右 ±0.35/±0.5 m 共 5 球，只有球桌及球拍接触；右侧固定球
约 1.063 s 被动触拍，因此完整碰撞轨迹没有穿过预测截面，不能混用隔离球路的通过时刻。
仅 `--waist-prepare` 时未学习挥拍或验收合法回球，固定隔离球在截面的球心与拍心
仍有约 6.76 cm 高度差；下面的高度准备模式补上高度调整。

当时的物理记录、独立复现脚本、运动预览和检查日志已清理；下面保留源码回归测试入口。

```sh
uv run --no-sync pytest -q tests/tasks/manipulation/table_tennis/test_waist_preparation.py ../tests/test_rail_alignment.py
```

## 同步高度准备

在 UniLab 目录运行；`--height-prepare` 自动包含转腰、导轨和预测：

```sh
uv run --no-sync mjpython scripts/tools/preview_panthera_ball_launcher.py --height-prepare --fixed-serve --robot-collisions
uv run --no-sync mjpython scripts/tools/preview_panthera_ball_launcher.py --height-prepare --seed 7 --robot-collisions
```

link2、3、4 按同一个标量 s 协调调整，增量为 `(s/2, s, -s/2)`，因此
`q2-q3-q4` 保持 home 值，link5 固定、link6 保持原目标。加载模型时，真实 FK
采样局部分支 s∈[-0.1,0.8]，得到单调的高度与横向伸展表；接球时根据预测截面
球心高度插值，再按该姿态的拍心 Y 重新补偿导轨。左右 ±90° 的最终拍心 X
均保持 -1.53 m，预测截面不变。当前表支持约 0.981～1.181 m 的拍心高度；
超出该局部分支会取消准备有效性，不能将高度截断后宣布到位。

导轨、link1 和 link2--4 在同一个物理步开始控制。link1 保持原有 3 rad/s、
12 rad/s² 的轨迹参数；link2--4 的共同轨迹限速 1.5 rad/s、加速度 8 rad/s²。
四个活动关节的惯性前馈包含彼此及导轨的加速度耦合，仍通过原生驱动施力。
实际高度误差 ≤1 cm 加入既有到位条件，保留 30 ms 连续稳定及至少 150 ms 余量。
动态运行不写入关节实际位置或速度。离线播放图片来自记录的物理轨迹。

2026-10-05：3 条固定左右/中央球、7 条中线与 ±3 cm 边界球、64 条随机球及
16 条发球边界球，共 90 条隔离机器人碰撞的物理球路全部通过。最大实际截面
高度误差约 0.0143 mm，横向误差约 3.31 mm；最短实际提前到位 238.5 ms。
link1 实测峰值 3.00314 rad/s，link2--4 实测峰值 1.50177 rad/s，均记录了目标
限速的跟随误差，没有裁剪实际速度。对应 180 个最终高度姿态的真实 FK 均无接触。
共同曲线的全部 602 个离散样本在完整碰撞模型中也无接触；这不代表任意高度可达。

另有 26 条完整碰撞试验，包含固定、中线和发球边界，全部通过实际触拍时高度/
横向误差、提前准备余量、运动限制及接触类别检查，只出现球桌和球拍接触。
右侧固定球约 0.800 s 准备完成、1.065 s 被动触拍。被动触拍后球未经过预测截面，
故完整碰撞结果与隔离球路的通过时刻分开记录；尚未主动挥拍、验收合法回球或训练 RL。

本轮实测冷路径高度表加载约 8.77 ms；90 条球中单次准备目标更新最大约 0.076 ms，
不含球路预测、物理推进或显示。静态可达性检查发生在开发/加载阶段，接球时不迭代 IK。
这些运动与前馈参数属于仿真试验，未改变 Manager/RL 配置或验证实机性能。

当时的物理结果、轨迹、左右运动预览和检查日志已清理，源码测试仍保留。
本轮 21 项针对性测试通过；格式及类型检查通过（已有可选 Drake 导入警告）。
完整测试 1583 项通过，2 项因 Allegro 的 `base_link.stl` 资源缺失失败。

```sh
uv run --no-sync pytest -q tests/tasks/manipulation/table_tennis/test_height_preparation.py tests/tasks/manipulation/table_tennis/test_waist_preparation.py ../tests/test_rail_alignment.py
```
