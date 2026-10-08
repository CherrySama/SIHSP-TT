---
orphan: true
---

# ADR-0011 物理子步跟踪控制与事件观测

- Status: Accepted
- Date: 2026-10-05
- Owners: Backend / Env / Table tennis task maintainers
- Supersedes: None
- Superseded by: None

## Context

回球训练需要在每个已完成的物理子步记录碰撞，包括控制周期内的短暂触拍和最后一个子步。
原准备控制还需要动力学跟踪补偿与冷路径球拍位姿查询。这些能力不能通过任务直接读取
MuJoCo 私有 model/data 实现，也不能把预览程序复制为第二套训练 runtime。

## Decision

沿用 ManagerBasedRlEnv、manager 和正式 RSL-RL runner。UniSim 提供公共
`set_post_step_observer`、`evaluate_body_poses` 与 `bind_tracking_control`。
UniLab env 的 `physics_backend` 返回公共 SimBackend；close 时解除子步观察者。
任务 action term 负责准备与挥拍轨迹、事件状态，配置、奖励和终止条件由 owner YAML 声明。

横拍变体 `PantheraHorizontalReturn` 复用这一 runtime：共享 stroke owner 负责确定性
准备、参考轨迹和逐子步 URDF 守卫；PPO 控制单调参考时钟与小幅关节修正。
零动作推进参考时钟，等待动作通过有限加速度降至零，时间重参数化同时处理参考速度和
加速度。策略初始均值为参考动作并保留随机探索，后续仍使用正式 RSL-RL PPO 更新。
此变体的观察、动作语义和 actor 声明写入独立 owner 配置及训练快照，即使同为六维动作，
也不能直接复用旧速度控制策略。实际合法回球和横拍动作验收分别统计，注册支持仅表示
Manager-Based 接口兼容，不代表已完成能力验收。

## Stable Contracts

- pre-step action 在每个物理子步提交控制；post-step observer 在已完成的每个子步读取状态及接触。
  控制周期奖励使用累计事件，首次触拍、过网和合法落台各计一次。
- 冷路径 FK 与跟踪绑定由 backend 解析引擎名字和索引；task 不持有引擎 model/data。
- 跟踪控制的输入是关节位置、速度、加速度，输出按绑定 actuator 顺序排列的控制量。
  伺服 PD、motor 动力学耦合、限力及 control 限幅由 adapter 实现；受约束或饱和时不保证
  实际加速度严格等于目标。
- 未实现这些能力的 adapter 明确拒绝非空绑定，不静默降级。
- PantheraReturn 当前仅注册 MuJoCo。局部开发使用旁边的 UniSim editable 源码，发布前须
  将此依赖换成包含这些公共能力的正式版本；不声称已发布新版 UniSim。

## Alternatives Considered

- 在任务或预览中访问 MuJoCo 私有状态：破坏 UniSim owner 边界，且无法复用正式 runtime。
- 只在控制周期末读取接触：会漏掉周期内已结束的碰撞。
- 为回球写专用 PPO 环境 wrapper：产生第二套 reset、step、奖励与训练接口。

## Consequences

事件观测属于物理采样，奖励和合法性属于 task。新增奖励无需改写 backend 或 runner。
接触参数和策略能力仍须独立验证，接口测试通过不构成回球成功率证明。

## Evidence In Repo

- `src/unilab/envs/manager_based_rl_env.py`
- `src/unilab/managers/_types.py`
- `src/unilab/tasks/manipulation/table_tennis/return_control.py`
- `src/unilab/tasks/manipulation/table_tennis/return_events.py`
- `src/unilab/conf/ppo/task/panthera_return/base.yaml`
- `src/unilab/tasks/manipulation/table_tennis/stroke_control.py`
- `src/unilab/tasks/manipulation/table_tennis/horizontal_return.py`
- `src/unilab/tasks/manipulation/table_tennis/horizontal_policy.py`
- `src/unilab/conf/ppo/task/panthera_horizontal/base.yaml`
- `tests/tasks/manipulation/table_tennis/test_horizontal_return.py`
- `tests/tasks/manipulation/table_tennis/test_return_runtime.py`
- `tests/tasks/manipulation/table_tennis/test_return_events.py`
- `../UniSim/src/unisim/backend/base.py` 和 MuJoCo adapter 的 tracking 实现与测试。

## Related Documents

- {doc}`ADR Index </adr/ADR-0000-index>`
- {doc}`UniSim Extraction Boundary </adr/ADR-0007-unisim-extraction-boundary>`
- {doc}`Task Owner And Config Compose Contract </adr/ADR-0003-task-owner-and-config-compose-contract>`
