# 子步观察器与轨迹跟踪控制

[English](../en/tracking-observers.md)

`SimBackend.set_post_step_observer(fn)` 注册回调，在**每个已经完成的物理子步** 后接收公共 backend，包括 `step(ctrl, nsteps)` 的最后一个子步。传入 `None` 移除回调。Reset/forward 不会调用观察器。不支持的适配器注册非空观察器时抛出 `NotImplementedError`；MuJoCo 实现了这一契约。

观察器只能读取公共状态和绑定传感器视图，不应 reset、step、施加力或替换观察器。 MuJoCo 拒绝递归 step。回调异常会向上传播，保留已完成的物理步并清空待施加及动态 外力；观察器通知不会回滚物理过程。

MuJoCo 每次通知前执行一个批量物理子步。已有 pre-step 控制在每次求解前计算， interval 外力在整个控制调用期间持续有效。这条按需启用的路径增加 Python/native 调用开销，保留 Euler、implicitfast 或 RK4，不依赖 executor 仅支持 Euler 的原生 callback 传感器 copyout。不执行末尾 `mj_forward`：加速度阶段传感器，包括 MJCF 的 `contact data='found'`，保留该子步实际求解的接触。广义状态已经积分；用户定义的位置 和速度传感器保持 MuJoCo 原生求解时刻语义，tracked-body getter 同步最终运动学。

```python
contact = backend.bind_sensor_data(["racket_contact"])
backend.set_post_step_observer(lambda sim: event_tracker.observe(contact.read()))
backend.step(ctrl, nsteps=40)
backend.set_post_step_observer(None)
```

`evaluate_body_poses(joint_names, configurations, body_names)` 是冷路径查询。 `configurations` 的形状为 `(nconfig, njoint)`，包含标量 hinge 角度（弧度）或 slide 位移（米）。未指定关节使用构造时的 `qpos0`，不使用当前在线状态。返回世界坐标系位置， 形状为 `(nconfig, nbody, 3)`，以及 **wxyz** 四元数，形状为 `(nconfig, nbody, 4)`。查询不会改变在线状态。

`bind_tracking_control(joint_names, actuator_names)` 在冷路径解析名称并缓存数值地址 和 scratch 状态。返回的 `TrackingControlView` 包含不可变 tuple：`joint_names`、 `actuator_names`、`lower`、`upper`、`kp` 和 `kd`；边界是执行器控制范围，motor 的 增益为零。`compute(positions, velocities, accelerations)` 接收形状为 `(num_envs, njoint)` 的有限实数数组，返回按**绑定执行器顺序**排列、形状为 `(num_envs, nactuator)` 的控制。关节位置参考必须符合模型关节范围。调用者将这些列 放入完整的 backend 控制数组。

MuJoCo 实现基于当前在线状态计算不包含约束力的广义补偿： `tau = M(q) @ a_ref + bias(q, v) - passive(q, v)`。未绑定关节的参考加速度设为零。 直接 motor 从 `tau` 开始计算；线性位置伺服接收 `q_ref + kd/kp * v_ref + tau/kp`。 包含位置及速度反馈的总伺服目标力先按执行器力范围裁剪，再转换并按控制范围裁剪。 实际加速度仍由求解器的接触和关节约束决定。Motor 还会补偿伺服反馈引起的耦合加速度残差。伺服力及控制裁剪后，适配器使用当前无约束广义力和逆质量响应列求解 `delta_motor_force = solve(invM[motor_dofs, motor_dofs], a_ref_motor - a_actual_motor)`，再按关节/执行器力范围和控制范围裁剪 motor 力。这个多 motor 求解保留伺服 PD 反馈，在无约束且未饱和时等价于 0 N / 1 N 加速度探测；接触、等式/摩擦约束和饱和仍可能阻止期望加速度。 Motor 的参考加速度应已经包含所需反馈； 本接口不会自动增加 motor 位置控制器。

这些接口支持每个执行器对应一个标量关节、单位 gear 的直接关节传动、无 activation 动态，以及单位增益 motor 或正增益 affine 位置伺服。不支持的增益、传动和执行器重力 补偿会明确报错。当前 MuJoCo FK/tracking 要求单一且未经随机化的构造模型：固定 variants 和 reset 时的模型字段写入均被拒绝，包括在模型写入后使用已有绑定视图的情况。 其他适配器继承不支持实现。

```python
tracking = backend.bind_tracking_control(
    ["rail", "joint1"], ["rail_force", "joint1_position"]
)
controls = tracking.compute(reference_position, reference_velocity, reference_acceleration)
positions, quaternions = backend.evaluate_body_poses(
    ["rail", "joint1"], candidate_configurations, ["racket"]
)
```
