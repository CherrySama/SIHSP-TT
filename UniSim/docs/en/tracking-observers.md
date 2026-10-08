# Substep observers and tracking control

[中文](../zh/tracking-observers.md)

`SimBackend.set_post_step_observer(fn)` registers a callback receiving the public backend after **every completed physical substep**, including the final substep of `step(ctrl, nsteps)`. `None` removes it. Reset/forward operations do not notify observers. Unsupported adapters reject nonempty registration with `NotImplementedError`; MuJoCo implements the contract.

Observers should only read public state and bound sensor views. They must not reset, step, apply forces, or replace the observer. MuJoCo rejects recursive stepping. Callback exceptions propagate after retaining completed physics and clearing staged/dynamic wrenches; notification never rolls back physics.

MuJoCo dispatches one batch substep per notification. Existing pre-step control is evaluated before each solve and interval wrenches remain active throughout the entire control call. This opt-in path adds Python/native dispatch overhead and preserves Euler, implicitfast, or RK4 rather than requiring the executor's Euler-only native callback sensor copyout. No final `mj_forward` is performed: acceleration-stage sensors, including MJCF `contact data='found'`, retain the contact solved during that substep. Generalized state is integrated; authored position/velocity sensors keep MuJoCo's native solve-time timing, while tracked-body getters synchronize final kinematics.

```python
contact = backend.bind_sensor_data(["racket_contact"])
backend.set_post_step_observer(lambda sim: event_tracker.observe(contact.read()))
backend.step(ctrl, nsteps=40)
backend.set_post_step_observer(None)
```

`evaluate_body_poses(joint_names, configurations, body_names)` is a cold query. `configurations` has shape `(nconfig, njoint)` and contains scalar hinge angles in radians or slide distances in meters. Unspecified joints use construction `qpos0`, not current online state. The result is world-frame positions with shape `(nconfig, nbody, 3)` and **wxyz** quaternions with shape `(nconfig, nbody, 4)`. The query does not change online state.

`bind_tracking_control(joint_names, actuator_names)` resolves names and caches numeric addresses/scratch state on the cold path. Its `TrackingControlView` contains immutable tuples `joint_names`, `actuator_names`, `lower`, `upper`, `kp`, and `kd`; bounds are actuator control limits and motor gains are zero. `compute(positions, velocities, accelerations)` accepts finite real arrays with shape `(num_envs, njoint)` and returns controls in **bound actuator order** with shape `(num_envs, nactuator)`. Joint-position references must respect authored joint limits. The caller places those columns in the full backend control array.

The MuJoCo implementation computes the unconstrained generalized compensation `tau = M(q) @ a_ref + bias(q, v) - passive(q, v)` from current online state. Unbound joint accelerations are zero. A direct motor starts from `tau`; a linear position servo receives `q_ref + kd/kp * v_ref + tau/kp`. The total desired servo force, including position/velocity feedback, is clipped to actuator force limits before conversion, then controls are clipped to control limits. Existing solver contacts and joint constraints still determine actual acceleration. Motors additionally compensate the coupled servo-feedback residual. After servo force/control clipping, the adapter uses the current unconstrained generalized force and inverse-mass response columns to solve `delta_motor_force = solve(invM[motor_dofs, motor_dofs], a_ref_motor - a_actual_motor)`, then clips motor forces to both joint/actuator force and control limits. This multi-motor solve retains servo PD feedback and matches a 0 N / 1 N acceleration probe when unconstrained and unsaturated; contacts, equality/friction constraints, and saturation can still prevent the requested acceleration. The motor reference acceleration must already include any desired feedback; this API does not invent a motor position controller.

These APIs support one scalar joint per actuator, direct joint transmission with unit gear, no activation dynamics, and either a unit-gain motor or a positive-gain affine position servo. Unsupported gains/transmissions and actuator gravity compensation fail closed. Current MuJoCo FK/tracking require a single unrandomized construction model: fixed variants and reset-time model-field writes are rejected, including when an existing bound view is used after a model write. Other adapters inherit an unsupported implementation.

```python
tracking = backend.bind_tracking_control(
    ["rail", "joint1"], ["rail_force", "joint1_position"]
)
controls = tracking.compute(reference_position, reference_velocity, reference_acceleration)
positions, quaternions = backend.evaluate_body_poses(
    ["rail", "joint1"], candidate_configurations, ["racket"]
)
```
