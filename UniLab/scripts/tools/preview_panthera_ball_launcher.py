"""Preview random incoming balls with the arm displayed at home.

On macOS: uv run mjpython scripts/tools/preview_panthera_ball_launcher.py
Default: ball-trajectory preview with robot collisions disabled locally.
Use --robot-collisions to observe the full scene, including passive home hits.
Use --predict with the table-tennis dependency group for prediction overlays.
Use --rail-align for physical lateral rail preparation with the arm at home.
Use --waist-prepare for simultaneous rail alignment and 90-degree waist yaw.
Use --height-prepare to also coordinate joint2--4 to the predicted ball height.
This viewer uses the task sampler and scene; it is not a hitting controller.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.tasks.manipulation.table_tennis.ball_launcher import (
    ServeLauncherCfg,
    sample_ball_launch,
)

if TYPE_CHECKING:
    from ball_prediction import BallPrediction


def _load_preview_model(*, robot_collisions: bool, rail_align: bool = False) -> mujoco.MjModel:
    """Load a separate preview model without modifying the task scene."""
    path = ASSETS_ROOT_PATH / "robots/panthera/scene.xml"
    if rail_align:
        from unilab.visualization.panthera_rail_preview import load_rail_preview_model

        model = load_rail_preview_model(path)
    else:
        model = mujoco.MjModel.from_xml_path(str(path))
    if not robot_collisions:
        # The current scene has no explicit ball/robot contact pairs. Disable
        # robot geometry collisions only in this separately loaded viewer model.
        robot_bodies = {model.body("rail_carriage").id}
        for body in range(model.nbody):
            if model.body_parentid[body] in robot_bodies:
                robot_bodies.add(body)
        robot_geoms = np.isin(model.geom_bodyid, list(robot_bodies))
        model.geom_contype[robot_geoms] = 0
        model.geom_conaffinity[robot_geoms] = 0
    return model


def _draw_prediction(
    scene: mujoco.MjvScene,
    prediction: BallPrediction | None,
    observed: list[tuple[float, float, float]],
) -> None:
    """Draw measured history and forecasts as non-physical viewer geometry."""
    scene.ngeom = 0
    size = np.zeros(3)
    identity = np.eye(3).ravel()

    def line(a, b, color) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size, a, identity, color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.5, a, b)
        scene.ngeom += 1

    for a, b in zip(observed[:-1], observed[1:]):
        line(np.asarray(a), np.asarray(b), np.asarray([0.1, 0.8, 1.0, 1.0], dtype=np.float32))
    if prediction is None or not prediction.valid:
        return
    positions = [np.asarray(state.position) for state in prediction.trajectory]
    for a, b in zip(positions[:-1], positions[1:]):
        line(a, b, np.asarray([1.0, 0.5, 0.1, 1.0], dtype=np.float32))
    for point, color in (
        (prediction.opponent_bounce, [0.6, 0.3, 1.0, 1.0]),
        (prediction.first_bounce, [1.0, 0.1, 0.9, 1.0]),
        (prediction.candidate, [0.1, 1.0, 0.2, 1.0]),
    ):
        if point is not None and scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.014),
                np.asarray(point.position),
                identity,
                np.asarray(color, dtype=np.float32),
            )
            scene.ngeom += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fixed-serve", action="store_true", help="Repeat one baseline serve")
    parser.add_argument(
        "--robot-collisions", action="store_true", help="Keep robot collisions in the preview"
    )
    parser.add_argument(
        "--predict", action="store_true", help="Display ball trajectory predictions"
    )
    parser.add_argument(
        "--rail-align", action="store_true", help="Enable lateral-only rail preparation"
    )
    parser.add_argument(
        "--waist-prepare", action="store_true", help="Prepare a facing paddle with rail and joint1"
    )
    parser.add_argument(
        "--waist-speed", type=float, default=3.0, help="Trial waist trajectory speed in rad/s"
    )
    parser.add_argument(
        "--height-prepare", action="store_true", help="Add coordinated joint2--4 height preparation"
    )
    args = parser.parse_args()
    args.waist_prepare = args.waist_prepare or args.height_prepare
    args.rail_align = args.rail_align or args.waist_prepare
    args.predict = args.predict or args.rail_align
    predictor = None
    if args.predict:
        try:
            from ball_state import BallState

            from unilab.tasks.manipulation.table_tennis.prediction_adapter import (
                PantheraBallPredictor,
            )
        except ModuleNotFoundError as exc:
            if exc.name not in ("ball_state", "ball_prediction", "rail_alignment"):
                raise
            parser.error("预测预览需要使用 uv run --group table-tennis --extra mujoco")
        predictor = PantheraBallPredictor()
    model = _load_preview_model(robot_collisions=args.robot_collisions, rail_align=args.rail_align)
    if args.robot_collisions:
        print("完整场景预览：保留机器人碰撞；home 姿态可能被动碰球并改变球路。")
    else:
        print("球路预览：仅此预览关闭机器人碰撞；主环境保留碰撞。")
    if args.predict:
        print("青色：已观测轨迹；橙色：未来预测；淡紫：对方弹；紫色：己方弹；绿色：候选接球点。")
        print("候选点是预测验证截面，尚未证明机械臂可达；预测不包含机器人碰撞。")
    data = mujoco.MjData(model)
    rail = None
    if args.rail_align:
        from unilab.visualization.panthera_rail_preview import (
            PantheraRailPreview,
            draw_rail_alignment,
        )

        waist_cfg = None
        if args.waist_prepare:
            from rail_alignment import WaistPreparationCfg

            try:
                waist_cfg = WaistPreparationCfg(max_velocity=args.waist_speed)
            except ValueError as exc:
                parser.error(str(exc))
        rail = PantheraRailPreview(
            model, data, waist_preparation=waist_cfg, height_preparation=args.height_prepare
        )
        if waist_cfg is not None:
            print(
                f"同步转腰准备：link1 按来球侧转到 ±90°，目标轨迹速度 ≤{args.waist_speed:g} rad/s；"
                "导轨同时启动，按最终准备拍心补偿。"
            )
            print("浅蓝：实际拍心；粉色：最终准备姿态拍心；黄色：横向目标；绿色：预计球位置。")
            if rail.height is not None:
                print(
                    "同步高度准备：link2--4 保持合成角，跟随预测截面球心高度；"
                    "轨迹速度 ≤1.5 rad/s，加速度 ≤8 rad/s²；link5 固定，link6 保持 home。"
                )
                print("中心来球采用右侧准备。浅蓝为实测拍心；尚未主动挥拍。")
            else:
                print("其他关节保持 home。中心来球采用右侧准备；尚未调整高度或主动挥拍。")
        else:
            print("导轨横向预定位：手臂保持 home 目标，link5 固定；浅蓝为拍心，黄色为横向目标。")
        limits = rail.controller.cfg
        print(
            f"导轨暂定上限：{limits.max_velocity:g} m/s、{limits.max_acceleration:g} m/s²、"
            f"{limits.max_force:g} N；只验证左右对准，不代表能够回球。"
        )
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    qpos = model.jnt_qposadr[joint]
    dof = model.jnt_dofadr[joint]
    rng = np.random.default_rng(args.seed)
    cfg = ServeLauncherCfg()
    if args.fixed_serve:
        cfg = replace(
            cfg,
            lateral_range=(-0.35, -0.35) if args.waist_prepare else (0.0, 0.0),
            speed_range=(2.9, 2.9),
            receiver_bounce_range=(-0.94, -0.94),
            receiver_peak_range=(0.32, 0.32),
        )
    print(
        f"端线发球：X={cfg.launch_x:.2f} m，发射水平速度 "
        f"{cfg.speed_range[0]:.2f}–{cfg.speed_range[1]:.2f} m/s；"
        "先对方半台反弹，再过网、己方半台反弹；无初始旋转。"
    )
    display_dt = 0.01
    display_substeps = round(display_dt / model.opt.timestep)
    observed: deque[tuple[float, float, float]] = deque(maxlen=200)
    prediction = None
    last_prediction_time = -1.0
    reported_outcome = False

    def launch() -> None:
        nonlocal last_prediction_time, prediction, reported_outcome
        mujoco.mj_resetDataKeyframe(model, data, home)
        ball = sample_ball_launch(rng, cfg)
        data.qpos[qpos : qpos + 3] = ball.position
        data.qvel[dof : dof + 3] = ball.linear_velocity
        data.qvel[dof + 3 : dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        observed.clear()
        prediction = None
        last_prediction_time = -1.0
        reported_outcome = False
        if rail is not None:
            rail.reset()
        if predictor is not None:
            predictor.reset()

    launch()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            if data.time >= 2.0:
                launch()
            if predictor is not None and data.time - last_prediction_time >= 0.01 - 1e-9:
                if rail is not None:
                    mujoco.mj_forward(model, data)
                    predictor.cfg = replace(predictor.cfg, candidate_x=rail.candidate_x)
                position = data.qpos[qpos : qpos + 3]
                velocity = data.qvel[dof : dof + 3]
                state = BallState(
                    timestamp=float(data.time),
                    position=(float(position[0]), float(position[1]), float(position[2])),
                    velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
                )
                prediction = predictor.update(state)
                if rail is not None:
                    rail.observe(prediction)
                    if rail.first_ready_time is not None and not reported_outcome:
                        target = rail.controller.target
                        assert target is not None
                        print(
                            f"{'转腰准备' if args.waist_prepare else '横向预定位'}完成："
                            f"拍心横向偏差 {abs(rail.paddle_position[1] - target.ball_y) * 1000:.1f} mm，"
                            f"导轨速度 {abs(rail.velocity):.3f} m/s，"
                            f"距预计到达 {target.arrival_time - data.time:.3f} s。"
                        )
                        if rail.height is not None:
                            print(
                                f"拍心高度偏差 "
                                f"{abs(rail.paddle_position[2] - rail.height.target_height) * 1000:.2f} mm。"
                            )
                        reported_outcome = True
                    target = rail.controller.target
                    if (
                        target is not None
                        and data.time >= target.arrival_time
                        and not reported_outcome
                    ):
                        print("本条来球未在截止前满足准备标准。")
                        if rail.waist is not None:
                            print(
                                f"link1 实际 {math.degrees(data.qpos[rail.arm_qpos[0]]):.1f}°，"
                                f"速度 {data.qvel[rail.arm_dof[0]]:.3f} rad/s。"
                            )
                        reported_outcome = True
                observed.append(state.position)
                last_prediction_time = float(data.time)
                if len(observed) == 1 and prediction.candidate is not None:
                    point = prediction.candidate
                    print(
                        f"预计候选点：({point.position[0]:.3f}, {point.position[1]:.3f}, "
                        f"{point.position[2]:.3f}) m，发球后 {point.timestamp:.3f} s 到达。"
                    )
                with viewer.lock():
                    _draw_prediction(viewer.user_scn, prediction, list(observed))
                    if rail is not None:
                        draw_rail_alignment(viewer.user_scn, rail)
            for _ in range(display_substeps):
                if rail is not None:
                    rail.apply_force()
                mujoco.mj_step(model, data)
            viewer.sync()
            remaining = display_dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
