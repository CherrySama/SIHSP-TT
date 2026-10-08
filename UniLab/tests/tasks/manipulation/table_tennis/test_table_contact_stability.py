"""Table rebound must survive impact phase changes, not selected time levels."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from scripts.tools.preview_panthera_ball_launcher import _load_preview_model


def test_table_contact_phase_and_impact_speed_stability():
    model = _load_preview_model(robot_collisions=False)
    data = mujoco.MjData(model)
    qpos = int(model.joint("ball_free").qposadr[0])
    dof = int(model.joint("ball_free").dofadr[0])
    ball, table = model.geom("ball_geom").id, model.geom("table_top").id
    for impact_speed in (2.4, 2.8, 3.2, 3.7):
        heights = []
        for phase in np.linspace(0, model.opt.timestep, 13):
            t = 0.05 + phase
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            data.qpos[qpos : qpos + 3] = [0.75, 0.0, 0.78 + impact_speed * t - 4.905 * t * t]
            data.qvel[dof : dof + 6] = [-2.8, 0.0, 9.81 * t - impact_speed, 0.0, 0.0, 0.0]
            contact = False
            peak = penetration = 0.0
            for _ in range(round(0.6 / model.opt.timestep)):
                mujoco.mj_step(model, data)
                contact |= any(
                    {c.geom1, c.geom2} == {ball, table} for c in data.contact[: data.ncon]
                )
                penetration = max(penetration, 0.78 - float(data.qpos[qpos + 2]))
                if contact and data.qvel[dof + 2] > 0:
                    peak = max(peak, float(data.qpos[qpos + 2] - 0.78))
                if peak > 0.1 and data.qvel[dof + 2] < 0:
                    break
            assert contact and penetration <= 0.006
            restitution = np.sqrt(2 * 9.81 * peak) / impact_speed
            assert 0.88 <= restitution <= 0.92
            heights.append(peak)
        assert np.ptp(heights) <= 0.015
