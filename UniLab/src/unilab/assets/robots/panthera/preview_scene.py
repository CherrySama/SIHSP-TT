"""Open the Panthera table-tennis scene in the passive MuJoCo viewer."""

# pyright: reportAttributeAccessIssue=false

import time
from pathlib import Path

import mujoco
import mujoco.viewer

scene_path = Path(__file__).with_name("scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)
