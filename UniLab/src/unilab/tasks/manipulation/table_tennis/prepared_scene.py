"""Cold-path training scene with the validated rigid link5 preparation profile."""

from __future__ import annotations

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH

ARM_CONTACT_NAMES = (
    "return_contact_arm_self",
    "return_contact_arm_table",
    "return_contact_arm_floor",
)


@lru_cache(maxsize=1)
def prepared_scene_path() -> str:
    """Materialize the prepared arm and actual solved-contact sensors once."""
    import mujoco

    source = Path(ASSETS_ROOT_PATH) / "robots" / "panthera"
    fingerprint = hashlib.sha256(
        (source / "scene.xml").read_bytes()
        + (source / "panthera.xml").read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / "unilab_panthera_return" / fingerprint
    target = directory / "scene.xml"
    if target.exists():
        return str(target)
    # MuJoCo's runtime exports MjSpec; the bundled static stub omits this export.
    spec = mujoco.MjSpec.from_file(str(source / "scene.xml"))  # pyright: ignore[reportAttributeAccessIssue]
    model = spec.compile()
    joint = model.joint("joint5")
    actuator = model.actuator("joint5_position")
    for key in spec.keys:
        key.qpos = np.delete(np.asarray(key.qpos), int(joint.qposadr[0]))
        if len(key.qvel):
            key.qvel = np.delete(np.asarray(key.qvel), int(joint.dofadr[0]))
        key.ctrl = np.delete(np.asarray(key.ctrl), actuator.id)
    spec.delete(spec.equality("joint5_lock"))
    spec.delete(spec.actuator("joint5_position"))
    spec.delete(spec.joint("joint5"))
    spec.equality("rail_lock").active = False
    spec.compile()
    root = ET.fromstring(spec.to_xml())
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(source / "assets"))
    sensors = ET.SubElement(root, "sensor")
    for label, attributes in {
        "paddle": {"geom2": "paddle_contact"},
        "table": {"geom2": "table_top"},
        "net": {"subtree2": "net"},
        "robot": {"subtree2": "rail_carriage"},
        "floor": {"geom2": "floor"},
    }.items():
        ET.SubElement(
            sensors,
            "contact",
            {
                "name": f"return_contact_{label}",
                "geom1": "ball_geom",
                "data": "found",
                "num": "1",
                **attributes,
            },
        )
    # These selectors observe robot contacts independently of the five ball
    # contact sensors. Self contacts include the carriage and attached paddle;
    # table contacts include the full table subtree, including its legs.
    for name, attributes in zip(
        ARM_CONTACT_NAMES,
        (
            {"subtree2": "rail_carriage"},
            {"subtree2": "table"},
            {"geom2": "floor"},
        ),
        strict=True,
    ):
        ET.SubElement(
            sensors,
            "contact",
            {
                "name": name,
                "subtree1": "rail_carriage",
                "data": "found",
                "num": "1",
                **attributes,
            },
        )
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=directory, delete=False) as output:
        output.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
        temporary = output.name
    os.replace(temporary, target)
    return str(target)
