from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from unisim import FakeBackend, TrackingControlView


def test_optional_tracking_and_observer_contract_fail_closed():
    backend = FakeBackend()
    backend.set_post_step_observer(None)
    with pytest.raises(NotImplementedError):
        backend.set_post_step_observer(lambda sim: None)
    with pytest.raises(TypeError):
        backend.set_post_step_observer(3)
    with pytest.raises(NotImplementedError):
        backend.evaluate_body_poses(["joint"], np.zeros((1, 1)), ["body"])
    with pytest.raises(NotImplementedError):
        backend.bind_tracking_control(["joint"], ["actuator"])


def test_tracking_view_validates_metadata_and_backend_output():
    view = TrackingControlView(
        joint_names=("joint",),
        actuator_names=("actuator",),
        num_envs=2,
        lower=(-1,),
        upper=(1,),
        kp=(0,),
        kd=(0,),
        _compute=lambda q, v, a: a,
    )
    with pytest.raises(FrozenInstanceError):
        view.kp = (100,)
    values = np.zeros((2, 1))
    np.testing.assert_array_equal(view.compute(values, values, values), values)
    with pytest.raises(ValueError, match="control limits"):
        view.compute(values, values, np.full((2, 1), 2))
    with pytest.raises(ValueError, match="lower limits"):
        TrackingControlView(
            joint_names=("joint",),
            actuator_names=("actuator",),
            num_envs=2,
            lower=(1,),
            upper=(-1,),
            kp=(0,),
            kd=(0,),
            _compute=lambda q, v, a: a,
        )
