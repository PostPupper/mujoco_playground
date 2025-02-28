# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PostPup base class."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
import jinja2
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.postpup import piper_model_utils


_ARM_JOINTS = [
    "joint_base_yaw",
    "joint_base_pitch",
    "joint_elbow",
    "joint_wrist_rotate",
    "joint_wrist_bend",
    "joint_wrist_rotate2",
]
_FINGER_JOINTS = ["joint_gripper_left", "joint_gripper_right"]
_ALL_JOINTS = _ARM_JOINTS + _FINGER_JOINTS

MONOREPO_ROOT = epath.Path(__file__).parents[6]
AGILEX_PIPER = MONOREPO_ROOT / "agilex-piper/"
AGILEX_PIPER_MESHES = AGILEX_PIPER / "meshes"
AGILEX_PIPER_URDF = AGILEX_PIPER / "urdf"


def get_assets() -> Dict[str, bytes]:
    assets = {}
    print(f"agilex_piper: {AGILEX_PIPER}")
    print(f"agilex_piper_meshes: {AGILEX_PIPER_MESHES}")
    print(f"agilex_piper_urdf: {AGILEX_PIPER_URDF}")
    mjx_env.update_assets(assets, AGILEX_PIPER_URDF, "*.xml")
    mjx_env.update_assets(assets, AGILEX_PIPER_MESHES)
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,  # TODO
    )


class PostPupBase(mjx_env.MjxEnv):
    """Base environment for PostPup."""

    def __init__(
        self,
        xml_template_path: epath.Path,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        # render template using MJX compatiblity
        self._xml_path = AGILEX_PIPER_URDF / "piper_rendered_mjx.xml"
        xml = piper_model_utils.create_model_xml(
            template_xml=xml_template_path, output_xml=self._xml_path, mode="mjx"
        )
        _ = piper_model_utils.create_model_xml(
            template_xml=xml_template_path,
            output_xml=AGILEX_PIPER_URDF / "piper_rendered_normal.xml",
            mode="normal",
        )

        mj_model = mujoco.MjModel.from_xml_string(xml, assets=get_assets())
        mj_model.opt.timestep = self.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)
        self._action_scale = config.action_scale

        # Populate
        self._robot_arm_qposadr = np.array(
            [
                self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
                for j in _ARM_JOINTS
            ]
        )
        self._robot_qposadr = np.array(
            [
                self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
                for j in _ALL_JOINTS
            ]
        )
        # self._gripper_site = self._mj_model.site("gripper").id # TODO
        self._left_finger_geom = self._mj_model.geom("collision_gripper_left").id
        self._right_finger_geom = self._mj_model.geom("collision_gripper_right").id
        self._hand_geom = self._mj_model.geom("collision_gripper_base").id

        print(f"robot_arm_qposadr: {self._robot_arm_qposadr}")
        print(f"robot_qposadr: {self._robot_qposadr}")
        print(f"left_finger_geom: {self._left_finger_geom}")
        print(f"right_finger_geom: {self._right_finger_geom}")
        print(f"hand_geom: {self._hand_geom}")

        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T
        print(f"lowers: {self._lowers}")
        print(f"uppers: {self._uppers}")

    def _post_init(self, obj_name: str, keyframe: str):
        self._floor_geom = self._mj_model.geom("floor").id

        self._obj_body = self._mj_model.body(obj_name).id
        self._obj_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body(obj_name).jntadr[0]
        ]
        self._init_q = self._mj_model.keyframe(keyframe).qpos
        self._init_obj_pos = jp.array(
            self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
            dtype=jp.float32,
        )
        self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl

    @property
    def xml_path(self) -> str:
        raise self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


if __name__ == "__main__":

    class TestConcrete(PostPupBase):
        def step(self, *args, **kwargs):
            pass

        def reset(self, *args, **kwargs):
            pass

    TestConcrete(AGILEX_PIPER_URDF / "piper_template.xml", default_config())
