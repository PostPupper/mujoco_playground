from mujoco_playground._src.manipulation.postpup import (
    base,
    model_utils,
    constants,
)
import mujoco.viewer as viewer

from ml_collections import config_dict
from typing import Any, Dict, Optional, Union


def default_config() -> config_dict.ConfigDict:
    """Returns the default config for bring_to_target tasks."""
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,  # TODO
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Gripper goes to the box.
                gripper_box=4.0,
                # Box goes to the target mocap.
                box_target=8.0,
                # Do not collide the gripper with the floor.
                no_floor_collision=0.25,
                # Arm stays close to target pose.
                robot_target_qpos=0.3,
            )
        ),
    )
    return config


class PlaceLetter(base.PostPupBase):
    def __int__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        xml_path = constants.PLACE_LETTER_SCENE_XML
        super().__init__(xml_path, config)


def control_fn(model, data):
    print(data.site("gripper_site_x_forward").xpos)
    print(data.site("gripper_site_x_forward").xmat)


if __name__ == "__main__":
    model_utils.visualize_model(constants.PLACE_LETTER_SCENE_XML, control_fn=control_fn)
