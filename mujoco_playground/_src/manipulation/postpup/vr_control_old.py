from typing import Callable, Optional
import numpy as np
from etils import epath
from mujoco_playground._src.manipulation.postpup import (
    base,
    model_utils,
    constants,
    place_letter,
)
import mujoco.viewer as viewer
import mujoco


def load_callback(xml_path: epath.Path, model=None, data=None):

    model_utils.create_all_model_variations()

    model = mujoco.MjModel.from_xml_path(
        xml_path.as_posix(),
        assets=model_utils.get_assets(),
    )
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.004
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    vr_controller = VRController(model, data)
    mujoco.set_mjcb_control(vr_controller.control_fn)

    return model, data


def rotation_matrix_to_angle_axis(R_rel):
    """
    Compute the angular velocity (in rad/s) that rotates R_rel over a time interval of 1s

    Parameters:
        R_rel (np.ndarray): Relative rotation matrix (3, 3).

    Returns:
        np.ndarray: Angular velocity vector (3,).
    """
    # Compute the relative rotation matrix: R_rel = R2 * R1^T

    # Calculate the rotation angle theta from the trace of R_rel.
    trace = np.trace(R_rel)
    theta = np.arccos((trace - 1) / 2.0)

    # To avoid division by zero in case theta is very small
    sin_theta = np.sin(theta)
    eps = 1e-6
    if np.abs(sin_theta) < eps:
        sin_theta = eps

    # Extract the skew-symmetric part of R_rel to get the rotation axis components.
    a_x = R_rel[2, 1] - R_rel[1, 2]
    a_y = R_rel[0, 2] - R_rel[2, 0]
    a_z = R_rel[1, 0] - R_rel[0, 1]

    # The factor to obtain the axis from the skew-symmetric part.
    factor = theta / (2.0 * sin_theta)
    rotation_vector = factor * np.array([a_x, a_y, a_z])

    return rotation_vector


class VRController:
    def __init__(self, model, data):

        self.jacp = np.zeros((3, model.nv))  # translation jacobian
        self.jacr = np.zeros((3, model.nv))

        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)
        ]
        print(f"{body_names=}")
        self.gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"
        )
        assert self.gripper_id != -1

        site_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            for i in range(model.nsite)
        ]
        print(f"{site_names=}")
        self.gripper_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site_x_forward"
        )
        assert self.gripper_site_id != -1

    def control_fn(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # print(data.ctrl)
        # data.ctrl[0] = 1.2
        import ipdb

        ipdb.set_trace()

        site_pos = data.site_xpos[self.gripper_site_id]
        site_orientation = data.site_xmat[self.gripper_site_id].reshape(
            3, 3
        )  # R_world_from_gripper_site
        print(f"{site_pos=}")
        print(f"{site_orientation=}")

        mujoco.mj_jacSite(model, data, self.jacp, self.jacr, self.gripper_site_id)
        print(f"{self.jacp.shape=} {self.jacp=}")  # (3, 15)
        print(f"{self.jacr.shape=} {self.jacr=}")  # (3, 15)

        goal = np.array([0.16866421, 0.0958612, 0.37914733])
        goal_orientation = np.eye(3)  # R_world_from_goal
        error = goal - site_pos
        desired_velocity = error
        # desired_velocity = 0.0 * error
        print(f"{desired_velocity=}")

        R_goal_from_gripper_site = goal_orientation.T @ site_orientation
        w_error = rotation_matrix_to_angle_axis(R_goal_from_gripper_site)
        desired_end_effector_angular_velocity = -0.01 * w_error
        # desired_end_effector_angular_velocity = 0.0 * w_error
        print(f"{desired_end_effector_angular_velocity=}")

        stacked_jac = np.vstack((self.jacp[:, :6], self.jacr[:, :6]))
        v_w = np.concatenate((desired_velocity, desired_end_effector_angular_velocity))
        print(f"{v_w.shape=} {v_w=}")
        print(f"{stacked_jac.shape=} {stacked_jac=}")
        # joint_vel = np.linalg.pinv(stacked_jac) @ v_w # breaks simulation
        joint_vel = stacked_jac.T @ v_w
        print(f"{joint_vel.shape=} {joint_vel=}")

        print(f"{data.qpos.shape=} {data.qpos=}")
        data.ctrl[:6] = joint_vel * 80.0
        print(f"{data.qvel.shape=} {data.qvel=}")
        # data.qvel[:6] = desired_joint_velocity

        # works
        # data.ctrl[:6] = (self.jacp[:, :6]).T @ error * 40.0


if __name__ == "__main__":
    viewer.launch(loader=lambda: load_callback(constants.PLACE_LETTER_SCENE_XML))
