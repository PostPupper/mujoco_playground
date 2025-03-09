from typing import Optional
import mujoco
from mujoco_playground._src.manipulation.postpup.hot_ik_training import RegressionMLP
from mujoco_playground._src.manipulation.postpup import (
    model_utils,
    constants,
    data_generation,
)
from etils import epath
import numpy as np
import torch


class PiperLearnedIK:
    def __init__(
        self,
        model_checkpoint: epath.Path = epath.Path(
            "./checkpoints/regression-mlp-epoch=999-train_loss=0.2214.ckpt"
        ),
        n_joints: int = 8,
        device="cpu",
    ):
        self.model = RegressionMLP.load_from_checkpoint(
            model_checkpoint.as_posix(), map_location=device
        )
        self.model.eval()
        self.device = device
        self.n_joints = n_joints

    def pad(self, ee_pos_quat: np.ndarray):
        return np.concatenate([ee_pos_quat, np.zeros(self.n_joints - ee_pos_quat.size)])

    def __call__(self, ee_pos_quat: np.ndarray, **kwds):
        input = torch.tensor(ee_pos_quat, dtype=torch.float32).to(self.device)
        return self.pad(self.model(input).detach().numpy())


class Piper3DOFIK:
    def __init__(self, model_xml: epath.Path, site_name: str = "wrist_base_site"):
        self.site_name = site_name
        self.fk = PiperFK(model_xml=model_xml, site_name=self.site_name)
        self.mj_model = self.fk.mj_model
        self.mj_data = self.fk.mj_data

        self.jacp = np.zeros((3, self.mj_model.nv))
        self.jacr = np.zeros((3, self.mj_model.nv))
        self.dofs = 3

        self.site_id = self.mj_model.site(self.site_name).id

    def normalize_angles(self, joint_angles: np.ndarray):
        # normalize angles between -pi and pi
        return np.arctan2(np.sin(joint_angles), np.cos(joint_angles))

    def __call__(
        self,
        ee_pos_quat: np.ndarray,
        tolerance=1e-4,
        max_iters=100,
        max_seeds=10,
        method: str = "newton",
        verbose: bool = False,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        # non-singularity starting position
        if initial_guess is None:
            initial_angles = np.array([0.0, 0.5, -0.9])
        else:
            initial_angles = initial_guess.copy()

        for attempts in range(max_seeds):
            joint_angles = initial_angles.copy()
            for i in range(max_iters):
                self.mj_data.qpos[: self.dofs] = joint_angles
                mujoco.mj_kinematics(self.mj_model, self.mj_data)
                mujoco.mj_comPos(self.mj_model, self.mj_data)
                mujoco.mj_jacSite(
                    self.mj_model, self.mj_data, self.jacp, self.jacr, self.site_id
                )

                error = self.mj_data.site(self.site_name).xpos - ee_pos_quat[:3]

                if verbose:
                    print(
                        f"{i=} norm: {np.linalg.norm(error)} error: {error[0]:.3f} {error[1]:.3f} {error[2]:.3f} "
                        + f"q[:3]: {joint_angles[0]:0.2f} {joint_angles[1]:0.2f} {joint_angles[2]:0.2f}"
                    )

                # Check if converged
                if np.linalg.norm(error) < tolerance:
                    joint_angles = self.normalize_angles(joint_angles)
                    if (joint_angles < self.fk.lowers[: self.dofs] - 1e-3).any() or (
                        joint_angles > self.fk.uppers[: self.dofs] + 1e-3
                    ).any():
                        print(
                            f"Joint angles out of bounds!!! q={joint_angles=} lowers={self.fk.lowers[:self.dofs]=} uppers={self.fk.uppers[:self.dofs]=}"
                        )
                        return None
                    return joint_angles

                # Update joint angles
                J = self.jacp[:, :3]

                # gradient descent
                if method == "gradient_descent":
                    joint_angles -= 20.0 * J.T @ error

                elif method == "newton":
                    joint_angles -= 1.0 * np.linalg.pinv(J) @ error

                else:
                    raise ValueError(f"Unknown method: {method}")

            initial_angles = np.random.uniform(self.fk.lowers, self.fk.uppers)[
                : self.dofs
            ]

        print(
            f"IK FAILED: Did not converge after {max_iters} iterations with {max_seeds} seeds"
        )
        return None


def small_angle_matrix_log(R_rel):
    a_x = R_rel[2, 1] - R_rel[1, 2]
    a_y = R_rel[0, 2] - R_rel[2, 0]
    a_z = R_rel[1, 0] - R_rel[0, 1]
    return 0.5 * np.array([a_x, a_y, a_z])


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
    if trace > 3 - 1e-6:
        theta = 0
    elif trace < -1 + 1e-6:
        theta = np.pi
    else:
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


class Piper6DOFIK:
    def __init__(
        self, model_xml: epath.Path, site_name: str = "gripper_site_x_forward"
    ):
        self.site_name = site_name
        print("loading model for ik")
        self.mj_model, self.mj_data = model_utils.load_callback(xml_path=model_xml)

        self.lowers = self.mj_model.jnt_range[:, 0]
        self.uppers = self.mj_model.jnt_range[:, 1]

        self.dofs = 6

        self.site_id = self.mj_model.site(self.site_name).id

    def __call__(
        self,
        ee_pos_quat: np.ndarray,
        tolerance=1e-4,
        max_iters=100,
        initial_q: np.ndarray = np.array([0.0, 0.5, -0.9, -0.2, 0.4, 0, 0, 0]),
    ):
        # print(self.mj_model, self.mj_data)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # non-singularity starting position
        joint_angles = initial_q.copy()

        for i in range(5):
            for i in range(max_iters):
                self.mj_data.qpos[:] = joint_angles
                mujoco.mj_kinematics(self.mj_model, self.mj_data)
                mujoco.mj_comPos(self.mj_model, self.mj_data)
                jacp = np.zeros((3, self.mj_model.nv))
                jacr = np.zeros((3, self.mj_model.nv))
                mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self.site_id)

                pos_error = (
                    self.mj_data.site(self.site_name).xpos.copy() - ee_pos_quat[:3]
                )
                R_world_from_site = (
                    self.mj_data.site(self.site_name).xmat.copy().reshape((3, 3))
                )

                R_world_from_goal = np.zeros(9)
                mujoco.mju_quat2Mat(R_world_from_goal, ee_pos_quat[3:])
                R_world_from_goal = R_world_from_goal.reshape((3, 3))

                R_site_from_goal = R_world_from_site.T @ R_world_from_goal
                w_site_from_goal = rotation_matrix_to_angle_axis(R_site_from_goal)

                J = jacr[:, :6].copy()
                # joint_angles[: self.dofs] += 0.8 * np.linalg.pinv(J) @ w_site_from_goal
                joint_angles[: self.dofs] += 0.5 * J.T @ w_site_from_goal

                # print(
                #     f"{i=} norm: {np.linalg.norm(w_site_from_goal)} w_error: {w_site_from_goal[0]:.3f} {w_site_from_goal[1]:.3f} {w_site_from_goal[2]:.3f}"
                #     # + f"q[:3]: {joint_angles[0]:0.2f} {joint_angles[1]:0.2f} {joint_angles[2]:0.2f}"
                # )
                joint_angles = np.clip(joint_angles, self.lowers, self.uppers)
                if np.linalg.norm(w_site_from_goal) < tolerance:
                    return joint_angles[: self.dofs]

            print(
                "IK WARNING: Did not converge. Trying again with a new random starting point"
            )
            joint_angles = np.random.uniform(self.lowers, self.uppers)

        print("IK FAILED: Did not converge after several attempts")
        return joint_angles[: self.dofs]


class PiperFK:
    def __init__(
        self,
        model_xml: epath.Path = constants.PIPER_RENDERED_NORMAL_XML,
        site_name: str = "gripper_site_x_forward",
    ):
        self.mj_model, self.mj_data = model_utils.load_callback(xml_path=model_xml)

        self.lowers = self.mj_model.jnt_range[:, 0]
        self.uppers = self.mj_model.jnt_range[:, 1]

        self.site_name = site_name

    def zero_pad(self, joint_angles: np.ndarray):
        return np.concatenate(
            [joint_angles, np.zeros(self.mj_model.nq - joint_angles.size)]
        )

    def __call__(self, joint_angles: np.ndarray):
        joint_angles = self.zero_pad(joint_angles)
        if (joint_angles < self.lowers - 1e-3).any() or (
            joint_angles > self.uppers + 1e-3
        ).any():
            print(
                f"PiperFK: Joint angles out of bounds!!! q={joint_angles=} lowers={self.lowers=} uppers={self.uppers=}"
            )
            # joint_angles = np.clip(joint_angles, self.lowers, self.uppers)

        self.mj_data.qpos[:] = joint_angles
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        site = self.mj_data.site(self.site_name)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, site.xmat.copy())
        return np.concatenate((site.xpos, quat))


if __name__ == "__main__":
    # model_checkpoint = epath.Path(
    #     "./checkpoints/regression-mlp-epoch=999-train_loss=0.2214.ckpt"
    # )
    # piper_ik = PiperLearnedIK(model_checkpoint)

    # ee_pos_quat = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    # joint_angles = piper_ik(ee_pos_quat)
    # print(joint_angles)

    piper_ik = Piper3DOFIK(
        constants.PIPER_RENDERED_NORMAL_XML, site_name="wrist_base_site"
    )
    joint_angles = np.zeros(8)
    joint_angles[0] = 1.5
    piper_fk = PiperFK(site_name="wrist_base_site")
    ee_pos_quat = piper_fk(joint_angles)

    est_joint_angles = piper_ik(ee_pos_quat)
    after_fk = piper_fk(est_joint_angles)
    print(f"{ee_pos_quat=}")
    print(f"{after_fk=}")
    print(f"{est_joint_angles=}")
    print(f"{joint_angles=}")
    print(np.linalg.norm(est_joint_angles - joint_angles[:3]))

    # print("\n\n\n6DOF IK\n")
    # piper_6dof_ik = Piper6DOFIK(
    #     constants.PIPER_RENDERED_NORMAL_XML, site_name="gripper_site_x_forward"
    # )
    # joint_angles = np.zeros(8)
    # joint_angles[4] = -np.pi / 2
    # piper_fk = PiperFK(site_name="gripper_site_x_forward")
    # ee_pos_quat = piper_fk(joint_angles)

    # est_joint_angles = piper_6dof_ik(ee_pos_quat, initial_q=np.zeros(8))
    # after_fk = piper_fk(est_joint_angles)
    # print(f"{ee_pos_quat=}")
    # print(f"{after_fk=}")
    # print(f"{est_joint_angles=}")
    # print(f"{joint_angles=}")
    # print(np.linalg.norm(est_joint_angles - joint_angles[:6]))

    print("\n\n\n6DOF IK\n")
    piper_6dof_ik = Piper6DOFIK(
        constants.PIPER_RENDERED_NORMAL_XML, site_name="gripper_site_x_forward"
    )
    piper_6dof_fk = PiperFK(site_name="gripper_site_x_forward")
    # joint_angles = np.zeros(8)
    # joint_angles[4] = -np.pi / 2
    # piper_fk = PiperFK(site_name="gripper_site_x_forward")
    # ee_pos_quat = piper_fk(joint_angles)
    ee_pos_quat = np.array(
        [
            0,
            0,
            0,
            0.7071068,  # w
            0,  # x
            -0.7071068,  # y
            0,  # z
        ]
    )

    est_joint_angles = piper_6dof_ik(ee_pos_quat, initial_q=np.zeros(8))
    after_fk = piper_6dof_fk(est_joint_angles)
    print(f"{ee_pos_quat=}")
    print(f"{after_fk=}")
    print(f"{est_joint_angles=}")
    # print(f"{joint_angles=}")
    # print(np.linalg.norm(est_joint_angles - joint_angles[:6]))
