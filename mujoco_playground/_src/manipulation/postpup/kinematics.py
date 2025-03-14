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

np.set_printoptions(formatter={"float": "{: 6.3f}".format})


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


class PiperDifferentialIK:
    def __init__(
        self,
        model_xml: epath.Path,
        site_name: str,
        mode: str = "full",
        joint_velocity_limit:float = 3.0,
    ):
        self.site_name = site_name
        self.fk = PiperFK(model_xml=model_xml, site_name=self.site_name)
        self.mj_model = self.fk.mj_model
        self.mj_data = self.fk.mj_data

        self.site_id = self.mj_model.site(self.site_name).id

        self.mode = mode
        assert self.mode in ["full", "position_only", "orientation_only"]

        self.arm_dofs = 6
        self.base_arm_dofs = 3
        self.joint_velocity_limit = joint_velocity_limit

    def limit_joint_velocities(self, joint_velocities: np.ndarray):
        return np.clip(joint_velocities, -self.joint_velocity_limit, self.joint_velocity_limit)

    # TODO: TEST THIS
    def error_twist(self, ee_pos_quat: np.ndarray):
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        mujoco.mj_comPos(self.mj_model, self.mj_data)
        site = self.mj_data.site(self.site_name)
        p_site_from_goal = site.xpos - ee_pos_quat[:3]

        R_world_from_site = site.xmat.copy().reshape((3, 3))
        R_world_from_goal = np.zeros(9)
        mujoco.mju_quat2Mat(R_world_from_goal, ee_pos_quat[3:])
        R_world_from_goal = R_world_from_goal.reshape((3, 3))
        R_site_from_goal = R_world_from_site.T @ R_world_from_goal
        w_site_from_goal = rotation_matrix_to_angle_axis(R_site_from_goal)
        w_goal_from_site = -w_site_from_goal

        if self.mode == "full":
            error = np.concatenate(
                (
                    p_site_from_goal.copy(),
                    w_goal_from_site.copy(),
                )
            )
        elif self.mode == "orientation_only":
            error = w_goal_from_site
        elif self.mode == "position_only":
            error = p_site_from_goal
        else:
            raise NotImplementedError()

        return error

    def set_joint_angles(self, joint_angles: np.ndarray):
        self.mj_data.qpos[:] = joint_angles.copy()

    def jacobian_6dof(self):
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        mujoco.mj_comPos(self.mj_model, self.mj_data)
        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self.site_id)
        return np.vstack((jacp[:, : self.arm_dofs], jacr[:, : self.arm_dofs]))

    def __call__(
        self,
        joint_angles: np.ndarray,
        desired_ee_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the joint velocities that will move the end effector with the desired velocity.
        Args:
            joint_angles (np.ndarray): Joint angles.
            desired_ee_velocity (np.ndarray): Desired end effector velocity (linear and angular).
        Returns:
            np.ndarray: Joint velocities of size 6
        """
        assert joint_angles.size == self.mj_model.nq
        assert desired_ee_velocity.size == 6

        self.set_joint_angles(joint_angles)
        J = self.jacobian_6dof()

        if self.mode == "full":
            joint_velocities = np.linalg.pinv(J) @ desired_ee_velocity
            joint_velocities = self.limit_joint_velocities(joint_velocities)
            return joint_velocities
        elif self.mode == "orientation_only":
            joint_velocities = np.linalg.pinv(J[3:, :]) @ desired_ee_velocity[3:]
            joint_velocities = self.limit_joint_velocities(joint_velocities)
            return joint_velocities
        elif self.mode == "position_only":
            joint_velocities = np.linalg.pinv(J[:3, :3]) @ desired_ee_velocity[:3]
            joint_velocities = self.limit_joint_velocities(joint_velocities)
            return np.concatenate((joint_velocities, np.zeros(3)))
        else:
            raise NotImplementedError


class Piper3DOFIK:
    def __init__(
        self,
        model_xml: epath.Path,
        dofs: int,
        site_name: str = "wrist_base_site",
        orientation_weight=1.0,
        mode: str = "full",
    ):
        self.site_name = site_name
        self.fk = PiperFK(model_xml=model_xml, site_name=self.site_name)
        self.mj_model = self.fk.mj_model
        self.mj_data = self.fk.mj_data

        self.dofs = dofs
        assert self.dofs == 3 or self.dofs == 6

        # error = pos_error + orientation_weight * orientation_error
        self.orientation_weight = orientation_weight

        self.mode = mode
        assert self.mode in ["full", "position_only", "orientation_only"]

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
        gradient_descent_step_size=20.0,
    ) -> Optional[np.ndarray]:
        # non-singularity starting position
        if initial_guess is None:
            initial_angles = np.array([0.0, 0.5, -0.9, -0.2, 0.4, 0, 0, 0])[: self.dofs]
        else:
            initial_angles = initial_guess.copy()

        for attempts in range(max_seeds):
            joint_angles = initial_angles.copy()
            for i in range(max_iters):
                self.mj_data.qpos[: self.dofs] = joint_angles
                mujoco.mj_kinematics(self.mj_model, self.mj_data)
                mujoco.mj_comPos(self.mj_model, self.mj_data)
                jacp = np.zeros((3, self.mj_model.nv))
                jacr = np.zeros((3, self.mj_model.nv))
                mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self.site_id)

                site = self.mj_data.site(self.site_name)
                pos_error = site.xpos - ee_pos_quat[:3]

                R_world_from_site = site.xmat.copy().reshape((3, 3))
                R_world_from_goal = np.zeros(9)
                mujoco.mju_quat2Mat(R_world_from_goal, ee_pos_quat[3:])
                R_world_from_goal = R_world_from_goal.reshape((3, 3))
                R_site_from_goal = R_world_from_site.T @ R_world_from_goal
                w_site_from_goal = rotation_matrix_to_angle_axis(R_site_from_goal)
                w_goal_from_site = -w_site_from_goal

                if self.dofs == 3:
                    error = pos_error.copy()
                elif self.dofs == 6:
                    if self.mode == "full":
                        error = np.concatenate(
                            (
                                pos_error.copy(),
                                self.orientation_weight * w_goal_from_site.copy(),
                            )
                        )
                    elif self.mode == "orientation_only":
                        error = w_goal_from_site
                    else:
                        raise NotImplementedError()

                if verbose:
                    msg = (
                        f"{i=} norm: {np.linalg.norm(error):0.6f} error: {error[0]:.3f} {error[1]:.3f} {error[2]:.3f} "
                        + (
                            f"{error[3]:.3f} {error[4]:.3f} {error[5]:.3f} "
                            if self.dofs == 6 and self.mode == "full"
                            else ""
                        )
                        + f"q: {joint_angles[0]:0.2f} {joint_angles[1]:0.2f} {joint_angles[2]:0.2f}"
                        + (
                            f" {joint_angles[3]:0.2f} {joint_angles[4]:0.2f} {joint_angles[5]:0.2f}"
                            if self.dofs == 6
                            else ""
                        )
                    )
                    print(msg)

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
                if self.dofs == 3:
                    J = jacp[:, :3].copy()
                elif self.dofs == 6:
                    if self.mode == "full":
                        Jp = jacp[:, :6].copy()
                        Jr = jacr[:, :6].copy()
                        J = np.vstack((Jp, Jr))
                        # breakpoint()
                    elif self.mode == "orientation_only":
                        J = jacr[:, :6].copy()
                    else:
                        raise NotImplementedError()

                # gradient descent
                if method == "gradient_descent":
                    joint_angles -= gradient_descent_step_size * J.T @ error
                elif method == "newton":
                    lambda_identity = 1e-3 * np.eye(J.shape[1])
                    joint_velocities = (
                        np.linalg.inv(J.T @ J + lambda_identity) @ J.T @ error
                    )
                    if verbose:
                        print(f"{i}: {joint_velocities}")
                    joint_angles -= 0.1 * joint_velocities
                else:
                    raise ValueError(f"Unknown method: {method}")

                # TODO figure out whether this is actually bad
                # joint_angles = self.normalize_angles(joint_angles)
                # joint_angles = np.clip(
                #     joint_angles, self.fk.lowers[: self.dofs], self.fk.uppers[self.dofs]
                # )

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

                print(
                    f"{i=} norm: {np.linalg.norm(w_site_from_goal)} w_error: {w_site_from_goal[0]:.3f} {w_site_from_goal[1]:.3f} {w_site_from_goal[2]:.3f}"
                    # + f"q[:3]: {joint_angles[0]:0.2f} {joint_angles[1]:0.2f} {joint_angles[2]:0.2f}"
                )
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
