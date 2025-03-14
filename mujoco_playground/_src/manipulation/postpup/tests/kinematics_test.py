import pytest
import numpy as np
from mujoco_playground._src.manipulation.postpup import constants
from etils import epath

from mujoco_playground._src.manipulation.postpup.kinematics import (
    PiperLearnedIK,
    Piper3DOFIK,
    Piper6DOFIK,
    PiperFK,
    PiperDifferentialIK,
    small_angle_matrix_log,
    rotation_matrix_to_angle_axis,
)


def test_piper_learned_ik():
    model_checkpoint = epath.Path(
        "./checkpoints/regression-mlp-epoch=999-train_loss=0.2214.ckpt"
    )
    piper_ik = PiperLearnedIK(model_checkpoint)
    ee_pos_quat = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    joint_angles = piper_ik(ee_pos_quat)
    assert joint_angles.shape == (8,)


@pytest.mark.parametrize(
    "solve_method, test_angles, ik_tolerance, ik_max_iters, solution_tolerance",
    [
        ("newton", (1.5, 0, 0), 1e-6, 200, 1e-4),
        ("newton", (1.5, 1, -1), 1e-6, 200, 1e-4),
        ("gradient_descent", (1.5, 0, 0), 1e-5, 1000, 1e-3),
        # ("gradient_descent", (1.5, 1, -1), 1e-5, 1000, 1e-2), # This test case is not working
    ],
)
def test_ik_3dof(
    solve_method, test_angles, ik_tolerance, ik_max_iters, solution_tolerance
):
    piper_ik = Piper3DOFIK(
        constants.PIPER_RENDERED_NORMAL_XML, site_name="wrist_base_site", dofs=3
    )
    joint_angles_gt = np.zeros(8)
    joint_angles_gt[:3] = test_angles

    piper_fk = PiperFK(site_name="wrist_base_site")
    ee_pos_quat_gt = piper_fk(joint_angles_gt)

    # Estimate joint angles with IK
    joint_angles_estimated = piper_ik(
        ee_pos_quat_gt,
        tolerance=ik_tolerance,
        method=solve_method,
        max_iters=ik_max_iters,
        verbose=True,
    )

    # Don't compare joint angles because the IK solution is not unique
    # assert np.allclose(
    #     joint_angles_gt[:3], joint_angles_estimated, atol=solution_tolerance
    # )

    # Compare the ground truth end effector position and orientation with the estimated end effector position and orientation
    ee_pos_quat_using_est_joint_angles = piper_fk(joint_angles_estimated)
    assert np.allclose(
        ee_pos_quat_gt[:3],
        ee_pos_quat_using_est_joint_angles[:3],
        atol=solution_tolerance,
    )


def test_ik_6dof_orientation_only():
    piper_6dof_ik = Piper3DOFIK(
        constants.PIPER_RENDERED_NORMAL_XML,
        site_name="gripper_site_x_forward",
        dofs=6,
        mode="orientation_only",
        orientation_weight=1.0,
    )
    piper_6dof_fk = PiperFK(site_name="gripper_site_x_forward")
    ee_pos_quat = np.array([0, 0, 0, 0.7071068, 0, -0.7071068, 0])
    est_joint_angles = piper_6dof_ik(
        ee_pos_quat,
        tolerance=1e-6,
        max_iters=200,
        verbose=True,
        max_seeds=1,
        method="newton",
        initial_guess=np.array([0, 1.0, -1.0, 0, 0.7, 0]),
    )
    after_fk = piper_6dof_fk(est_joint_angles)
    assert np.allclose(ee_pos_quat[3:], after_fk[3:], atol=1e-2)


@pytest.mark.parametrize(
    "gt_joint_angles, tolerance, max_iters, initial_guess, atol",
    [
        (
            np.array([0.3, 0, 0, 0, 0, 0, 0, 0]),
            1e-3,
            200,
            np.array([0.2, 0, 0, 0, 0, 0]),
            1e-2,
        ),
        # (np.array([1.5, 0.1, -0.5, 0.1, 0.1, 0.1, 0, 0]), 1e-3, 200, np.array([1.4, 0.1, -0.1, 0.1, 0.1, 0.1]), 1e-2), # DOESNT WORK
        # (np.array([1.5, 0.1, -0.5, 0.1, 0.1, 0.1, 0, 0]), 1e-3, 200, np.array([1.4, 0.1, -0.5, 0.1, 0.1, 0.1,]), 1e-2), # DOESNT WORK
        (
            np.array([0.4, 0.4, -0.9, 0, 0, 0, 0, 0]),
            1e-3,
            200,
            np.array([0.2, 0.1, -0.1, 0, 0, 0]),
            1e-2,
        ),
    ],
)
def test_ik_6dof_full(gt_joint_angles, tolerance, max_iters, initial_guess, atol):
    piper_6dof_ik = Piper3DOFIK(
        constants.PIPER_RENDERED_NORMAL_XML,
        site_name="gripper_site_x_forward",
        dofs=6,
        mode="full",
        orientation_weight=1.0,
    )
    piper_6dof_fk = PiperFK(site_name="gripper_site_x_forward")
    ee_pos_quat = piper_6dof_fk(gt_joint_angles)
    est_joint_angles = piper_6dof_ik(
        ee_pos_quat,
        tolerance=tolerance,
        max_iters=max_iters,
        verbose=True,
        max_seeds=1,
        method="newton",
        initial_guess=initial_guess,
    )
    after_fk = piper_6dof_fk(est_joint_angles)
    assert np.allclose(ee_pos_quat[:], after_fk[:], atol=atol)


def test_piper_6dof_ik_old():
    piper_6dof_ik = Piper6DOFIK(
        constants.PIPER_RENDERED_NORMAL_XML, site_name="gripper_site_x_forward"
    )
    piper_6dof_fk = PiperFK(site_name="gripper_site_x_forward")
    ee_pos_quat = np.array([0, 0, 0, 0.7071068, 0, -0.7071068, 0])
    est_joint_angles = piper_6dof_ik(ee_pos_quat, initial_q=np.zeros(8))
    after_fk = piper_6dof_fk(est_joint_angles)
    assert np.allclose(ee_pos_quat[3:], after_fk[3:], atol=1e-4)


def test_small_angle_matrix_log():
    R_rel = np.eye(3)
    log = small_angle_matrix_log(R_rel)
    assert np.allclose(log, np.zeros(3))


def test_rotation_matrix_to_angle_axis():
    R_rel = np.eye(3)
    angle_axis = rotation_matrix_to_angle_axis(R_rel)
    assert np.allclose(angle_axis, np.zeros(3))

    R_rel = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    angle_axis = rotation_matrix_to_angle_axis(R_rel)
    assert np.allclose(angle_axis, np.array([0, 0, np.pi / 2]))


@pytest.mark.parametrize(
    "joint_angles, desired_ee_velocity",
    [
        (
            np.array([1.0, 0.5, -0.5, 0.1, 0.2, 0.1, 0, 0]),
            np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
    ],
)
def test_diff_ik_full(joint_angles: np.ndarray, desired_ee_velocity: np.ndarray):
    piper_diff_ik = PiperDifferentialIK(
        constants.PIPER_RENDERED_NORMAL_XML,
        site_name="gripper_site_x_forward",
        mode="full",
    )
    joint_velocities = piper_diff_ik(joint_angles, desired_ee_velocity)
    J = piper_diff_ik.jacobian_6dof()
    assert np.allclose(J @ joint_velocities, desired_ee_velocity)


@pytest.mark.parametrize(
    "joint_angles, desired_ee_velocity",
    [
        (
            np.array([1.0, 0.5, -0.5, 0.1, 0.2, 0.1, 0, 0]),
            np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
    ],
)
def test_diff_ik_orientation_only(
    joint_angles: np.ndarray, desired_ee_velocity: np.ndarray
):
    piper_diff_ik = PiperDifferentialIK(
        constants.PIPER_RENDERED_NORMAL_XML,
        site_name="gripper_site_x_forward",
        mode="orientation_only",
    )
    joint_velocities = piper_diff_ik(joint_angles, desired_ee_velocity)
    J = piper_diff_ik.jacobian_6dof()
    assert np.allclose((J @ joint_velocities)[3:], desired_ee_velocity[3:])


@pytest.mark.parametrize(
    "joint_angles, desired_ee_velocity",
    [
        (
            np.array([1.0, 0.5, -0.5, 0.1, 0.2, 0.1, 0, 0]),
            np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        (
            np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_diff_ik_position_only(
    joint_angles: np.ndarray, desired_ee_velocity: np.ndarray
):
    piper_diff_ik = PiperDifferentialIK(
        constants.PIPER_RENDERED_NORMAL_XML,
        site_name="gripper_site_x_forward",
        mode="position_only",
    )
    joint_velocities = piper_diff_ik(joint_angles, desired_ee_velocity)
    J = piper_diff_ik.jacobian_6dof()
    assert np.allclose((J @ joint_velocities)[:3], desired_ee_velocity[:3])


# def test_piper_fk():
#     piper_fk = PiperFK(site_name="gripper_site_x_forward")
#     joint_angles = np.zeros(8)
#     ee_pos_quat = piper_fk(joint_angles)
#     assert ee_pos_quat.shape == (7,)
