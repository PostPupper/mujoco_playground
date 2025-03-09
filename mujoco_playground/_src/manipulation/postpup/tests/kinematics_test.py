import pytest
import numpy as np
from mujoco_playground._src.manipulation.postpup import constants
from etils import epath

from mujoco_playground._src.manipulation.postpup.kinematics import (
    PiperLearnedIK,
    Piper3DOFIK,
    Piper6DOFIK,
    PiperFK,
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
        ("newton", (1.5, 0, 0), 1e-6, 20, 1e-4),
        ("newton", (1.5, 1, -1), 1e-6, 20, 1e-4),
        ("gradient_descent", (1.5, 0, 0), 1e-5, 1000, 1e-3),
        # ("gradient_descent", (1.5, 1, -1), 1e-5, 1000, 1e-2), # This test case is not working
    ],
)
def test_piper_3dof_ik(
    solve_method, test_angles, ik_tolerance, ik_max_iters, solution_tolerance
):
    piper_ik = Piper3DOFIK(
        constants.PIPER_RENDERED_NORMAL_XML,
        site_name="wrist_base_site",
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


# def test_piper_6dof_ik():
#     piper_6dof_ik = Piper6DOFIK(
#         constants.PIPER_RENDERED_NORMAL_XML, site_name="gripper_site_x_forward"
#     )
#     piper_6dof_fk = PiperFK(site_name="gripper_site_x_forward")
#     ee_pos_quat = np.array([0, 0, 0, 0.7071068, 0, -0.7071068, 0])
#     est_joint_angles = piper_6dof_ik(ee_pos_quat, initial_q=np.zeros(8))
#     after_fk = piper_6dof_fk(est_joint_angles)
#     assert np.allclose(ee_pos_quat, after_fk, atol=1e-4)


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


# def test_piper_fk():
#     piper_fk = PiperFK(site_name="gripper_site_x_forward")
#     joint_angles = np.zeros(8)
#     ee_pos_quat = piper_fk(joint_angles)
#     assert ee_pos_quat.shape == (7,)
