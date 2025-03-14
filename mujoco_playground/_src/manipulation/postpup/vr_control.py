from dataclasses import dataclass
import time
from typing import Callable, Optional
import numpy as np
from etils import epath
from mujoco_playground._src.manipulation.postpup import (
    base,
    kinematics,
    model_utils,
    constants,
    place_letter,
    vr_client,
)
import mujoco.viewer as viewer
import mujoco
import pyautogui
import msgpack
import pytransform3d.transformations
import zmq
import threading
from collections import deque

import pytransform3d
import click
from pynput.mouse import Controller


class VRController:
    def __init__(self, model_checkpoint: epath.Path, get_goal_fn: Callable):
        # self.ik = kinematics.PiperLearnedIK(model_checkpoint, n_joints=6)
        self.ik = kinematics.Piper3DOFIK(constants.PIPER_RENDERED_NORMAL_XML)
        self.get_goal_fn = get_goal_fn

    def control_fn(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # map mouse position to -1, 1

        # goal = np.array([0.16866421, 0.0958612, 0.37914733])
        goal_pos_quat = self.get_goal_fn()

        # works
        print("doing ik")
        data.ctrl[:6] = self.ik(goal_pos_quat)
        print("done ik")


class IK3DOFController:
    def __init__(self, get_goal_fn: Callable):
        self.get_goal_fn = get_goal_fn
        self.ik = kinematics.Piper3DOFIK(
            dofs=3,
            model_xml=constants.PIPER_RENDERED_NORMAL_XML,
            site_name="gripper_site_x_forward",
        )
        self.last_solution: Optional[np.ndarray] = None

    def control_fn(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        goal_pos_quat = self.get_goal_fn()
        joint_angles = self.ik(goal_pos_quat, initial_guess=self.last_solution)
        self.last_solution = joint_angles
        # print joint angles to 2 dec

        if joint_angles is not None:
            # print(
            #     f"3dof ik result: j0: {joint_angles[0]:.2f}, j1: {joint_angles[1]:.2f}, j2: {joint_angles[2]:.2f}, "
            # )
            data.ctrl[:3] = joint_angles


class IK6DOFController:
    def __init__(self, get_goal_fn: Callable):
        self.get_goal_fn = get_goal_fn
        self.ik = kinematics.Piper6DOFIK(constants.PIPER_RENDERED_NORMAL_XML)

    def control_fn(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        goal_pos_quat = self.get_goal_fn()
        joint_angles = self.ik(goal_pos_quat)
        # print joint angles to 2 dec
        print(
            f"j0: {joint_angles[0]:.2f}, j1: {joint_angles[1]:.2f}, j2: {joint_angles[2]:.2f}, "
            + f"j3: {joint_angles[3]:.2f}, j4: {joint_angles[4]:.2f}, j5: {joint_angles[5]:.2f}"
        )
        data.ctrl[:6] = joint_angles


class DifferentialIKController:
    def __init__(
        self,
        get_goal_velocity_fn: Callable,
        dt: float,
        mode: str,
        keep_nominal: bool = False,
        site_name: str = "gripper_site_x_forward",
        initial_angles: np.ndarray = np.array([0, 1.0, -1.0, 0, 0.7, 0, 0, 0]),
        joint_velocity_limit: float = 3.0,
    ):
        self.get_goal_velocity_fn = get_goal_velocity_fn
        self.mode = mode
        self.keep_nominal = keep_nominal
        if self.keep_nominal:
            raise NotImplementedError("keep_nominal==True not implemented")

        self.diff_ik = kinematics.PiperDifferentialIK(
            model_xml=constants.PIPER_RENDERED_NORMAL_XML,
            site_name="gripper_site_x_forward",
            mode="full" if self.keep_nominal else mode,
            joint_velocity_limit=joint_velocity_limit,
        )
        self.fk = kinematics.PiperFK(
            constants.PIPER_RENDERED_NORMAL_XML, site_name=site_name
        )
        self.dt = dt
        self.joint_angles = initial_angles.copy()

    def control_fn(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        goal_vel = self.get_goal_velocity_fn()
        joint_velocities = self.diff_ik(
            joint_angles=self.joint_angles, desired_ee_velocity=goal_vel
        )
        print(
            f"v0: {joint_velocities[0]:.2f}, v1: {joint_velocities[1]:.2f}, v2: {joint_velocities[2]:.2f}, "
            + f"v3: {joint_velocities[3]:.2f}, v4: {joint_velocities[4]:.2f}, v5: {joint_velocities[5]:.2f}"
        )

        self.joint_angles[:6] = self.joint_angles[:6] + joint_velocities[:] * self.dt
        self.joint_angles[:6] = np.clip(
            self.joint_angles[:6], self.fk.lowers[:6], self.fk.uppers[:6]
        )
        data.ctrl[:6] = self.joint_angles[:6]


class DummyController:
    def __init__(self, input_fn: Callable):
        self.input_fn = input_fn
        pass

    def control_fn(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        out = self.input_fn()
        print(f"{time.time()} input: {out}")
        pass


class MouseClient:
    def __init__(self):
        pass

    def get_ee_pos_quat(self):
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        mouse_x, mouse_y = pyautogui.position()

        # Normalize coordinates to range -1 to 1
        x_norm = (mouse_x - screen_width / 2) / (screen_width / 2)
        y_norm = (mouse_y - screen_height / 2) / (screen_height / 2)

        x = 0.16866421 + y_norm * -0.3
        y = 0.0 + x_norm * -0.3
        z = 0.37914733  # + y_norm * 0.3
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        # print(
        #     f"mx: {x_norm:.4f}, my: {y_norm:.4f} x: {x:.4f}, y: {y:.4f}, z: {z:.4f} quat: {quat}"
        # )
        return np.array([x, y, z, *quat])


# TODO: DOES NOT WORK
class KeyboardVelocityClient:
    def __init__(self):
        self.velocity = np.zeros(6)
        self.key_map = {
            "w": np.array([0.1, 0, 0, 0, 0, 0]),
            "s": np.array([-0.1, 0, 0, 0, 0, 0]),
            "a": np.array([0, 0.1, 0, 0, 0, 0]),
            "d": np.array([0, -0.1, 0, 0, 0, 0]),
            "up": np.array([0, 0, 0, 0.1, 0, 0]),
            "down": np.array([0, 0, 0, -0.1, 0, 0]),
            "left": np.array([0, 0, 0, 0, 0.1, 0]),
            "right": np.array([0, 0, 0, 0, -0.1, 0]),
        }
        self.listener = threading.Thread(target=self._listen_to_keyboard)
        self.listener.daemon = True
        self.listener.start()

    def _listen_to_keyboard(self):
        import pynput.keyboard as keyboard

        def on_press(key):
            try:
                if key.char in self.key_map:
                    self.velocity += self.key_map[key.char]
            except AttributeError:
                if key in self.key_map:
                    self.velocity += self.key_map[key]

        def on_release(key):
            try:
                if key.char in self.key_map:
                    self.velocity -= self.key_map[key.char]
            except AttributeError:
                if key in self.key_map:
                    self.velocity -= self.key_map[key]

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def get_ee_velocity(self):
        print("Keyboard client velocity: ", self.velocity)
        return self.velocity


class MouseVelocityClient:
    def __init__(self, multiplier: float = 0.25, mode: str = "linear"):
        self.multiplier = multiplier
        self.mode = mode
        assert self.mode in ["linear", "angular"]

    def get_ee_velocity(self):
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        mouse_x, mouse_y = pyautogui.position()
        # Normalize coordinates to range -1 to 1
        x_norm = (mouse_x - screen_width / 2) / (screen_width / 2)
        y_norm = (mouse_y - screen_height / 2) / (screen_height / 2)

        if self.mode == "linear":
            ee_velocity = np.array(
                [
                    -y_norm * self.multiplier,
                    -x_norm * self.multiplier,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        elif self.mode == "angular":
            ee_velocity = np.array([0, 0, 0, x_norm, y_norm, 0])
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        print(f"Mouse client velocity: {ee_velocity}")
        return ee_velocity


class MouseOrientationClient:
    def __init__(self):
        pass

    def get_ee_pos_quat(self):
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        mouse_x, mouse_y = pyautogui.position()

        # Normalize coordinates to range -1 to 1
        x_norm = (mouse_x - screen_width / 2) / (screen_width / 2)
        y_norm = (mouse_y - screen_height / 2) / (screen_height / 2)

        # a good base position for robot arm
        x = 0.16866421
        y = 0.0958612
        z = 0.37914733

        # use x_norm to make quaternion rotated about y axis
        quat_y = np.array([np.cos(x_norm / 2), 0.0, np.sin(x_norm / 2), 0.0])
        # use y norm to make quaternion rotated about z axis
        quat_z = np.array(
            [
                np.cos(y_norm / 2),
                0.0,
                0.0,
                np.sin(y_norm / 2),
            ]
        )
        quat_x = np.array(
            [
                np.cos(y_norm / 2),
                np.sin(y_norm / 2),
                0.0,
                0.0,
            ]
        )
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, quat_y, quat_x)
        mujoco.mju_mulQuat(quat, quat, quat_z)
        # print(
        #     f"mx: {x_norm:.2f}, my: {y_norm:.2f} x: {x:.2f}, y: {y:.2f}, z: {z:.2f} q.w: {quat[0]:.2f}, q.x: {quat[1]:.2f}, q.y: {quat[2]:.2f}, q.z: {quat[3]:.2f}"
        # )
        return np.array([x, y, z, *quat])


@click.command()
@click.option(
    "--client",
    type=click.Choice(
        [
            "vr",
            "mouse",
            "mouse_orientation",
            "mouse_velocity_linear",
            "mouse_velocity_angular",
            "keyboard_velocity",
        ]
    ),
    default="vr",
    help="Specify the client to use.",
)
@click.option(
    "--ik",
    type=click.Choice(
        ["3dof", "6dof", "diff_orientation", "diff_position", "diff", "dummy"]
    ),
    default="3dof",
    help="Specify the IK controller to use.",
)
def main(client, ik):
    if client == "vr":
        goal_specifier = vr_client.VRClient()
    elif client == "mouse":
        goal_specifier = MouseClient()
    elif client == "mouse_orientation":
        goal_specifier = MouseOrientationClient()
    elif client == "mouse_velocity_linear":
        goal_specifier = MouseVelocityClient(mode="linear")
    elif client == "mouse_velocity_angular":
        goal_specifier = MouseVelocityClient(mode="angular")
    elif client == "keyboard_velocity":
        goal_specifier = KeyboardVelocityClient()
    else:
        raise ValueError(f"Unknown client {client}")

    if ik == "3dof":
        controller = IK3DOFController(get_goal_fn=goal_specifier.get_ee_pos_quat)
    elif ik == "6dof":
        controller = IK6DOFController(get_goal_fn=goal_specifier.get_ee_pos_quat)
    elif ik == "diff":
        controller = DifferentialIKController(
            get_goal_velocity_fn=goal_specifier.get_ee_velocity,
            dt=0.002,
            mode="full",
        )
    elif ik == "diff_orientation":
        controller = DifferentialIKController(
            get_goal_velocity_fn=goal_specifier.get_ee_velocity,
            dt=0.002,
            mode="orientation_only",
        )
    elif ik == "diff_position":
        controller = DifferentialIKController(
            get_goal_velocity_fn=goal_specifier.get_ee_velocity,
            dt=0.002,
            mode="position_only",
        )
    elif ik == "dummy":
        controller = DummyController(input_fn=goal_specifier.get_ee_velocity)
    else:
        raise ValueError(f"Unknown ik controller {ik}")

    def load_scene_callback():
        print("loading scene")
        return model_utils.load_callback(
            xml_path=constants.PLACE_LETTER_SCENE_XML,
            control_fn=controller.control_fn,
            dt=0.002,
        )

    print("Starting viewer")
    viewer.launch(loader=load_scene_callback)


if __name__ == "__main__":
    print("HI")
    main()
