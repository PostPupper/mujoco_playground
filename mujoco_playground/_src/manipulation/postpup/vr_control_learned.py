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
        self.ik = kinematics.Piper3DOFIK(constants.PIPER_RENDERED_NORMAL_XML)

    def control_fn(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        goal_pos_quat = self.get_goal_fn()
        joint_angles = self.ik(goal_pos_quat)
        # print joint angles to 2 dec
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


class VRClient:
    @dataclass
    class HandData:
        T_world_from_controller: np.ndarray
        T_world_from_reference: np.ndarray
        T_reference_from_my_controller: np.ndarray
        T_controller_from_my_controller: np.ndarray
        mode: str

        def __init__(self):
            self.T_world_from_controller = np.eye(4)
            self.T_world_from_reference = np.eye(4)
            self.T_reference_from_my_controller = np.eye(4)
            self.T_controller_from_my_controller = np.eye(4)
            self.mode = "inactive"

    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect("tcp://localhost:5555")

        # Should only be accessed by the receive_messages_thread
        self.hand_data = [VRClient.HandData(), VRClient.HandData()]
        self.hand_data[1].T_controller_from_my_controller = np.eye(4)
        self.hand_data[1].T_controller_from_my_controller[:3, :3] = np.array(
            [[0, -1, 1], [-1, 0, 0], [0, 0, -1]]
        )

        # Thread safe access to the latest hand data
        self.latest_hand_T_reference_from_my_controller = [
            deque([np.eye(4)], maxlen=1),
            deque([np.eye(4)], maxlen=1),
        ]

        self.recv_thread = threading.Thread(
            target=self.receive_messages_thread, daemon=True
        )
        self.recv_thread.start()

    def update_hands(self, data):
        hand_data = self.hand_data[data["hand"]]
        pos_quat = np.array(
            [
                data["position"]["x"],
                data["position"]["y"],
                data["position"]["z"],
                data["orientation"]["w"],
                data["orientation"]["x"],
                data["orientation"]["y"],
                data["orientation"]["z"],
            ]
        )
        hand_data.T_world_from_controller = (
            pytransform3d.transformations.transform_from_pq(pos_quat)
        )
        # print(data)
        # print(hand_data)

        # Update reference point if trigger transitioned from <0.5 to >=0.5
        if hand_data.mode == "inactive" and data["trigger"] >= 0.5:
            print("SWITCHING TO ACTIVE AND SETTING ORIGIN")
            hand_data.mode = "active"
            hand_data.T_world_from_reference = hand_data.T_world_from_controller.copy()
            print(f"{hand_data.T_world_from_reference}=")

        elif hand_data.mode == "active" and data["trigger"] < 0.5:
            hand_data.mode = "inactive"

        if hand_data.mode == "active":
            hand_data.T_reference_from_my_controller = (
                pytransform3d.transformations.invert_transform(
                    hand_data.T_world_from_reference
                )
                @ hand_data.T_world_from_controller
                @ hand_data.T_controller_from_my_controller
            )
            print("relative pos: ", hand_data.T_reference_from_my_controller[:3, 3])
        else:
            hand_data.T_reference_from_my_controller = np.eye(4)

        self.latest_hand_T_reference_from_my_controller[data["hand"]].append(
            hand_data.T_reference_from_my_controller
        )

    def receive_messages_thread(self):
        while True:
            msg = self.socket.recv()
            hand_data = msgpack.unpackb(msg)
            print(f"{time.time()} UPDATING HANDS")
            self.update_hands(hand_data)

    def get_right_hand(self):
        if len(self.latest_hand_T_reference_from_my_controller[1]) == 0:
            return None
        data = self.latest_hand_T_reference_from_my_controller[1].popleft()
        self.latest_hand_T_reference_from_my_controller[1].append(data)
        return data

    def get_ee_pos_quat(self):
        T_reference_from_controller_R = self.get_right_hand()
        pos_quat = pytransform3d.transformations.pq_from_transform(
            T_reference_from_controller_R
        )
        x = 0.16866421
        y = 0.0958612
        z = 0.37914733
        ee_pos_quat = pos_quat.copy()
        ee_pos_quat[:3] += [x, y, z]

        # print ee_pos_quat with each element 3 dec
        # print(
        #     f"ee_pos_quat: x: {ee_pos_quat[0]:.3f}, y: {ee_pos_quat[1]:.3f}, z: {ee_pos_quat[2]:.3f}, "
        #     + f"q.w: {ee_pos_quat[3]:.3f}, q.x: {ee_pos_quat[4]:.3f}, q.y: {ee_pos_quat[5]:.3f}, q.z: {ee_pos_quat[6]:.3f}"
        # )
        return ee_pos_quat


class MouseClient:
    def __init__(self):
        pass

    def get_goal(self):
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        mouse_x, mouse_y = pyautogui.position()

        # Normalize coordinates to range -1 to 1
        x_norm = (mouse_x - screen_width / 2) / (screen_width / 2)
        y_norm = (mouse_y - screen_height / 2) / (screen_height / 2)

        # return x_norm, y_norm
        z = 0.37914733  # + y_norm * 0.3
        x = 0.16866421 + y_norm * -0.3
        y = 0.0958612 + x_norm * -0.3
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        print(
            f"mx: {x_norm:.4f}, my: {y_norm:.4f} x: {x:.4f}, y: {y:.4f}, z: {z:.4f} quat: {quat}"
        )
        return np.array([x, y, z, *quat])


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


def load_scene_callback():
    print("loading scene")
    return model_utils.load_callback(
        xml_path=constants.PLACE_LETTER_SCENE_XML,
        control_fn=controller.control_fn,
        dt=0.002,
    )


if __name__ == "__main__":
    goal_specifier = VRClient()
    # goal_specifier = MouseOrientationClient()

    # controller = IK6DOFController(get_goal_fn=goal_specifier.get_ee_pos_quat)
    controller = IK3DOFController(get_goal_fn=goal_specifier.get_ee_pos_quat)
    # controller = VRController(

    viewer.launch(loader=load_scene_callback)
