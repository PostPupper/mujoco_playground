import copy
from dataclasses import dataclass
from typing import Tuple
import pytransform3d
import numpy as np

import zmq
import threading
import msgpack

from collections import deque
import time


class VRClient:
    @dataclass
    class HandData:
        T_world_from_controller: np.ndarray
        T_world_from_my_reference: np.ndarray
        T_reference_from_my_controller: np.ndarray
        T_controller_from_my_controller: np.ndarray
        v_reference_from_my_controller: np.ndarray

        mode: str

        def __init__(self):
            self.T_world_from_controller = np.eye(4)
            self.T_world_from_my_reference = np.eye(4)
            self.T_reference_from_my_controller = np.eye(4)
            self.T_controller_from_my_controller = np.eye(4)
            self.v_reference_from_my_controller = np.zeros(3)  # linear velocity
            self.w_reference_from_my_controller = np.zeros(3)  # angular velocity
            self.mode = "inactive"

    # @dataclass
    # class LatestHandData:
    #     T_reference_from_my_controller: np.ndarray
    #     v_reference_from_my_controller: np.ndarray
    #     w_reference_from_my_controller: np.ndarray

    #     def __init__(self):
    #         self.T_reference_from_my_controller = np.eye(4)
    #         self.v_reference_from_my_controller = np.zeros(3)
    #         self.w_reference_from_my_controller = np.zeros(3)

    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect("tcp://localhost:5555")

        # Should only be accessed by the receive_messages_thread
        self.hand_data = [VRClient.HandData(), VRClient.HandData()]
        self.hand_data[1].T_controller_from_my_controller = np.eye(4)
        self.hand_data[1].T_controller_from_my_controller[:3, :3] = np.array(
            [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
        )

        # Thread safe access to the latest hand data
        self.latest_hand_data = [
            deque([VRClient.HandData()], maxlen=1),
            deque([VRClient.HandData()], maxlen=1),
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
        v_world_from_controller = np.array(
            [
                data["velocity"]["x"],
                data["velocity"]["y"],
                data["velocity"]["z"],
            ]
        )
        w_world_from_controller = np.array(
            [
                data["angular_velocity"]["x"],
                data["angular_velocity"]["y"],
                data["angular_velocity"]["z"],
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
            hand_data.T_world_from_my_reference = (
                hand_data.T_world_from_controller
                @ hand_data.T_controller_from_my_controller
            )
            print(f"{hand_data.T_world_from_my_reference}=")

        elif hand_data.mode == "active" and data["trigger"] < 0.5:
            hand_data.mode = "inactive"

        if hand_data.mode == "active":
            hand_data.T_reference_from_my_controller = (
                pytransform3d.transformations.invert_transform(
                    hand_data.T_world_from_my_reference
                )
                @ hand_data.T_world_from_controller
                @ hand_data.T_controller_from_my_controller
            )
            hand_data.v_reference_from_my_controller = (
                hand_data.T_world_from_my_reference[:3, :3].T @ v_world_from_controller
            )
            hand_data.w_reference_from_my_controller = (
                hand_data.T_world_from_my_reference[:3, :3].T @ w_world_from_controller
            )
            print(
                "relative pos: ",
                hand_data.T_reference_from_my_controller[:3, 3],
                "vel: ",
                hand_data.v_reference_from_my_controller,
                "ang vel: ",
                hand_data.w_reference_from_my_controller,
            )
        else:
            hand_data.T_reference_from_my_controller = np.eye(4)
            hand_data.v_reference_from_my_controller = np.zeros(3)
            hand_data.w_reference_from_my_controller = np.zeros(3)

        self.latest_hand_data[data["hand"]].append(copy.deepcopy(hand_data))

    def receive_messages_thread(self):
        while True:
            msg = self.socket.recv()
            hand_data = msgpack.unpackb(msg)
            print(f"{time.time()} UPDATING HANDS")
            self.update_hands(hand_data)

    def get_right_hand(self):
        if len(self.latest_hand_data[1]) == 0:
            return None
        data = self.latest_hand_data[1].popleft()
        self.latest_hand_data[1].append(copy.deepcopy(data))
        return data

    def get_ee_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        hand_data = self.get_right_hand()
        return np.concatenate(
            (
                1.5 * hand_data.v_reference_from_my_controller,
                hand_data.w_reference_from_my_controller,
            )
        )

    def get_ee_pos_quat(self) -> np.ndarray:
        hand_data = self.get_right_hand()
        pos_quat = pytransform3d.transformations.pq_from_transform(
            hand_data.T_reference_from_controller
        )
        x = 0.16866421
        y = 0.0
        z = 0.37914733
        ee_pos_quat = pos_quat.copy()
        ee_pos_quat[:3] = ee_pos_quat[:3] * 2 + np.array([x, y, z])

        # print ee_pos_quat with each element 3 dec
        # print(
        #     f"ee_pos_quat: x: {ee_pos_quat[0]:.3f}, y: {ee_pos_quat[1]:.3f}, z: {ee_pos_quat[2]:.3f}, "
        #     + f"q.w: {ee_pos_quat[3]:.3f}, q.x: {ee_pos_quat[4]:.3f}, q.y: {ee_pos_quat[5]:.3f}, q.z: {ee_pos_quat[6]:.3f}"
        # )
        return ee_pos_quat
