import msgpack
import zmq
from collections import deque
import threading

latest_hand_data: deque = deque(maxlen=1)
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
import time


def recv_thread():
    while True:
        msg = socket.recv()
        hand_data = msgpack.unpackb(msg)
        latest_hand_data.append(hand_data)
        # print(hand_data)


if __name__ == "__main__":
    socket.connect("tcp://127.0.0.1:5555")

    thread = threading.Thread(target=recv_thread, daemon=True)
    thread.start()

    while len(latest_hand_data) == 0:
        time.sleep(0.1)
    while True:
        print(latest_hand_data[0])
        time.sleep(0.01)
