import msgpack
import zmq

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://127.0.0.1:5001")

    while True:
        msg = {
            "hand": 0,
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"w": 0, "x": 0, "y": 0, "z": 0},
        }
        socket.send(msgpack.packb(msg))
        print("sent message")
        input("Press Enter to send another message")
