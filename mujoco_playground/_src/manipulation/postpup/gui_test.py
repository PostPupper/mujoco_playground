import pyautogui
import time

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

while True:
    # Get current mouse position
    mouse_x, mouse_y = pyautogui.position()

    # Normalize coordinates to range -1 to 1
    x_norm = (mouse_x - screen_width / 2) / (screen_width / 2)
    y_norm = (mouse_y - screen_height / 2) / (screen_height / 2)

    print("Normalized Mouse Coordinates:")
    print("x:", x_norm)
    print("y:", y_norm)
    time.sleep(0.01)
