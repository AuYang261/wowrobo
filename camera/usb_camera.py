# Description: 控制usb相机的基本操作
import cv2


def open_camera(camera_index=0):
    """
    Opens a USB camera and sets the resolution.

    Parameters:
        camera_index (int): Index of the camera to open.

    Returns:
        cap: Opened video capture object.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError(f"Unable to open camera with index {camera_index}")

    return cap
