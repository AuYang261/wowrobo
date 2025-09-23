# Description: 控制usb相机，获取图片用来构建数据集
import cv2
import numpy as np
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from usb_camera import open_camera


def main():
    print("Opening camera...")
    cap = open_camera()
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "original"
    )
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    img_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == ord("c") or key == ord(" "):  # 按 'c'/空格 键拍照
            img_name = os.path.join(dataset_path, f"img_{img_count:03d}.png")
            cv2.imwrite(img_name, frame)
            print(f"Captured {img_name}")
            img_count += 1
        elif key == ord("q"):  # 按 'q' 键退出
            print("Exiting...")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
