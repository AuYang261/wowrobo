import imp
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from camera.orb_camera import open_camera, get_frames, close_camera


def main():
    parser = argparse.ArgumentParser(description="Object Detection using YOLOv8")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "runs", "best.pt"),
        help="Path to the YOLOv8 model",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.7,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run the model on (e.g., 'cpu', '0', '0,1')",
    )
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Set device
    if args.device:
        model.to(args.device)

    # Open video capture
    cap = open_camera(True, False)

    while True:
        frames = get_frames(cap)
        if frames.get("color") is None:
            print("Failed to grab frame")
            continue

        start_time = time.time()

        # Perform inference
        results = model(frames["color"], conf=args.conf_thres, iou=args.iou_thres)[0]
        detections = results.obb.xywhr.cpu().numpy()  # xywhr format
        scores = results.obb.conf.cpu().numpy()
        class_ids = results.obb.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_ids]
        annotated_frame = frames["color"].copy()
        for (x, y, w, h, r), score, class_id, class_name in zip(
            detections, scores, class_ids, class_names
        ):
            box = cv2.boxPoints(((x, y), (w, h), np.rad2deg(r)))
            box = np.int0(box)
            cv2.drawContours(annotated_frame, [box], 0, (0, 255, 0), 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(
                annotated_frame,
                label,
                (int(x - w / 2), int(y - h / 2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break
    close_camera(cap)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
