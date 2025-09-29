# Description: 机械臂控制封装
from collections.abc import Sequence
import sys
import os
from unicodedata import digit
from pynput import keyboard
import time

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lerobot/src/")
)
from lerobot.robots.koch_follower import config_koch_follower, koch_follower
from pathlib import Path
from typing import Union, List


def connect_arm(
    port,
    calibration_dir=os.path.join(os.path.dirname(__file__), "..", "calibration"),
    id="koch_follower",
):
    config = config_koch_follower.KochFollowerConfig(
        port=port,
        disable_torque_on_disconnect=True,
        use_degrees=True,
        id=id,
        calibration_dir=Path(calibration_dir).resolve(),
    )
    arm = koch_follower.KochFollower(config)
    arm.connect()
    return arm


def set_arm_angles(
    arm: koch_follower.KochFollower,
    angles: Sequence[float | int] | None,
    gripper_angle: float | int | None,
):
    """
    设置机械臂角度和夹爪张开角度
    angles: 机械臂各关节角度列表，单位度，顺序从底座到末端执行器
    gripper_angle: 夹爪张开角度，范围0-100，越大越开
    """
    motor_names = list(arm.bus.motors.keys())
    action: dict[str, float] = {}
    if gripper_angle is not None:
        action[motor_names[-1] + ".pos"] = gripper_angle
    if angles is not None:
        for motor_name, angle in zip(motor_names[:-1], angles):
            action[motor_name + ".pos"] = angle
    if len(action) > 0:
        arm.send_action(action)


def get_read_arm_angles(
    arm: koch_follower.KochFollower,
) -> tuple[Union[List[float], None], Union[float, None]]:
    """
    获取机械臂各关节角度和夹爪状态
    """
    try:
        angles = list(arm.get_observation().values())
    except Exception as e:
        return None, None
    return angles[:-1], angles[-1]


def disconnect_arm(arm: koch_follower.KochFollower):
    arm.disconnect()


if __name__ == "__main__":
    arm = connect_arm("COM3")
    time.sleep(1)
    set_arm_angles(arm, [0, 0, 0, 0, 0], gripper_angle=80)
    time.sleep(1)
    angles, gripper = get_read_arm_angles(arm)
    print("机械臂角度:", angles)
    print("夹爪状态:", gripper)
    set_arm_angles(arm, None, gripper_angle=0)
    time.sleep(1)
    angles, gripper = get_read_arm_angles(arm)
    print("机械臂角度:", angles)
    print("夹爪状态:", gripper)
    disconnect_arm(arm)
