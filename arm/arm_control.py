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


class Arm:
    def __init__(
        self,
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
        # 这个offset是用来修正机械臂零位的，目前不知道为什么舵机全零位置不是机械臂的零位
        # 所以每次重新标定或在新机械臂上需要重新测量这个offset
        # 方法是标定完后，把机械臂放到零位位置(见docs/image1.png)，然后读取各关节角度，作为offset保存下来
        # 一行一个，单位度，夹爪角度不需要
        self.offset = [
            float(x.strip())
            for x in open(
                os.path.join(calibration_dir, "arm_offset.txt"), "r", encoding="utf-8"
            ).readlines()
            if x.strip() != ""
        ]
        if len(self.offset) != 5:
            raise ValueError("机械臂offset文件格式错误，应该有5个值")
        self.arm = koch_follower.KochFollower(config)
        self.arm.connect()

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None,
        gripper_angle_deg: float | int | None,
    ):
        """
        设置机械臂角度和夹爪张开角度
        angles: 机械臂各关节角度列表，单位度，顺序从底座到末端执行器
        gripper_angle: 夹爪张开角度，范围0-100，越大越开
        """
        motor_names = list(self.arm.bus.motors.keys())
        action: dict[str, float] = {}
        if gripper_angle_deg is not None:
            action[motor_names[-1] + ".pos"] = gripper_angle_deg
        if angles_deg is not None:
            for motor_name, angle_deg, offset in zip(
                motor_names[:-1], angles_deg, self.offset, strict=True
            ):
                action[motor_name + ".pos"] = angle_deg + offset
        if len(action) > 0:
            self.arm.send_action(action)

    def get_read_arm_angles(
        self,
    ) -> tuple[Union[List[float], None], Union[float, None]]:
        """
        获取机械臂各关节角度和夹爪状态，单位度
        """
        try:
            angles_deg = list(self.arm.get_observation().values())
        except Exception as e:
            return None, None
        return [
            angle - offset
            for angle, offset in zip(angles_deg[:-1], self.offset, strict=True)
        ], angles_deg[-1]

    def disconnect_arm(self):
        self.arm.disconnect()

    def enable_torque(self):
        """
        连接后默认是enabled力矩的
        """
        self.arm.bus.enable_torque()

    def disable_torque(self):
        self.arm.bus.disable_torque()


if __name__ == "__main__":
    arm = Arm("COM3")
    time.sleep(1)
    arm.set_arm_angles([0, 0, 0, 0, 0], gripper_angle=80)
    time.sleep(1)
    angles, gripper = arm.get_read_arm_angles()
    print("机械臂角度:", angles)
    print("夹爪状态:", gripper)
    arm.set_arm_angles(None, gripper_angle=0)
    time.sleep(1)
    angles, gripper = arm.get_read_arm_angles()
    print("机械臂角度:", angles)
    print("夹爪状态:", gripper)
    arm.disconnect_arm()
