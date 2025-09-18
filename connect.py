# Description: 控制机械臂demo
import sys
import os
from pynput import keyboard

sys.path.append(os.path.join(os.path.dirname(__file__), "lerobot/src/"))
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.model.kinematics import RobotKinematics
import tqdm
import time
import json
import threading

# kinematics = RobotKinematics("low-cost-arm.urdf")
# pose = kinematics.forward_kinematics([0, 0, 0, 0, 0])
# print(pose)

leader_left_port = "COM14"
follower_left_port = "COM15"

norm_mode_body = MotorNormMode.DEGREES
leader_arm_left = DynamixelMotorsBus(
    port=leader_left_port,
    motors={
        "shoulder_pan": Motor(1, "xl330-m077", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "xl330-m077", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "xl330-m077", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "xl330-m077", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "xl330-m077", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
    },
)

follower_arm_left = DynamixelMotorsBus(
    port=follower_left_port,
    motors={
        "shoulder_pan": Motor(1, "xl430-w250", norm_mode_body),
        "shoulder_lift": Motor(2, "xl430-w250", norm_mode_body),
        "elbow_flex": Motor(3, "xl330-m288", norm_mode_body),
        "wrist_flex": Motor(4, "xl330-m288", norm_mode_body),
        "wrist_roll": Motor(5, "xl330-m288", norm_mode_body),
        "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
    },
)

# 确保连接成功
# leader_arm_left.connect()
follower_arm_left.connect()

# leader_arm_left.disable_torque()
follower_arm_left.disable_torque()

# calibration_path_leader = r"calibration/leader.json"
# calibration_data_leader = json.load(open(calibration_path_leader, "r"))
# for motor_name, calib in calibration_data_leader.items():
#     calibration_data_leader[motor_name] = MotorCalibration(**calib)
# leader_arm_left.write_calibration(calibration_dict=calibration_data_leader)

calibration_path_follower = r"calibration/follower.json"
calibration_data_follower = json.load(open(calibration_path_follower, "r"))
for motor_name, calib in calibration_data_follower.items():
    calibration_data_follower[motor_name] = MotorCalibration(**calib)
follower_arm_left.write_calibration(calibration_dict=calibration_data_follower)

# 运行操作
seconds = 3000
frequency = 10

follower_arm_left.enable_torque()
# follower_arm_left.set_half_turn_homings()
for idx, motor in enumerate(follower_arm_left.motors.keys()):
    follower_arm_left.write("Goal_Position", motor, 0)

pos = 0


def on_press(key):
    global pos
    if key == keyboard.Key.left:
        pos += 5
    elif key == keyboard.Key.right:
        pos -= 5


def on_release(key):
    pass


def get_input():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


input_thread = threading.Thread(target=get_input)
input_thread.daemon = True
input_thread.start()

for _ in range(seconds * frequency):
    # leader_pos_left = {}
    # for idx, motor in enumerate(leader_arm_left.motors.keys()):
    #     try:
    #         leader_pos_left[idx] = leader_arm_left.read("Present_Position", motor=motor)
    #     except Exception as e:
    #         pass

    motors = list(follower_arm_left.motors.keys())
    # # 将 leader 的位置发送到 follower
    # for idx, motor in enumerate(motors):
    #     if idx not in leader_pos_left:
    #         continue
    #     try:
    #         follower_arm_left.write("Goal_Position", motor, leader_pos_left[idx])
    #     except Exception as e:
    #         pass

    try:
        follower_arm_left.write("Goal_Position", motor=motors[3], value=pos)
    except Exception as e:
        pass

    follower_pos_left = {}
    for motor in follower_arm_left.motors.keys():
        try:
            follower_pos_left[motor] = (
                f'{follower_arm_left.read("Present_Position", motor=motor):.2f}'
            )
        except Exception as e:
            follower_pos_left[motor] = "Error"
    print("\r", list(follower_pos_left.values()), end="")

    # 确保不会在过高频率下运行，可以加入一些延迟
    time.sleep(1 / frequency)
