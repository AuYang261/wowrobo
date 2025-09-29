import math
import random
import numpy as np
import kinpy as kp

def rad(deg):
    return deg * math.pi / 180.0


def init_kp() -> kp.SerialChain:
    # 1. 读取 URDF
    urdf_path = "./config/low_cost_robot-feature.urdf"
    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_xml = f.read()

    # 2. 构建链条
    end_link_name = "gripper_moving_1"
    serial_chain = kp.build_serial_chain_from_urdf(urdf_xml, end_link_name)
    joint_names = serial_chain.get_joint_parameter_names()
    
    print("\n=== SerialChain joint names ===")
    print(joint_names)

    return serial_chain

def forward_kinematics_demo(serial_chain: kp.SerialChain, th_init):
    print("\n随机初始关节角度 (rad):", th_init)

    # 初始位置正向运动学
    init_tf = serial_chain.forward_kinematics(th_init, end_only=True)
    print("\n初始末端位姿:")
    print("pos =", init_tf.pos)
    print("rot =", init_tf.rot)

    return init_tf

def inverse_kinematics_demo(serial_chain: kp.SerialChain, target_tf, th_init):
    print("\n目标末端位姿:")
    print("pos =", target_tf.pos)
    print("rot =", target_tf.rot)

    # 逆运动学解算
    ik_solution = serial_chain.inverse_kinematics(target_tf, th_init)

    print("\n逆运动学解算结果:")
    print("Joint angles =", ik_solution)

    return ik_solution


def main():
    # 1. 读取 URDF
    urdf_path = "./config/low_cost_robot-feature.urdf"
    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_xml = f.read()

    # 2. 构建链条
    end_link_name = "gripper_moving_1"
    serial_chain = kp.build_serial_chain_from_urdf(urdf_xml, end_link_name)
    joint_names = serial_chain.get_joint_parameter_names()

    print("\n=== SerialChain joint names ===")
    print(joint_names)

    # 3. 随机生成初始角度 [-60°, 60°]
    th_init = [random.uniform(-rad(60), rad(60)) for _ in joint_names]
    print("\n随机初始关节角度 (rad):", th_init)

    # 初始位置正向运动学
    init_tf = serial_chain.forward_kinematics(th_init, end_only=True)
    print("\n初始末端位姿:")
    print("pos =", init_tf.pos)
    print("rot =", init_tf.rot)

    # 4. 生成目标位置：再随机一个关节角度
    th_goal = [random.uniform(-rad(60), rad(60)) for _ in joint_names]
    goal_tf = serial_chain.forward_kinematics(th_goal, end_only=True)
    print("\n目标末端位姿:")
    print("pos =", goal_tf.pos)
    print("rot =", goal_tf.rot)

    # 5. 逆运动学：从目标位姿 IK 解回初始角度
    print("\n=== 逆运动学过程 ===")
    steps = 10
    for alpha in np.linspace(0, 1, steps):
        # 插值末端位姿 (只插值位置，旋转保持目标值简化处理)
        interp_pos = init_tf.pos * (1 - alpha) + goal_tf.pos * alpha
        interp_tf = kp.Transform(pos=interp_pos, rot=goal_tf.rot)

        # 逆运动学解算
        ik_solution = serial_chain.inverse_kinematics(interp_tf, th_init)

        # 正向验证
        check_tf = serial_chain.forward_kinematics(ik_solution, end_only=True)

        print(f"\nStep {alpha:.2f}:")
        print("  Joint angles =", ik_solution)
        print("  End-effector pos =", check_tf.pos)

    print("\n=== 最终末端位姿 (IK 结果) ===")
    print("pos =", check_tf.pos)
    print("rot =", check_tf.rot)

if __name__ == "__main__":
    main()
