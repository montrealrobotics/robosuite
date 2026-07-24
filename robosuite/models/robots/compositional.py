import numpy as np

from robosuite.models.robots import *


class PandaOmron(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}

    @property
    def init_qpos(self):
        return np.array([0, np.pi / 16.0 - 0.2, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.4, np.pi / 4])

    @property
    def init_torso_qpos(self):
        return np.array([0.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }


class SpotWithArm(SpotArm):
    @property
    def default_base(self):
        return "Spot"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-1.05, -0.1, -0.22),
            "empty": (-1.1, 0, -0.22),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, -0.22),
        }


class SpotWithArmFloating(SpotArm):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.7, -0.1, 0.0),
            "empty": (-0.6, 0, 0.0),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, 0.0),
        }


class PandaDexRH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}


class PandaDexLH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireLeftHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.5, -0.5, 0.5, -0.5]}


class PandaDexLeapRH(Panda):
    @property
    def default_gripper(self):
        return {"right": "LEAPRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.06, -0.02, 0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0, 1, 0, 0]}


class XArm6DexLeapRH(XArm6):
    """XArm6 arm + LEAP right hand (the lab's Franka-replacement setup)."""

    @property
    def default_gripper(self):
        return {"right": "LEAPRightHand"}

    # These reproduce the exact palm pose relative to the flange that PandaDexLeapRH uses, expressed
    # in the xArm tool frame (which is the Panda's right_hand frame rotated by 135 deg about z).
    # Retune both if the physical mounting bracket differs.
    @property
    def gripper_mount_pos_offset(self):
        return {"right": [-0.028256, 0.056582, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.0, 0.382691, 0.923785, 0.0]}


class XArm6DexLeapRHOmron(XArm6DexLeapRH):
    """XArm6 arm + LEAP right hand on Omron wheeled base (for Robocasa kitchen tasks)."""

    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "XArm6"}

    @property
    def init_qpos(self):
        # same idea as XArm6.init_qpos, re-solved so the flange matches PandaDexLeapRHOmron's once
        # the torso lift is accounted for
        return np.array([0.0, 0.0767, -0.7906, 0.0, 0.5933, 0.0])

    @property
    def init_torso_qpos(self):
        return np.array([0.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def default_controller_config(self):
        return {
            "right": "default_xarm6",
            "base": "joint_vel",
            "torso": "joint_pos",
        }


class PandaDexLeapRHOmron(Panda):
    """Panda arm + LEAP right hand on Omron wheeled base (for Robocasa kitchen tasks)."""

    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_gripper(self):
        return {"right": "LEAPRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.06, -0.02, 0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0, 1, 0, 0]}

    @property
    def default_arms(self):
        return {"right": "Panda"}

    @property
    def init_qpos(self):
        return np.array([0, np.pi / 16.0 - 0.2, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.4, np.pi / 4])

    @property
    def init_torso_qpos(self):
        return np.array([0.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def default_controller_config(self):
        return {
            "right": "default_panda",
            "base": "joint_vel",
            "torso": "joint_pos",
        }
