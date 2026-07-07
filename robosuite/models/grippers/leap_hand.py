"""
LEAP dexterous hand gripper for robosuite.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class LEAPRightHand(GripperModel):
    """
    LEAP dexterous right hand (3 fingers + thumb, 16 DOF).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/leap_hand_right.xml"), idn=idn)

    def format_action(self, action):
        assert len(action) == self.dof
        return np.array(action)

    @property
    def init_qpos(self):
        return np.array([0.0] * 16)

    @property
    def grasp_qpos(self):
        return {
            -1: np.array([0.0] * 16),  # open
            1: np.array(
                [
                    1.5,
                    0.0,
                    1.5,
                    1.5,  # index: mcp, rot, pip, dip
                    1.5,
                    0.0,
                    1.5,
                    1.5,  # middle: mcp, rot, pip, dip
                    1.5,
                    0.0,
                    1.5,
                    1.5,  # ring: mcp, rot, pip, dip
                    1.0,
                    1.0,
                    1.5,
                    1.0,
                ]
            ),  # thumb: cmc, axl, mcp, ipl
        }

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 16

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "th_mp_collision",
                "th_bs_collision_1",
                "th_bs_collision_2",
                "th_bs_collision_3",
                "th_px_collision_1",
                "th_px_collision_2",
                "th_px_collision_3",
                "th_px_collision_4",
                "th_px_collision_5",
                "th_ds_collision_1",
                "th_ds_collision_2",
                "th_ds_collision_3",
            ],
            "right_finger": [
                "if_bs_collision_1",
                "if_bs_collision_2",
                "if_px_collision",
                "if_md_collision_5",
                "if_ds_collision_1",
                "mf_bs_collision_1",
                "mf_bs_collision_2",
                "mf_px_collision",
                "mf_md_collision_5",
                "mf_ds_collision_1",
                "rf_bs_collision_1",
                "rf_bs_collision_2",
                "rf_px_collision",
                "rf_md_collision_5",
                "rf_ds_collision_1",
            ],
            "left_fingerpad": [
                "th_ds_collision_1",
                "th_ds_collision_2",
            ],
            "right_fingerpad": [
                "if_ds_collision_1",
                "mf_ds_collision_1",
                "rf_ds_collision_1",
            ],
        }
