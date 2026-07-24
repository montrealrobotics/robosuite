"""
Gripper for UFactory's XArm6 (has two fingers).

UFactory ships the same parallel gripper across the xArm family, so this reuses the XArm7 gripper
model verbatim and exists only so that `XArm6.default_gripper` reads sensibly.
"""

from robosuite.models.grippers.xarm7_gripper import XArm7Gripper, XArm7GripperBase


class XArm6GripperBase(XArm7GripperBase):
    """
    Gripper for UFactory's XArm6.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """


class XArm6Gripper(XArm7Gripper):
    """
    Modifies XArm6 Gripper to only take one action.
    """
