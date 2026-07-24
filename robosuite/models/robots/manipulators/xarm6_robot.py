import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class XArm6(ManipulatorModel):
    """
    XArm6 is a single-arm 6 DOF robot designed by UFactory.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/xarm6/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "XArm6Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_xarm6"}

    @property
    def init_qpos(self):
        # solved so the flange lands on the same pose the Panda's does (0.437, 0, 0.186 relative to
        # the base, pointing straight down), which keeps task setups tuned for the Panda usable
        return np.array([0.0, 0.2204, -0.778, 0.0, 0.5576, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
