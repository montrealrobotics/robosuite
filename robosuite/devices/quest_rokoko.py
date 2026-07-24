"""
Driver class for combined Meta Quest 3 + Rokoko Glove teleoperation.

Receives wrist pose from Meta Quest 3 headset via UDP and fingertip
positions from Rokoko motion capture gloves via UDP. Translates these
into arm and dexterous-hand actions for a LEAP-hand robot. The arm is
auto-detected from the env (xArm or Franka/Panda); override with arm_type.

Based on the DexCap teleoperation system.
Retargeting Logic: Ported from DexCap (using PyBullet IK).
"""

import json
import os
import select
import socket
import threading
import time
from copy import deepcopy

import mujoco
import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation

import robosuite.utils.transform_utils as T
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix

# ── Default network configuration ──────────────────────────────────────────
# DEFAULT_VR_IP = "192.168.50.89"
DEFAULT_VR_IP = "192.168.50.66"
DEFAULT_LOCAL_IP = "192.168.50.178"
DEFAULT_POSE_CMD_PORT = 12346  # Quest wrist pose
DEFAULT_IK_RESULT_PORT = 12345  # Send IK results back to Quest for rendering
DEFAULT_ROKOKO_PORT = 14043  # Rokoko glove finger data
DEFAULT_HAND_INFO_PORT = 65432  # port to send tip visualisation to Quest

# ── DexCap Assets & Constants ──────────────────────────────────────────────
# Absolute path to the URDF file used in DexCap
DEXCAP_URDF_PATH = "/home/artur/dexcap_ws/src/DexCap/STEP3_inference/assets/leap_hand/robot_pybullet.urdf"

# Per-arm PyBullet IK configuration. Each entry uses the *same* URDF, end-effector link, and rest
# pose that the real DexCap teleop server solves against (configs/{arm}_arm.yaml), so sim and real
# run identical IK. Base orientation is DexCap's 90° Z convention.
ARM_CONFIGS = {
    "xarm": {
        "urdf": "/home/artur/dexcap_ws/src/DexCap/STEP3_inference/assets/xarm_arm/xarm6_robot.urdf",
        "ee_index": 7,  # xarm_grasptarget_vis
        "num_joints": 6,
        "rest_pose": [0.0, -1.92, -0.39460, 0.0, 1.51, -0.00435],
        "base_ori": [0, 0, 0.7071068, 0.7071068],
    },
    "franka": {
        "urdf": "/home/artur/dexcap_ws/src/DexCap/STEP3_inference/assets/franka_arm/panda_leap.urdf",
        "ee_index": 9,  # panda_grasptarget
        "num_joints": 7,
        "rest_pose": [
            0.4,
            -0.49826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.00396583422025,
            1.1980939705504309,
        ],
        "base_ori": [0, 0, 0.7071068, 0.7071068],
    },
}


def _detect_arm_type(robot_model):
    """Infer which DexCap arm config to use from the robosuite robot model class name, or None."""
    name = type(robot_model).__name__.lower()
    if "xarm" in name:
        return "xarm"
    if "panda" in name or "franka" in name:
        return "franka"
    return None


# DexCap wrist offsets (from QuestRightArmLeapModule)
# Applied in solve_system_world before IK
RIGHT_HAND_POS_OFFSET = np.array([0.0, 0.0, -0.0])
RIGHT_HAND_ORN_OFFSET = Rotation.from_euler("xyz", [-np.pi, 0.0, 0.0])
RIGHT_PALM_ORN_OFFSET = np.array([-0.1, -0.05, 0.05, 0.0, 0.0, -np.pi / 2])
RIGHT_HAND_MOUNT_OFFSET = [0.05, -0.05, 0.1]  # LEAP hand mount offset on EE

# From DexCap QuestRightArmLeapModule
RIGHT_HAND_Q = [
    np.pi / 6,
    -np.pi / 6,
    np.pi / 3,
    np.pi / 6,
    np.pi / 6,
    0.0,
    np.pi / 3,
    np.pi / 6,
    np.pi / 6,
    np.pi / 6,
    np.pi / 3,
    np.pi / 6,
    np.pi / 6,
    np.pi / 6,
    np.pi / 3,
    np.pi / 6,
]

# PyBullet Link Indices for Fingertips (Index, Middle, Ring, Thumb)
# Derived from QuestRightArmLeapModule.fingertip_idx = [4, 9, 14, 19]
# Note: DexCap ordering seems to be Index, Middle, Ring, Thumb (based on URDF usually)
# We will verify ordering in _solve_finger_ik
FINGERTIP_IDX = [4, 9, 14, 19]


# ── Rokoko Constants ───────────────────────────────────────────────────────
HAND_LINK_NAMES = [
    "Hand",
    "ThumbProximal",
    "ThumbMedial",
    "ThumbDistal",
    "ThumbTip",
    "IndexProximal",
    "IndexMedial",
    "IndexDistal",
    "IndexTip",
    "MiddleProximal",
    "MiddleMedial",
    "MiddleDistal",
    "MiddleTip",
    "RingProximal",
    "RingMedial",
    "RingDistal",
    "RingTip",
    "LittleProximal",
    "LittleMedial",
    "LittleDistal",
    "LittleTip",
]

# Bone‑chain adjacency used by adjust_bone_length (Copied from DexCap RokokoModule)
_FIRST_KPS = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
_SECOND_KPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
_BONE_SCALE = [1, 1, 0.7, 0.7, 1, 0.7, 0.7, 0.7, 1, 0.7, 0.7, 0.7, 1, 0.7, 0.7, 0.7, 1, 0.7, 0.7, 0.7]

# Tip IDs from DexCap RokokoModule (using first row of tip_id matrix)
# [4, 8, 12, 16, 20] -> Thumb, Index, Middle, Ring, Little
# We only use [4, 8, 12, 16] (Thumb, Index, Middle, Ring)
_TIP_IDS = [4, 8, 12, 16]


def _adjust_bone_length(positions, factors=None):
    """Scale bone-segment lengths (ported from DexCap RokokoModule)."""
    if factors is None:
        factors = np.array(_BONE_SCALE) * 1.3
    else:
        factors = np.array(factors) * 1.3

    if positions.shape[0] == 21:
        diff = positions[_SECOND_KPS] - positions[_FIRST_KPS]
        diff = diff * factors.reshape((-1, 1))
        diff = diff.reshape((5, 4, 3))
        new_positions = diff.cumsum(axis=1).reshape(-1, 3)
        positions = np.vstack([positions[0], new_positions])
    return positions


class QuestRokoko(Device):
    """
    Combined Meta Quest 3 + Rokoko glove teleoperation device.
    Uses PyBullet for Finger IK (DexCap Port).
    """

    def __init__(
        self,
        env,
        vr_ip=DEFAULT_VR_IP,
        local_ip=DEFAULT_LOCAL_IP,
        pose_cmd_port=DEFAULT_POSE_CMD_PORT,
        ik_result_port=DEFAULT_IK_RESULT_PORT,
        rokoko_port=DEFAULT_ROKOKO_PORT,
        hand_info_port=DEFAULT_HAND_INFO_PORT,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        arm_type=None,
    ):
        super().__init__(env)

        self.vr_ip = vr_ip
        self.local_ip = local_ip
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # ── Resolve which arm to solve IK for ─────────────────────────────
        # Default: auto-detect from the env's robot so sim uses the matching DexCap URDF/rest pose.
        # Pass arm_type="xarm"|"franka" to override.
        if arm_type is None:
            arm_type = _detect_arm_type(env.robots[0].robot_model)
        if arm_type not in ARM_CONFIGS:
            raise ValueError(
                f"Could not resolve arm_type (got {arm_type!r}). " f"Pass arm_type=one of {list(ARM_CONFIGS)}."
            )
        self.arm_type = arm_type
        arm_cfg = ARM_CONFIGS[arm_type]
        self._arm_urdf = arm_cfg["urdf"]
        self._arm_ee_index = arm_cfg["ee_index"]
        self._arm_num_joints = arm_cfg["num_joints"]
        self._arm_rest_pose = arm_cfg["rest_pose"]
        self._arm_base_ori = arm_cfg["base_ori"]

        # ── Initialize PyBullet for IK ────────────────────────────────────
        self._pb_client = pb.connect(pb.DIRECT)  # Headless mode

        # Load LEAP Hand URDF (finger IK)
        if not os.path.exists(DEXCAP_URDF_PATH):
            print(f"[QuestRokoko] WARNING: PyBullet URDF not found at {DEXCAP_URDF_PATH}. Finger IK will fail.")
            self._pb_hand = None
        else:
            self._pb_hand = pb.loadURDF(DEXCAP_URDF_PATH, useFixedBase=True, physicsClientId=self._pb_client)
            self._set_pb_joint_positions(self._pb_hand, RIGHT_HAND_Q)
            self._pb_lower, self._pb_upper, self._pb_ranges = self._get_pb_joint_limits(self._pb_hand)

        # Load arm URDF (arm IK, same solver + URDF as the real DexCap teleop server)
        if os.path.exists(self._arm_urdf):
            self._pb_arm = pb.loadURDF(
                self._arm_urdf,
                basePosition=[0, 0, 0],
                baseOrientation=self._arm_base_ori,
                useFixedBase=True,
                physicsClientId=self._pb_client,
            )
            self._set_pb_joint_positions(self._pb_arm, self._arm_rest_pose)
            self._pb_arm_lower, self._pb_arm_upper, self._pb_arm_ranges = self._get_pb_joint_limits(self._pb_arm)
            print(f"[QuestRokoko] Loaded {self.arm_type} arm for PyBullet IK")
        else:
            print(f"[QuestRokoko] WARNING: arm URDF not found at {self._arm_urdf}")
            self._pb_arm = None

        # ── Quest & Rokoko Sockets ─────────────────────────────────────────
        self._quest_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._quest_sock.bind(("", pose_cmd_port))
        self._quest_sock.setblocking(0)
        self._quest_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)

        self._rokoko_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rokoko_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0)
        self._rokoko_sock.bind(("", rokoko_port))
        self._rokoko_sock.setblocking(0)

        self._ik_result_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ik_result_dest = (vr_ip, ik_result_port)

        self._tip_vis_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tip_vis_dest = (vr_ip, hand_info_port)

        # ── Internal state ─────────────────────────────────────────────────
        self._world_frame = None
        self._reset_state = 0
        self._enabled = False
        self._recording = False

        self._last_arm_q = None
        self._last_hand_q = None

        self._wrist_pos = np.zeros(3)
        self._wrist_rot = np.eye(3)

        self._eef_origin_pos = None
        self._eef_origin_rot = None

        self._right_fingertip_positions = None  # Wrist-relative

        # Latest raw teleop signals (pre-IK), cached each input2action for optional logging.
        # These are the only quantities that cannot be reconstructed offline from sim states, so
        # capturing them lets us re-run retargeting/IK or train wrist-pose policies later.
        self._last_teleop_obs = None

        # ── Coordinate transform from Unity to MuJoCo ──────────────────────
        self._Q = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.0]])

        # ── Resolve MuJoCo Gripper IDs (for output mapping) ────────────────
        self._gripper_joint_ids = []
        robot = self.env.robots[0]
        arm = robot.arms[0]
        self._gripper_joint_ids = np.array(robot._ref_gripper_joint_pos_indexes[arm], dtype=int)

        self._display_controls()
        self._reset_internal_state()

        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._listener_loop, daemon=True)
        self._thread.start()

    # ── PyBullet Helpers ───────────────────────────────────────────────────
    def _set_pb_joint_positions(self, robot, joint_positions):
        jid = 0
        for i in range(len(joint_positions)):
            if pb.getJointInfo(robot, jid)[2] != pb.JOINT_FIXED:
                pb.resetJointState(robot, jid, joint_positions[i])
            else:
                jid += 1
                pb.resetJointState(robot, jid, joint_positions[i])
            jid += 1

    def _get_pb_joint_limits(self, robot):
        lower, upper, ranges = [], [], []
        for i in range(pb.getNumJoints(robot)):
            info = pb.getJointInfo(robot, i)
            if info[2] == pb.JOINT_FIXED:
                continue
            lower.append(info[8])
            upper.append(info[9])
            ranges.append(info[9] - info[8])
        return lower, upper, ranges

    # ── Display Controls ───────────────────────────────────────────────────
    @staticmethod
    def _display_controls():
        print("Controls: Quest (Move Arm), Rokoko (Move Fingers)")

    # ── State Management ───────────────────────────────────────────────────
    def _reset_internal_state(self):
        super()._reset_internal_state()
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)
        self.pos = np.zeros(3)
        self.last_pos = np.zeros(3)
        self._first_frame = True

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    # ── Packet Processing ──────────────────────────────────────────────────
    def _compute_rel_transform(self, pose):
        world_frame = self._world_frame.copy()
        world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
        pose[:3] = np.array([pose[0], pose[2], pose[1]])
        Q = self._Q
        rot_base = Rotation.from_quat(world_frame[3:]).as_matrix()
        rot = Rotation.from_quat(pose[3:]).as_matrix()
        rel_rot = Q @ (rot_base.T @ rot) @ Q.T
        rel_pos = Rotation.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
        return rel_pos, rel_rot

    def _unpack_rokoko(self, data_bytes):
        info = json.loads(data_bytes)
        body = info["scene"]["actors"][0]["body"]
        right_link_positions = []
        if "rightHand" in body:
            raw_wrist_orn = body["rightHand"]["rotation"]
            wrist_orn = Rotation.from_quat(
                [
                    raw_wrist_orn["x"],
                    raw_wrist_orn["y"],
                    raw_wrist_orn["z"],
                    raw_wrist_orn["w"],
                ]
            )
            raw_wrist_pos = body["rightHand"]["position"]
            wrist_position = np.array(
                [
                    raw_wrist_pos["x"],
                    raw_wrist_pos["y"],
                    raw_wrist_pos["z"],
                ]
            )
            for link_name in HAND_LINK_NAMES:
                full_name = "right" + link_name
                raw_pos = body[full_name]["position"]
                pos = np.array([raw_pos["x"], raw_pos["y"], raw_pos["z"]])
                rel_pos = wrist_orn.inv().apply(pos - wrist_position)
                right_link_positions.append(rel_pos)
            right_link_positions = -np.array(right_link_positions)
        else:
            right_link_positions = np.zeros((21, 3))

        right_link_positions = _adjust_bone_length(right_link_positions)
        return right_link_positions[_TIP_IDS]  # [Thumb, Index, Middle, Ring]

    def _listener_loop(self):
        while True:
            if not self._enabled:
                time.sleep(0.01)
                continue
            readable, _, _ = select.select([self._quest_sock, self._rokoko_sock], [], [], 0.01)
            for sock in readable:
                try:
                    if sock is self._quest_sock:
                        self._process_quest_packet()
                    elif sock is self._rokoko_sock:
                        self._process_rokoko_packet()
                except Exception:
                    pass

    def _process_quest_packet(self):
        data, _ = self._quest_sock.recvfrom(1024)
        data_string = data.decode()
        if data_string.startswith("WorldFrame"):
            vals = [float(v) for v in data_string[11:].split(",")]
            with self._lock:
                self._world_frame = np.array(vals)
            self._calibrate_eef_origin()
            return
        if data_string.startswith("Start"):
            with self._lock:
                self._recording = True
            return
        if data_string.startswith("Stop"):
            with self._lock:
                self._recording = False
            return
        if data_string.startswith("Remove"):
            with self._lock:
                self._reset_state = 1
                self._recording = False
            return
        if data_string.find("RHand") != -1 and self._world_frame is not None:
            vals = [float(v) for v in data_string[7:].split(",")]
            wrist_tf = np.array(vals[:7])
            rel_wrist_pos, rel_wrist_rot = self._compute_rel_transform(wrist_tf)
            with self._lock:
                self._wrist_pos = rel_wrist_pos
                self._wrist_rot = rel_wrist_rot

    def _process_rokoko_packet(self):
        data, _ = self._rokoko_sock.recvfrom(40000)
        right_tips = self._unpack_rokoko(data.decode())
        with self._lock:
            self._right_fingertip_positions = right_tips

    def _calibrate_eef_origin(self):
        """Called when WorldFrame signal arrives. Reset arm to rest."""
        if self._pb_arm is not None:
            self._set_pb_joint_positions(self._pb_arm, self._arm_rest_pose)

    # ── Device Interface ───────────────────────────────────────────────────
    def get_controller_state(self):
        with self._lock:
            wrist_pos = self._wrist_pos.copy()
            wrist_rot = self._wrist_rot.copy()
            reset = self._reset_state
            self._reset_state = 0

        return dict(
            wrist_pos=wrist_pos,
            wrist_rot=wrist_rot,
            reset=reset,
        )

    def _postprocess_device_outputs(self, dpos, drotation):
        dpos = dpos * 75
        drotation = drotation * 1.5
        return np.clip(dpos, -1, 1), np.clip(drotation, -1, 1)

    def input2action(self, mirror_actions=False, goal_update_mode="achieved"):
        """
        Absolute-pose-based teleoperation using PyBullet arm IK.

        With a JOINT_POSITION arm controller (the default, and what the real robot runs):
        1. Quest wrist pose (from _compute_rel_transform) → PyBullet arm IK → joint angles
        2. Joint angles are commanded directly

        With an OSC arm controller there is no joint-space interface, so the IK solution has to be
        pushed back through FK to a Cartesian target and tracked with a P-gain. That round trip is
        lossy and has no counterpart on hardware -- prefer JOINT_POSITION for anything you intend
        to transfer.
        """
        robot = self.env.robots[self.active_robot]
        active_arm = self.active_arm

        # ── Read Quest state ──
        state = self.get_controller_state()
        wrist_pos = state["wrist_pos"]
        wrist_rot = state["wrist_rot"]
        reset = state["reset"]

        if reset:
            # Reset PyBullet arm to rest pose
            if self._pb_arm is not None:
                self._set_pb_joint_positions(self._pb_arm, self._arm_rest_pose)
            return None

        # ── Wait for Quest calibration before moving ──
        if self._world_frame is None:
            ac_dict = {}
            for arm in robot.arms:
                arm_action = self._hold_arm_action(robot, arm)
                ac_dict[f"{arm}_abs"] = arm_action["abs"]
                ac_dict[f"{arm}_delta"] = arm_action["delta"]
                ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)
            return ac_dict

        # ── Solve arm IK in PyBullet ──
        #   wrist_pos/wrist_rot are in _compute_rel_transform frame (same as DexCap)
        pb_arm_q = self._solve_arm_ik_pb(wrist_pos, wrist_rot)

        # ── Build action dict ──
        ac_dict = {}
        for arm in robot.arms:
            if arm == active_arm:
                continue
            arm_action = self._hold_arm_action(robot, arm, goal_update_mode=goal_update_mode)
            ac_dict[f"{arm}_abs"] = arm_action["abs"]
            ac_dict[f"{arm}_delta"] = arm_action["delta"]
            ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)

        arm_action = self._arm_action_from_joint_target(robot, active_arm, pb_arm_q)
        ac_dict[f"{active_arm}_abs"] = arm_action["abs"]
        ac_dict[f"{active_arm}_delta"] = arm_action["delta"]
        ac_dict[f"{active_arm}_gripper"] = np.zeros(robot.gripper[active_arm].dof)

        # Diagnostic print
        self._diag_count = getattr(self, "_diag_count", 0) + 1
        if self._diag_count % 100 == 1:
            pb_ee = (
                pb.getLinkState(self._pb_arm, self._arm_ee_index, physicsClientId=self._pb_client)
                if self._pb_arm
                else None
            )
            pb_ee_pos = pb_ee[0] if pb_ee else (0, 0, 0)
            q_cur = self.env.sim.data.qpos[robot._ref_joint_pos_indexes[: self._arm_num_joints]]
            print(
                f"[ARM IK] wrist_pos=[{wrist_pos[0]:.3f},{wrist_pos[1]:.3f},{wrist_pos[2]:.3f}]  "
                f"pb_ee=[{pb_ee_pos[0]:.3f},{pb_ee_pos[1]:.3f},{pb_ee_pos[2]:.3f}]  "
                f"q_target={np.round(pb_arm_q[:self._arm_num_joints], 3)}  "
                f"q_current={np.round(q_cur, 3)}"
            )

        # ── Finger IK (PyBullet, DexCap world-frame approach) ──
        with self._lock:
            finger_tips = self._right_fingertip_positions
            has_rokoko = finger_tips is not None
            if has_rokoko:
                finger_tips = finger_tips.copy()

        if has_rokoko:
            finger_joints = self._solve_finger_ik_pb(finger_tips, wrist_pos, wrist_rot)
        else:
            finger_joints = self.env.sim.data.qpos[self._gripper_joint_ids].copy()

        ac_dict[f"{active_arm}_gripper"] = finger_joints

        # ── Send IK Result to Quest (using PyBullet arm solution) ──
        try:
            self._send_ik_result(pb_arm_q, finger_joints)
        except:
            pass

        # ── Clip deltas ──
        for k, v in ac_dict.items():
            if "abs" not in k and "gripper" not in k:
                ac_dict[k] = np.clip(v, -1, 1)

        # ── Cache raw teleop signals (pre-IK) for optional logging ──
        wrist_quat_xyzw = Rotation.from_matrix(wrist_rot).as_quat()
        self._last_teleop_obs = {
            "teleop_quest_wrist_pos": np.asarray(wrist_pos, dtype=np.float32),
            # store wxyz to match MuJoCo/robosuite quaternion convention
            "teleop_quest_wrist_quat": np.array(
                [wrist_quat_xyzw[3], wrist_quat_xyzw[0], wrist_quat_xyzw[1], wrist_quat_xyzw[2]],
                dtype=np.float32,
            ),
            "teleop_rokoko_fingertips": (
                np.asarray(finger_tips, dtype=np.float32)
                if has_rokoko
                else np.full((len(FINGERTIP_IDX), 3), np.nan, dtype=np.float32)
            ),
            "teleop_rokoko_valid": np.array([1.0 if has_rokoko else 0.0], dtype=np.float32),
        }

        return ac_dict

    def get_teleop_obs(self):
        """
        Return the most recent raw teleop signals (pre-IK) cached during input2action, or None if
        no step has been produced yet. Consumed by the data-collection wrapper to log signals that
        cannot be recovered from sim states offline.
        """
        return self._last_teleop_obs

    # ── Arm action construction ────────────────────────────────────────────

    def _arm_action_from_joint_target(self, robot, arm, joint_target):
        """
        Turn an IK joint solution into the arm's action.

        For a JOINT_POSITION controller the targets are the action, so they pass straight through --
        the same values the real robot receives. For an OSC controller there is no joint-space
        interface, so we FK the solution to a Cartesian target and track it with a P-gain.
        """
        controller = robot.part_controllers[arm]
        q_target = np.asarray(joint_target[: self._arm_num_joints], dtype=float)

        if controller.name == "JOINT_POSITION":
            q_current = self.env.sim.data.qpos[robot._ref_joint_pos_indexes[: self._arm_num_joints]]
            # delta mode expects a normalized action that scale_action maps back onto output range
            norm_delta = np.clip((q_target - q_current) / controller.output_max, -1.0, 1.0)
            return {"abs": q_target, "delta": norm_delta}

        # ── OSC fallback: joint targets → FK → Cartesian error → normalized delta ──
        from robosuite.utils.control_utils import orientation_error

        site_name = f"gripper0_{arm}_grip_site"
        current_pos = self.env.sim.data.get_site_xpos(site_name).copy()
        current_rot = self.env.sim.data.get_site_xmat(site_name).copy().reshape(3, 3)
        target_pos, target_rot = self._fk_to_mujoco_pose(q_target, robot, arm)

        pos_error = target_pos - current_pos
        rot_error = orientation_error(target_rot, current_rot)

        # If the OSC controller uses input_ref_frame="base", rotate the world-frame errors
        # into the robot's base frame.
        if getattr(controller, "input_ref_frame", "world") == "base" and controller.origin_ori is not None:
            R_base_inv = controller.origin_ori.T  # origin_ori is the 3x3 base rotation matrix
            pos_error = R_base_inv @ pos_error
            rot_error = R_base_inv @ rot_error

        POS_SCALE = 0.05  # matches OSC output_max for position
        ROT_SCALE = 0.5  # matches OSC output_max for rotation
        norm_delta = np.concatenate([np.clip(pos_error / POS_SCALE, -1, 1), np.clip(rot_error / ROT_SCALE, -1, 1)])
        return self.get_arm_action(robot, arm, norm_delta=norm_delta)

    def _hold_arm_action(self, robot, arm, goal_update_mode="achieved"):
        """Zero-motion action for a non-active arm, in whatever space its controller expects."""
        controller = robot.part_controllers[arm]
        if controller.name == "JOINT_POSITION":
            q_current = self.env.sim.data.qpos[robot._ref_joint_pos_indexes[: self._arm_num_joints]]
            return {"abs": np.asarray(q_current, dtype=float), "delta": np.zeros(self._arm_num_joints)}
        return self.get_arm_action(robot, arm, norm_delta=np.zeros(6), goal_update_mode=goal_update_mode)

    # ── PyBullet Arm IK ────────────────────────────────────────────────────

    def _solve_arm_ik_pb(self, wrist_pos, wrist_rot):
        """
        Solve xArm6 IK in PyBullet, matching DexCap's solve_system_world.

        Applies the same wrist offsets as DexCap before solving IK:
        1. right_hand_pos_offset (zero) and right_hand_orn_offset (-π X rotation)
        2. right_palm_orn_offset (position + euler wrist offset)
        """
        if self._pb_arm is None:
            return list(self._arm_rest_pose)

        # Match DexCap's solve_system_world call:
        #   solve_arm_ik(wrist_pos + wrist_orn.apply(right_hand_pos_offset),
        #               wrist_orn * right_hand_orn_offset.inv(),
        #               right_palm_orn_offset)
        wrist_orn = Rotation.from_matrix(wrist_rot)
        ik_pos = wrist_pos + wrist_orn.apply(RIGHT_HAND_POS_OFFSET)
        ik_orn = wrist_orn * RIGHT_HAND_ORN_OFFSET.inv()

        # Apply palm offset (same as DexCap's solve_arm_ik wrist_offset logic)
        ik_pos = ik_orn.apply(RIGHT_PALM_ORN_OFFSET[:3]) + ik_pos
        ik_orn = ik_orn * Rotation.from_euler("xyz", RIGHT_PALM_ORN_OFFSET[3:])

        target_q = pb.calculateInverseKinematics(
            self._pb_arm,
            self._arm_ee_index,
            ik_pos.tolist(),
            ik_orn.as_quat().tolist(),
            lowerLimits=self._pb_arm_lower,
            upperLimits=self._pb_arm_upper,
            jointRanges=self._pb_arm_ranges,
            restPoses=list(self._arm_rest_pose),
            maxNumIterations=40,
            residualThreshold=0.001,
            physicsClientId=self._pb_client,
        )
        arm_q = list(target_q[: self._arm_num_joints])
        self._set_pb_joint_positions(self._pb_arm, arm_q)
        return arm_q

    def _fk_to_mujoco_pose(self, joint_angles, robot, arm_name):
        """
        Forward-kinematic PyBullet arm joint angles through MuJoCo to get
        the EEF pose in MuJoCo world frame.

        Temporarily sets MuJoCo arm joints, runs forward kinematics,
        reads the EEF site pose, then restores original joint positions.
        """
        sim = self.env.sim
        # Save current arm joint positions
        arm_qpos_indexes = robot._ref_joint_pos_indexes[: self._arm_num_joints]
        saved_qpos = sim.data.qpos[arm_qpos_indexes].copy()
        saved_qvel = sim.data.qvel[arm_qpos_indexes].copy()

        # Set PyBullet IK solution
        for i, idx in enumerate(arm_qpos_indexes):
            sim.data.qpos[idx] = joint_angles[i]
            sim.data.qvel[idx] = 0.0

        # Run forward kinematics (updates site positions/orientations)
        mujoco.mj_fwdPosition(sim.model._model, sim.data._data)

        # Read EEF pose
        site_name = f"gripper0_{arm_name}_grip_site"
        target_pos = sim.data.get_site_xpos(site_name).copy()
        target_rot = sim.data.get_site_xmat(site_name).copy().reshape(3, 3)

        # Restore original joint positions
        for i, idx in enumerate(arm_qpos_indexes):
            sim.data.qpos[idx] = saved_qpos[i]
            sim.data.qvel[idx] = saved_qvel[i]

        # Restore forward kinematics for the original state
        mujoco.mj_fwdPosition(sim.model._model, sim.data._data)

        return target_pos, target_rot

    # ── PyBullet Finger IK Solver ─────────────────────────────────────────
    _ik_frame_count = 0  # Class-level debug counter

    def _solve_finger_ik_pb(self, tip_positions, wrist_pos, wrist_rot):
        """
        Solve fingertip IK matching DexCap's solve_system_world approach:
        1. Position LEAP hand at arm's EE (with mount offsets)
        2. Transform Rokoko tips to world frame
        3. Solve IK in world frame

        Args:
            tip_positions: (4, 3) wrist-relative, ordered [Thumb, Index, Middle, Ring]
            wrist_pos: wrist position from _compute_rel_transform
            wrist_rot: wrist rotation matrix from _compute_rel_transform
        """
        if self._pb_hand is None:
            return np.zeros(16)

        wrist_orn = Rotation.from_matrix(wrist_rot)

        # ── Position LEAP hand at arm's EE (matching DexCap) ──
        ee_state = pb.getLinkState(self._pb_arm, self._arm_ee_index, physicsClientId=self._pb_client)
        hand_xyz = np.array(ee_state[0])
        hand_orn = Rotation.from_quat(ee_state[1])
        mount_orn = hand_orn * RIGHT_HAND_ORN_OFFSET
        hand_base_pos = hand_xyz + mount_orn.apply(RIGHT_HAND_MOUNT_OFFSET)
        hand_base_quat = mount_orn.as_quat()
        pb.resetBasePositionAndOrientation(
            self._pb_hand,
            hand_base_pos.tolist(),
            hand_base_quat.tolist(),
            physicsClientId=self._pb_client,
        )

        # ── Transform Rokoko tips to world frame (DexCap convention) ──
        #   Reorder from [Thumb, Index, Middle, Ring] to [Index, Middle, Ring, Thumb]
        tips_reordered = tip_positions[[1, 2, 3, 0]]
        world_tips = wrist_orn.apply(tips_reordered) + wrist_pos

        # ── Solve IK for each finger ──
        target_q = []
        for i in range(4):
            q_sol = pb.calculateInverseKinematics(
                self._pb_hand,
                FINGERTIP_IDX[i],
                world_tips[i].tolist(),
                lowerLimits=self._pb_lower,
                upperLimits=self._pb_upper,
                jointRanges=self._pb_ranges,
                restPoses=RIGHT_HAND_Q,
                maxNumIterations=40,
                residualThreshold=0.001,
                physicsClientId=self._pb_client,
            )
            target_q.extend(q_sol[4 * i : 4 * (i + 1)])

        # ── Debug logging ──
        QuestRokoko._ik_frame_count += 1
        if QuestRokoko._ik_frame_count % 100 == 1:
            finger_names = ["Index", "Middle", "Ring", "Thumb"]
            print(f"\n[IK DEBUG] Frame {QuestRokoko._ik_frame_count}")
            print(f"  LEAP base pos=[{hand_base_pos[0]:.3f},{hand_base_pos[1]:.3f},{hand_base_pos[2]:.3f}]")
            for i in range(4):
                wt = world_tips[i]
                print(f"  {finger_names[i]:7s}  world=[{wt[0]:+.4f},{wt[1]:+.4f},{wt[2]:+.4f}]")
            print(f"  Joints: [{', '.join(f'{q:.2f}' for q in target_q)}]")

        return np.array(target_q)

    def _send_ik_result(self, arm_q, hand_q):
        delta_ok = self._check_delta(arm_q, self._last_arm_q) and self._check_delta(hand_q, self._last_hand_q, 0.2)
        status = "G" if not self._recording else ("Y" if delta_ok else "N")
        parts = [status] + [f"{q:.3f}" for q in arm_q] + [f"{q:.3f}" for q in hand_q]
        try:
            self._ik_result_sock.sendto(",".join(parts).encode(), self._ik_result_dest)
        except:
            pass
        self._last_arm_q, self._last_hand_q = np.array(arm_q), np.array(hand_q)

    @staticmethod
    def _check_delta(curr, prev, thres=0.1):
        return True if prev is None else np.all(np.abs(np.array(curr) - np.array(prev)) < thres)

    def close(self):
        self._enabled = False
        for s in [self._quest_sock, self._rokoko_sock, self._ik_result_sock, self._tip_vis_sock]:
            try:
                s.close()
            except:
                pass
        if self._pb_client is not None:
            pb.disconnect(self._pb_client)
