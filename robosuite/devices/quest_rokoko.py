"""
Driver class for combined Meta Quest 3 + Rokoko Glove teleoperation.

Receives wrist pose from Meta Quest 3 headset via UDP and fingertip
positions from Rokoko motion capture gloves via UDP. Translates these
into arm and dexterous-hand actions for the PandaDexLeapRH robot.

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
DEFAULT_VR_IP = "192.168.50.89"
DEFAULT_LOCAL_IP = "192.168.50.178"
DEFAULT_POSE_CMD_PORT = 12346  # Quest wrist pose
DEFAULT_IK_RESULT_PORT = 12345  # Send IK results back to Quest for rendering
DEFAULT_ROKOKO_PORT = 14043  # Rokoko glove finger data
DEFAULT_HAND_INFO_PORT = 65432  # port to send tip visualisation to Quest

# ── DexCap Assets & Constants ──────────────────────────────────────────────
# Absolute path to the URDF file used in DexCap
DEXCAP_URDF_PATH = "/home/artur/dexcap_ws/src/DexCap/STEP3_inference/assets/leap_hand/robot_pybullet.urdf"

# Panda arm URDF for PyBullet IK (must match DexCap's panda_leap.urdf for Unity vis)
PANDA_URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "assets",
    "bullet_data",
    "panda_description",
    "urdf",
    "panda_leap.urdf",
)
PANDA_EE_INDEX = 9  # panda_grasptarget (matches DexCap franka config)
PANDA_NUM_JOINTS = 7  # 7-DOF arm
# DexCap franka rest pose from configs/franka_arm.yaml
PANDA_REST_POSE = [
    0.4,
    -0.49826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.00396583422025,
    1.1980939705504309,
]
# Base orientation: 90° Z rotation (DexCap convention)
PANDA_BASE_ORI = [0, 0, 0.7071068, 0.7071068]

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
    ):
        super().__init__(env)

        self.vr_ip = vr_ip
        self.local_ip = local_ip
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

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

        # Load Panda arm URDF (arm IK, mirrors DexCap's xarm6 approach)
        if os.path.exists(PANDA_URDF_PATH):
            self._pb_arm = pb.loadURDF(
                PANDA_URDF_PATH,
                basePosition=[0, 0, 0],
                baseOrientation=PANDA_BASE_ORI,
                useFixedBase=True,
                physicsClientId=self._pb_client,
            )
            self._set_pb_joint_positions(self._pb_arm, PANDA_REST_POSE)
            self._pb_arm_lower, self._pb_arm_upper, self._pb_arm_ranges = self._get_pb_joint_limits(self._pb_arm)
            print(f"[QuestRokoko] Loaded Panda arm for PyBullet IK")
        else:
            print(f"[QuestRokoko] WARNING: Panda URDF not found at {PANDA_URDF_PATH}")
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
            self._set_pb_joint_positions(self._pb_arm, PANDA_REST_POSE)

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

        Pipeline:
        1. Quest wrist pose (from _compute_rel_transform) → PyBullet arm IK → joint angles
        2. Joint angles → MuJoCo FK → EEF pose in MuJoCo world frame
        3. EEF pose error → proportional control → OSC norm_delta

        This mirrors DexCap's approach and avoids coordinate frame conversion.
        """
        from robosuite.utils.control_utils import orientation_error

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
                self._set_pb_joint_positions(self._pb_arm, PANDA_REST_POSE)
            return None

        # ── Wait for Quest calibration before moving ──
        if self._world_frame is None:
            ac_dict = {}
            for arm in robot.arms:
                arm_action = self.get_arm_action(robot, arm, norm_delta=np.zeros(6))
                ac_dict[f"{arm}_abs"] = arm_action["abs"]
                ac_dict[f"{arm}_delta"] = arm_action["delta"]
                ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)
            return ac_dict

        # ── Current EEF pose from simulation ──
        site_name = f"gripper0_{active_arm}_grip_site"
        current_pos = self.env.sim.data.get_site_xpos(site_name).copy()
        current_rot = self.env.sim.data.get_site_xmat(site_name).copy().reshape(3, 3)

        # ── Solve arm IK in PyBullet ──
        #   wrist_pos/wrist_rot are in _compute_rel_transform frame (same as DexCap)
        pb_arm_q = self._solve_arm_ik_pb(wrist_pos, wrist_rot)

        # ── FK through MuJoCo to get target EEF pose in world frame ──
        target_pos, target_rot = self._fk_to_mujoco_pose(pb_arm_q, robot, active_arm)

        # ── Proportional control → normalized delta ──
        pos_error = target_pos - current_pos
        rot_error = orientation_error(target_rot, current_rot)

        # If the OSC controller uses input_ref_frame="base", we need to rotate
        # the world-frame errors into the robot's base frame.
        controller = robot.part_controllers[active_arm]
        if getattr(controller, "input_ref_frame", "world") == "base" and controller.origin_ori is not None:
            R_base_inv = controller.origin_ori.T  # origin_ori is the 3x3 base rotation matrix
            pos_error = R_base_inv @ pos_error
            rot_error = R_base_inv @ rot_error

        POS_SCALE = 0.05  # matches OSC output_max for position
        ROT_SCALE = 0.5  # matches OSC output_max for rotation
        norm_dpos = np.clip(pos_error / POS_SCALE, -1, 1)
        norm_drot = np.clip(rot_error / ROT_SCALE, -1, 1)
        arm_norm_delta = np.concatenate([norm_dpos, norm_drot])

        # Diagnostic print
        self._diag_count = getattr(self, "_diag_count", 0) + 1
        if self._diag_count % 100 == 1:
            # Also get PyBullet FK EE position for comparison
            pb_ee = (
                pb.getLinkState(self._pb_arm, PANDA_EE_INDEX, physicsClientId=self._pb_client) if self._pb_arm else None
            )
            pb_ee_pos = pb_ee[0] if pb_ee else (0, 0, 0)
            print(
                f"[ARM IK] wrist_pos=[{wrist_pos[0]:.3f},{wrist_pos[1]:.3f},{wrist_pos[2]:.3f}]  "
                f"pb_ee=[{pb_ee_pos[0]:.3f},{pb_ee_pos[1]:.3f},{pb_ee_pos[2]:.3f}]  "
                f"mj_target=[{target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}]  "
                f"mj_current=[{current_pos[0]:.3f},{current_pos[1]:.3f},{current_pos[2]:.3f}]"
            )

        # ── Build action dict ──
        ac_dict = {}
        for arm in robot.arms:
            arm_action = self.get_arm_action(
                robot,
                arm,
                norm_delta=np.zeros(6),
                goal_update_mode=goal_update_mode,
            )
            ac_dict[f"{arm}_abs"] = arm_action["abs"]
            ac_dict[f"{arm}_delta"] = arm_action["delta"]
            ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)

        arm_action = self.get_arm_action(
            robot,
            active_arm,
            norm_delta=arm_norm_delta,
        )
        ac_dict[f"{active_arm}_abs"] = arm_action["abs"]
        ac_dict[f"{active_arm}_delta"] = arm_action["delta"]

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

        return ac_dict

    # ── PyBullet Arm IK ────────────────────────────────────────────────────

    def _solve_arm_ik_pb(self, wrist_pos, wrist_rot):
        """
        Solve Panda arm IK in PyBullet, matching DexCap's solve_system_world.

        Applies the same wrist offsets as DexCap before solving IK:
        1. right_hand_pos_offset (zero) and right_hand_orn_offset (-π X rotation)
        2. right_palm_orn_offset (position + euler wrist offset)
        """
        if self._pb_arm is None:
            return list(PANDA_REST_POSE)

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
            PANDA_EE_INDEX,
            ik_pos.tolist(),
            ik_orn.as_quat().tolist(),
            lowerLimits=self._pb_arm_lower,
            upperLimits=self._pb_arm_upper,
            jointRanges=self._pb_arm_ranges,
            restPoses=list(PANDA_REST_POSE),
            maxNumIterations=40,
            residualThreshold=0.001,
            physicsClientId=self._pb_client,
        )
        arm_q = list(target_q[:PANDA_NUM_JOINTS])
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
        arm_qpos_indexes = robot._ref_joint_pos_indexes[:PANDA_NUM_JOINTS]
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
        ee_state = pb.getLinkState(self._pb_arm, PANDA_EE_INDEX, physicsClientId=self._pb_client)
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
