import numpy as np
from pathlib import Path
import re

from fk_sync.config import load_skeleton_configs
from fk_sync.kinematics import KinematicsModel


PATH_ROOT = Path(__file__).parent.parent.parent
print("Path root:", PATH_ROOT)

path = PATH_ROOT / "config" / "robots.yaml"
path_urdf = PATH_ROOT / "config" / "g1_29dof_with_hand_rev_1_0.urdf"

skeletons = load_skeleton_configs(path)
skel = skeletons["g1"]

print("Skeleton:", skel.name)
print("Num joints:", len(skel.joint_order))
print("First joints:", skel.joint_order[:5])

kin = KinematicsModel.from_urdf(
    urdf_path=path_urdf,
    joint_order=skel.joint_order,
    root_frame_name=None,
)

q0 = np.zeros(len(skel.joint_order), dtype=float)

end_effectors = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_hand_palm_link",
    "right_hand_palm_link",
]


def run_and_print(tag, q):
    fk_res = kin.fk(q, end_effectors, frame="world")
    print(f"\n=== {tag} ===")
    for ln, (pos, quat) in fk_res.items():
        print(f"\nEnd-effector: {ln}")
        print("  pos:", pos)
        print("  quat:", quat)


def set_joint(q_vec, joint_name, value):
    try:
        idx = skel.joint_order.index(joint_name)
    except ValueError:
        # suggerisci nomi simili
        pat = re.compile(
            joint_name.replace("left_", "").replace("right_", ""), re.IGNORECASE
        )
        candidates = [j for j in skel.joint_order if pat.search(j)]
        raise ValueError(
            f"Joint '{joint_name}' non trovato in skel.joint_order.\n"
            f"Possibili candidati:\n  - " + "\n  - ".join(candidates[:20])
        )
    q_vec[idx] = value


def quat_dist_sign_invariant(q1, q2):
    q1 = np.asarray(q1, dtype=float).reshape(4)
    q2 = np.asarray(q2, dtype=float).reshape(4)
    return min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))


# Baseline
run_and_print("baseline (all zeros)", q0)

# Test 1: piega SOLO la gamba sinistra (cambia POS)
q1 = q0.copy()
set_joint(q1, "left_knee_joint", 0.7)
run_and_print("left_knee_joint = 0.7 rad", q1)

# Test 2: ruota SOLO la caviglia sinistra roll (cambia QUAT)
q2 = q0.copy()
set_joint(q2, "left_ankle_roll_joint", 0.5)
run_and_print("left_ankle_roll_joint = 0.5 rad", q2)

# Test 3: yaw anca sinistra (dovrebbe muovere/ruotare la chain)
q3 = q0.copy()
set_joint(q3, "left_hip_yaw_joint", 0.4)
run_and_print("left_hip_yaw_joint = 0.4 rad", q3)

# Test braccio: dovrebbe cambiare almeno left_hand_palm_link
print("\n=== left_shoulder_pitch_joint = 0.5 rad ===")
q4 = q0.copy()
set_joint(q4, "left_shoulder_pitch_joint", 0.5)
fk_res = kin.fk(q4, ["left_hand_palm_link", "right_hand_palm_link"], frame="world")
for ln, (pos, quat) in fk_res.items():
    print(f"\nEnd-effector: {ln}")
    print("  pos:", pos)
    print("  quat:", quat)

# Test mapping: q(29) vs q_full(43)
print("\n=== same pose with q(29) vs q_full(43) ===")
q29 = q0.copy()
set_joint(q29, "left_knee_joint", 0.7)

res29 = kin.fk(q29, ["left_ankle_roll_link"], frame="world")["left_ankle_roll_link"]

qfull = kin._expand_q(q29)  # ok anche se "privata": la stai usando solo nel test
res43 = kin.fk(qfull, ["left_ankle_roll_link"], frame="world")["left_ankle_roll_link"]

print("pos diff:", np.linalg.norm(res29[0] - res43[0]))
print("quat diff (sign-invariant):", quat_dist_sign_invariant(res29[1], res43[1]))
