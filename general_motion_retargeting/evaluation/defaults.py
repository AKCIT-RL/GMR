"""Default end-effector body names per robot key (MuJoCo body names)."""

DEFAULT_EE_BODIES: dict[str, list[str]] = {
    "unitree_g1": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "unitree_g1_with_hands": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "booster_t1": ["left_foot_link", "right_foot_link"],
    "booster_t1_29dof": ["left_foot_link", "right_foot_link"],
}


def ee_bodies_for_robot(robot: str) -> list[str] | None:
    return DEFAULT_EE_BODIES.get(robot)
