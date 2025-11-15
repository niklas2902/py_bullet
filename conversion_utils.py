import numpy as np


def quaternion_to_euler(p, quat):
    """Convert quaternion to Euler angles (returns z-rotation for 2D case)"""
    euler = p.getEulerFromQuaternion(quat)
    return euler[2]  # Return z-axis rotation


def quaternion_to_rotation_matrix(p, quat):
    """Convert quaternion to 3x3 rotation matrix"""
    x, y, z, w = quat

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ])

    return R