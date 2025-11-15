from typing import Any

import numpy as np

from conversion_utils import quaternion_to_euler, quaternion_to_rotation_matrix


def create_transform_data(p, pos, quat, scale):
    """Create transform dictionary with origin and basis"""
    rot_matrix = quaternion_to_rotation_matrix(p, quat)

    # Scale the basis vectors
    scaled_basis = rot_matrix * np.array(scale)

    return {
        "origin": {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2])
        },
        "basis": {
            "x": {
                "x": float(scaled_basis[0, 0]),
                "y": float(scaled_basis[1, 0]),
                "z": float(scaled_basis[2, 0])
            },
            "y": {
                "x": float(scaled_basis[0, 1]),
                "y": float(scaled_basis[1, 1]),
                "z": float(scaled_basis[2, 1])
            },
            "z": {
                "x": float(scaled_basis[0, 2]),
                "y": float(scaled_basis[1, 2]),
                "z": float(scaled_basis[2, 2])
            }
        }
    }


def record_collision(p, collision_data: list[Any], contact_points, frame: int, plane_id,
                     prev_angular_vel: list[int] | Any,
                     prev_linear_vel: list[int] | Any, sphere_id):
    # Get current state after collision
    pos, quat = p.getBasePositionAndOrientation(sphere_id)
    linear_vel, angular_vel = p.getBaseVelocity(sphere_id)

    total_impulse = [0, 0, 0]
    total_angular_impulse = [0, 0, 0]

    # Extract impulse from contact points
    for contact in contact_points:
        # contact[9] is the normal force (impulse magnitude)
        # contact[7] is the contact normal on B
        normal_impulse = contact[9]
        contact_normal = contact[7]

        # Add the impulse (force * dt, but PyBullet gives us the impulse directly)
        for i in range(3):
            total_impulse[i] += normal_impulse * contact_normal[i]

        # For angular impulse, we need the contact position and the impulse
        contact_pos_on_self = contact[5]  # Position on bodyA (sphere)

        # Calculate torque = r × F (cross product of position and force)
        r = [contact_pos_on_self[i] - pos[i] for i in range(3)]
        torque = [
            r[1] * (normal_impulse * contact_normal[2]) - r[2] * (normal_impulse * contact_normal[1]),
            r[2] * (normal_impulse * contact_normal[0]) - r[0] * (normal_impulse * contact_normal[2]),
            r[0] * (normal_impulse * contact_normal[1]) - r[1] * (normal_impulse * contact_normal[0])
        ]

        for i in range(3):
            total_angular_impulse[i] += torque[i]

    # Get plane position and orientation
    plane_pos, plane_quat = p.getBasePositionAndOrientation(plane_id)

    collision_entry = {
        "frame": frame,
        "impulse": {
            "x": float(total_impulse[0]),
            "y": float(total_impulse[1]),
            "z": float(total_impulse[2])
        },
        "angular_impulse": {
            "x": float(total_angular_impulse[0]),
            "y": float(total_angular_impulse[1]),
            "z": float(total_angular_impulse[2])
        },
        "pre_collision_linear_velocity": {
            "x": float(prev_linear_vel[0]),
            "y": float(prev_linear_vel[1]),
            "z": float(prev_linear_vel[2])
        },
        "pre_collision_angular_velocity": {
            "x": float(prev_angular_vel[0]),
            "y": float(prev_angular_vel[1]),
            "z": float(prev_angular_vel[2])
        },
        "linear_velocity": {
            "x": float(linear_vel[0]),
            "y": float(linear_vel[1]),
            "z": float(linear_vel[2])
        },
        "angular_velocity": {
            "x": float(angular_vel[0]),
            "y": float(angular_vel[1]),
            "z": float(angular_vel[2])
        },
        "self_mesh": None,
        "collider_mesh": None,
        "collider_name": "plane",
        "collider_id": plane_id,
        "collider_shape_index": None,
        "self_position": {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2])
        },
        "self_rotation": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quaternion_to_euler(p, quat))
        },
        "self_scale": {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0
        },
        "self_transform": create_transform_data(p, pos, quat, [1.0, 1.0, 1.0]),
        "collider_position": {
            "x": float(plane_pos[0]),
            "y": float(plane_pos[1]),
            "z": float(plane_pos[2])
        },
        "collider_rotation": {
            "x": float(plane_quat[0]),
            "y": float(plane_quat[1]),
            "z": float(quaternion_to_euler(p, plane_quat))
        },
        "collider_scale": {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0
        },
        "collider_transform": create_transform_data(p, plane_pos, plane_quat, [1.0, 1.0, 1.0])
    }

    collision_data.append(collision_entry)