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

def to_vector(list_vector):
    output = {}

    for index, element in enumerate(["x", "y", "z"]):
        output[element] = list_vector[index]
    return output


def record_collision(p, collision_data: list[Any], collision_points: list[Any], contact_points, frame: int, plane_id,
                     prev_angular_vel: list[int] | Any,
                     current_linear_vel: list[int] | Any, sphere_id):
    from own_physics import calculate_force, _contact_point_velocity

    # Get global poses (world frame)
    sphere_pos, sphere_quat = p.getBasePositionAndOrientation(sphere_id)
    plane_pos, plane_quat = p.getBasePositionAndOrientation(plane_id)

    # Get velocities
    _, angular_vel = p.getBaseVelocity(sphere_id)

    # Convert quaternions to Euler angles for easier understanding
    cube_euler = p.getEulerFromQuaternion(sphere_quat)
    plane_euler = p.getEulerFromQuaternion(plane_quat)

    # Calculate relative position (cube relative to plane)
    relative_pos = [
        sphere_pos[0] - plane_pos[0],
        sphere_pos[1] - plane_pos[1],
        sphere_pos[2] - plane_pos[2],
    ]

    # Calculate relative rotation (cube relative to plane)
    inv_plane_quat = p.invertTransform([0, 0, 0], plane_quat)[1]
    relative_quat = p.multiplyTransforms([0, 0, 0], inv_plane_quat,
                                         [0, 0, 0], sphere_quat)[1]
    relative_euler = p.getEulerFromQuaternion(relative_quat)

    # Prepare contact points list
    points = []

    for contact in contact_points:
        contact_normal = contact[7]
        contact_pos_on_self = contact[5]  # contact position in world space (on cube)

        # Calculate lever arm (contact position relative to cube center)
        lever_arm = [
            contact_pos_on_self[0] - sphere_pos[0],
            contact_pos_on_self[1] - sphere_pos[1],
            contact_pos_on_self[2] - sphere_pos[2],
        ]

        v_contact = _contact_point_velocity(sphere_id, contact_pos_on_self, current_linear_vel, angular_vel)
        force = calculate_force(contact_normal, contact[8], sphere_id, v_contact.tolist())

        points.append({
            "contact_position_world": {
                "x": float(contact_pos_on_self[0]),
                "y": float(contact_pos_on_self[1]),
                "z": float(contact_pos_on_self[2]),
            },
            "contact_position_relative_to_self": {
                "x": float(lever_arm[0]),
                "y": float(lever_arm[1]),
                "z": float(lever_arm[2]),
            },
            "force": {
                "x": float(force[0]),
                "y": float(force[1]),
                "z": float(force[2]),
            },
            "penetration": contact[8],
            "contact_normal": {
                "x": float(contact_normal[0]),
                "y": float(contact_normal[1]),
                "z": float(contact_normal[2]),
            }
        })

    collision_entry = {
        "frame": frame,
        "linear_velocity": {
            "x": float(current_linear_vel[0]),
            "y": float(current_linear_vel[1]),
            "z": float(current_linear_vel[2]),
        },
        "angular_velocity": {
            "x": float(angular_vel[0]),
            "y": float(angular_vel[1]),
            "z": float(angular_vel[2]),
        },
        "self_position": {
            "x": float(sphere_pos[0]),
            "y": float(sphere_pos[1]),
            "z": float(sphere_pos[2]),
        },
        "self_rotation": {
            "qx": float(sphere_quat[0]),
            "qy": float(sphere_quat[1]),
            "qz": float(sphere_quat[2]),
            "qw": float(sphere_quat[3]),
            "roll": float(cube_euler[0]),
            "pitch": float(cube_euler[1]),
            "yaw": float(cube_euler[2]),
        },
        "relative_position_to_collider": {
            "x": float(relative_pos[0]),
            "y": float(relative_pos[1]),
            "z": float(relative_pos[2]),
        },
        "relative_rotation_to_collider": {
            "roll": float(relative_euler[0]),
            "pitch": float(relative_euler[1]),
            "yaw": float(relative_euler[2]),
        },
        "self_scale": {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
        },
        "self_transform": create_transform_data(p, sphere_pos, sphere_quat, [1.0, 1.0, 1.0]),
        "collider_position": {
            "x": float(plane_pos[0]),
            "y": float(plane_pos[1]),
            "z": float(plane_pos[2]),
        },
        "collider_rotation": {
            "qx": float(plane_quat[0]),
            "qy": float(plane_quat[1]),
            "qz": float(plane_quat[2]),
            "qw": float(plane_quat[3]),
            "roll": float(plane_euler[0]),
            "pitch": float(plane_euler[1]),
            "yaw": float(plane_euler[2]),
        },
        "collider_scale": {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
        },
        "collider_transform": create_transform_data(p, plane_pos, plane_quat, [1.0, 1.0, 1.0]),
        "points": points,
        "collider_name": "plane",
        "collider_id": plane_id,
        "collider_shape_index": None,
        "self_mesh": None,
        "collider_mesh": None,
    }

    collision_data.append(collision_entry)

    # Store a simpler summary for collision points if needed
    collision_point_entry = {
        "self_position": {
            "x": float(sphere_pos[0]),
            "y": float(sphere_pos[1]),
            "z": float(sphere_pos[2]),
        },
        "linear_velocity": {
            "x": float(current_linear_vel[0]),
            "y": float(current_linear_vel[1]),
            "z": float(current_linear_vel[2]),
        },
        "angular_velocity": {
            "x": float(angular_vel[0]),
            "y": float(angular_vel[1]),
            "z": float(angular_vel[2]),
        },
        "self_rotation": {
            "qx": float(sphere_quat[0]),
            "qy": float(sphere_quat[1]),
            "qz": float(sphere_quat[2]),
            "qw": float(sphere_quat[3]),
            "roll": float(cube_euler[0]),
            "pitch": float(cube_euler[1]),
            "yaw": float(cube_euler[2]),
        },
        "collider_position": {
            "x": float(plane_pos[0]),
            "y": float(plane_pos[1]),
            "z": float(plane_pos[2]),
        },
        "collider_rotation": {
            "qx": float(plane_quat[0]),
            "qy": float(plane_quat[1]),
            "qz": float(plane_quat[2]),
            "qw": float(plane_quat[3]),
            "roll": float(plane_euler[0]),
            "pitch": float(plane_euler[1]),
            "yaw": float(plane_euler[2]),
        },

        "relative_position_to_collider": {
            "x": float(relative_pos[0]),
            "y": float(relative_pos[1]),
            "z": float(relative_pos[2]),
        },
        "relative_rotation_to_collider": {
            "roll": float(relative_euler[0]),
            "pitch": float(relative_euler[1]),
            "yaw": float(relative_euler[2]),
        },
        "points": points,
    }

    collision_points.append(collision_point_entry)

def record_collision_empty(p, plane_id: int, cube_id:int, empty_collision_points:list[Any], current_linear_vel:list):
    # Get current state after collision
    pos, quat = p.getBasePositionAndOrientation(cube_id)

    collision_point_entry={}
    # Get plane position and orientation
    plane_pos, plane_quat = p.getBasePositionAndOrientation(plane_id)

    # Calculate relative position (cube relative to plane)
    relative_pos = [
        pos[0] - plane_pos[0],
        pos[1] - plane_pos[1],
        pos[2] - plane_pos[2],
    ]

    # Calculate relative rotation (cube relative to plane)
    inv_plane_quat = p.invertTransform([0, 0, 0], plane_quat)[1]
    relative_quat = p.multiplyTransforms([0, 0, 0], inv_plane_quat,
                                         [0, 0, 0], quat)[1]
    relative_euler = p.getEulerFromQuaternion(relative_quat)


    _, angular_vel = p.getBaseVelocity(cube_id)



    collision_point_entry["self_position"] = {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2])
    }

    collision_point_entry["relative_position_to_collider"] = {
        "x": float(relative_pos[0]),
        "y": float(relative_pos[1]),
        "z": float(relative_pos[2])
    }

    collision_point_entry["relative_rotation_to_collider"] = {
        "roll": float(relative_euler[0]),
        "pitch": float(relative_euler[1]),
        "yaw": float(relative_euler[2])
    }

    collision_point_entry["self_rotation"] = {
        "x": float(quat[0]),
        "y": float(quat[1]),
        "z": float(quaternion_to_euler(p, quat))
    }
    collision_point_entry["collider_position"] = {
        "x": float(plane_pos[0]),
        "y": float(plane_pos[1]),
        "z": float(plane_pos[2])
    }
    collision_point_entry["self_rotation"] = {
        "x": float(quat[0]),
        "y": float(quat[1]),
        "z": float(quaternion_to_euler(p, quat))

    }
    collision_point_entry["collider_rotation"] = {
        "x": float(plane_quat[0]),
        "y": float(plane_quat[1]),
        "z": float(quaternion_to_euler(p, plane_quat))
    }
    collision_point_entry["linear_velocity"] = {
        "x": current_linear_vel[0],
        "y": current_linear_vel[1],
        "z": current_linear_vel[2]
    }
    collision_point_entry["angular_velocity"] = {
        "x": angular_vel[0],
        "y": angular_vel[1],
        "z": angular_vel[2]
    }

    collision_point_entry["points"] = []
    empty_collision_points.append(collision_point_entry)