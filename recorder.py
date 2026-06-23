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


def record_collision(p, collision_data: list[Any], collision_points: list[Any], 
                     contact_points_A, contact_points_B, frame: int,
                    objectA_id, objectB_id):
    from own_physics import calculate_force, _contact_point_velocity
    assert(len(contact_points_A) == len(contact_points_B))
    # Get global poses (world frame)
    objectA_pos, objectA_quat = p.getBasePositionAndOrientation(objectA_id)
    objectB_pos, objectB_quat = p.getBasePositionAndOrientation(objectB_id)

    # Get velocities
    linear_velA, angular_velA = p.getBaseVelocity(objectA_id)
    linear_velB, angular_velB = p.getBaseVelocity(objectB_id)

    # Convert quaternions to Euler angles for easier understanding
    objectA_euler = p.getEulerFromQuaternion(objectA_quat)
    objectB_euler = p.getEulerFromQuaternion(objectB_quat)

    # Calculate relative position (cube relative to plane)
    relative_pos = [
        objectA_pos[0] - objectB_pos[0],
        objectA_pos[1] - objectB_pos[1],
        objectA_pos[2] - objectB_pos[2],
    ]

    # Calculate relative rotation (cube relative to plane)
    inv_plane_quat = p.invertTransform([0, 0, 0], objectB_quat)[1]
    relative_quat = p.multiplyTransforms([0, 0, 0], inv_plane_quat,
                                         [0, 0, 0], objectA_quat)[1]
    relative_euler = p.getEulerFromQuaternion(relative_quat)

    # Prepare contact points list
    points = []

    for collision_point_index in range(len(contact_points_A)):
        contactA = contact_points_A[collision_point_index]
        contactB = contact_points_B[collision_point_index]

        contact_normalA = contactA[7]
        contact_pos_on_selfA = contactA[5]
        contact_normalB = contactB[7]
        contact_pos_on_selfB = contactB[5]

        # Calculate lever arm (contact position relative to cube center)
        lever_armA = [
            contact_pos_on_selfA[0] - objectA_pos[0],
            contact_pos_on_selfA[1] - objectA_pos[1],
            contact_pos_on_selfA[2] - objectA_pos[2],
        ]

        lever_armB = [
            contact_pos_on_selfB[0] - objectB_pos[0],
            contact_pos_on_selfB[1] - objectB_pos[1],
            contact_pos_on_selfB[2] - objectB_pos[2],
        ]

        v_contactA = _contact_point_velocity(objectA_id, contact_pos_on_selfA, linear_velA, angular_velA)
        forceA = calculate_force(contact_normalA, contactA[8], objectA_id, v_contactA.tolist())

        v_contactB = _contact_point_velocity(objectB_id, contact_pos_on_selfB, linear_velB, angular_velB)
        forceB = calculate_force(contact_normalB, contactB[8], objectB_id, v_contactB.tolist())

        points.append({
            "contact_position_world": {
                "x": float(contact_pos_on_selfA[0]),
                "y": float(contact_pos_on_selfA[1]),
                "z": float(contact_pos_on_selfA[2]),
            },
            "contact_position_relative_to_A": {
                "x": float(lever_armA[0]),
                "y": float(lever_armA[1]),
                "z": float(lever_armA[2]),
            },
            "contact_position_relative_to_B": {
                "x": float(lever_armB[0]),
                "y": float(lever_armB[1]),
                "z": float(lever_armB[2]),
            },
            "forceA": {
                "x": float(forceA[0]),
                "y": float(forceA[1]),
                "z": float(forceA[2]),
            },
            "forceB": {
                "x": float(forceB[0]),
                "y": float(forceB[1]),
                "z": float(forceB[2]),
            },
            "penetration": contactA[8],
            "contact_normalA": {
                "x": float(contact_normalA[0]),
                "y": float(contact_normalA[1]),
                "z": float(contact_normalA[2]),
            },
            "contact_normalB": {
                "x": float(contact_normalB[0]),
                "y": float(contact_normalB[1]),
                "z": float(contact_normalB[2]),
            }
        })

    collision_entry = {
        "frame": frame,
        "linear_velocityA": {
            "x": float(linear_velA[0]),
            "y": float(linear_velA[1]),
            "z": float(linear_velA[2]),
        },
        "linear_velocityB": {
            "x": float(linear_velB[0]),
            "y": float(linear_velB[1]),
            "z": float(linear_velB[2]),
        },
        "angular_velocityA": {
            "x": float(angular_velA[0]),
            "y": float(angular_velA[1]),
            "z": float(angular_velA[2]),
        },
        "angular_velocityB": {
            "x": float(angular_velB[0]),
            "y": float(angular_velB[1]),
            "z": float(angular_velB[2]),
        },
        "A_pos": {
            "x": float(objectA_pos[0]),
            "y": float(objectA_pos[1]),
            "z": float(objectA_pos[2]),
        },
        "B_pos": {
            "x": float(objectB_pos[0]),
            "y": float(objectB_pos[1]),
            "z": float(objectB_pos[2]),
        },
        "A_rotation": {
            "qx": float(objectA_quat[0]),
            "qy": float(objectA_quat[1]),
            "qz": float(objectA_quat[2]),
            "qw": float(objectA_quat[3]),
            "roll": float(objectA_euler[0]),
            "pitch": float(objectA_euler[1]),
            "yaw": float(objectA_euler[2]),
        },

        "B_rotation": {
            "qx": float(objectB_quat[0]),
            "qy": float(objectB_quat[1]),
            "qz": float(objectB_quat[2]),
            "qw": float(objectB_quat[3]),
            "roll": float(objectB_euler[0]),
            "pitch": float(objectB_euler[1]),
            "yaw": float(objectB_euler[2]),
        },
        "relative_position_A_to_B": {
            "x": float(relative_pos[0]),
            "y": float(relative_pos[1]),
            "z": float(relative_pos[2]),
        },
        "relative_rotation_A_to_B": {
            "roll": float(relative_euler[0]),
            "pitch": float(relative_euler[1]),
            "yaw": float(relative_euler[2]),
        },
        "A_scale": {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
        },
        "B_scale": {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
        },
        "A_transform": create_transform_data(p, objectA_pos, objectA_quat, [1.0, 1.0, 1.0]),
        "B_transform": create_transform_data(p, objectB_pos, objectB_quat, [1.0, 1.0, 1.0]),
        "collider_scale": {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
        },
        "collider_transform": create_transform_data(p, objectB_pos, objectB_quat, [1.0, 1.0, 1.0]),
        "points": points,
        "A_mesh": None,
        "B_mesh": None,
    }

    collision_data.append(collision_entry)

    # Store a simpler summary for collision points if needed
    collision_point_entry = {
        "A_position": {
            "x": float(objectA_pos[0]),
            "y": float(objectA_pos[1]),
            "z": float(objectA_pos[2]),
        },
        "linear_velocityA": {
            "x": float(linear_velA[0]),
            "y": float(linear_velA[1]),
            "z": float(linear_velA[2]),
        },
        "angular_velocity": {
            "x": float(angular_velA[0]),
            "y": float(angular_velA[1]),
            "z": float(angular_velA[2]),
        },
        "A_rotation": {
            "qx": float(objectA_quat[0]),
            "qy": float(objectA_quat[1]),
            "qz": float(objectA_quat[2]),
            "qw": float(objectA_quat[3]),
            "roll": float(objectA_euler[0]),
            "pitch": float(objectA_euler[1]),
            "yaw": float(objectA_euler[2]),
        },
        "B_position": {
            "x": float(objectB_pos[0]),
            "y": float(objectB_pos[1]),
            "z": float(objectB_pos[2]),
        },
        "B_rotation": {
            "qx": float(objectB_quat[0]),
            "qy": float(objectB_quat[1]),
            "qz": float(objectB_quat[2]),
            "qw": float(objectB_quat[3]),
            "roll": float(objectB_euler[0]),
            "pitch": float(objectB_euler[1]),
            "yaw": float(objectB_euler[2]),
        },

        "relative_position_A_to_B": {
            "x": float(relative_pos[0]),
            "y": float(relative_pos[1]),
            "z": float(relative_pos[2]),
        },
        "relative_rotation_A_to_B": {
            "roll": float(relative_euler[0]),
            "pitch": float(relative_euler[1]),
            "yaw": float(relative_euler[2]),
        },
        "points": points,
    }

    collision_points.append(collision_point_entry)

def record_collision_empty(p, objectA_id: int, objectB_id:int, empty_collision_points:list[Any]):
    # Get current state after collision
    posA, quatA = p.getBasePositionAndOrientation(objectA_id)
    posB, quatB = p.getBasePositionAndOrientation(objectB_id)

    collision_point_entry={}
    
    # Calculate relative position (cube relative to plane)
    relative_pos = [
        posA[0] - posB[0],
        posA[1] - posB[1],
        posA[2] - posB[2],
    ]

    # Calculate relative rotation (cube relative to plane)
    inv_plane_quat = p.invertTransform([0, 0, 0], quatB)[1]
    relative_quat = p.multiplyTransforms([0, 0, 0], inv_plane_quat,
                                         [0, 0, 0], quatA)[1]
    
    relative_euler = p.getEulerFromQuaternion(relative_quat)


    linear_velA, angular_velA = p.getBaseVelocity(objectA_id)
    linear_velB, angular_velB = p.getBaseVelocity(objectB_id)



    collision_point_entry["A_position"] = {
        "x": float(posA[0]),
        "y": float(posA[1]),
        "z": float(posA[2])
    },
    
    collision_point_entry["B_position"] = {
        "x": float(posB[0]),
        "y": float(posB[1]),
        "z": float(posB[2])
    }

    collision_point_entry["relative_position_A_to_B"] = {
        "x": float(relative_pos[0]),
        "y": float(relative_pos[1]),
        "z": float(relative_pos[2])
    }

    collision_point_entry["relative_rotation_A_to_B"] = {
        "roll": float(relative_euler[0]),
        "pitch": float(relative_euler[1]),
        "yaw": float(relative_euler[2])
    }

    collision_point_entry["A_rotation"] = {
        "x": float(quatA[0]),
        "y": float(quatA[1]),
        "z": float(quaternion_to_euler(p, quatA))
    }
    collision_point_entry["B_rotation"] = {
        "x": float(quatB[0]),
        "y": float(quatB[1]),
        "z": float(quaternion_to_euler(p, quatB))
    }
    collision_point_entry["linear_velocity_A"] = {
        "x": linear_velA[0],
        "y": linear_velA[1],
        "z": linear_velA[2]
    }
    collision_point_entry["linear_velocity_B"] = {
        "x": linear_velB[0],
        "y": linear_velB[1],
        "z": linear_velB[2]
    }
    collision_point_entry["angular_velocity_A"] = {
        "x": angular_velA[0],
        "y": angular_velA[1],
        "z": angular_velA[2]
    }
    collision_point_entry["angular_velocity_B"] = {
        "x": angular_velB[0],
        "y": angular_velB[1],
        "z": angular_velB[2]
    }

    collision_point_entry["points"] = []
    empty_collision_points.append(collision_point_entry)