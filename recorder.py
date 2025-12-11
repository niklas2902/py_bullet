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


def record_collision(p, collision_data: list[Any], collision_points:list[Any], contact_points, frame: int, plane_id,
                     prev_angular_vel: list[int] | Any,
                     prev_linear_vel: list[int] | Any, sphere_id):
    from own_physics import calculate_force
    # Get current state after collision
    pos, quat = p.getBasePositionAndOrientation(sphere_id)
    linear_vel, angular_vel = p.getBaseVelocity(sphere_id)

    # Extract impulse from contact points
    id = 0
    collision_point_entry={}
    points = []
    plane_pos, plane_quat = p.getBasePositionAndOrientation(plane_id)

    collision_point_entry["self_position"] ={
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2])
        }

    collision_point_entry["self_rotation"] ={
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quaternion_to_euler(p, quat))
        }
    collision_point_entry["collider_position"] = {
            "x": float(plane_pos[0]),
            "y": float(plane_pos[1]),
            "z": float(plane_pos[2])
        },
    collision_point_entry["self_rotation"] = {
           "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quaternion_to_euler(p, quat))

    }
    collision_point_entry["collider_rotation"]=  {
        "x": float(plane_quat[0]),
        "y": float(plane_quat[1]),
        "z": float(quaternion_to_euler(p, plane_quat))
    },

    for contact in contact_points:

        contact_normal = contact[7]
        # For angular impulse, we need the contact position and the impulse
        contact_pos_on_self = contact[5]  # Position on bodyA (sphere)

        force = calculate_force(contact, sphere_id, contact_normal)

        points.append({
            "contact_position": {
                "x": float(contact_pos_on_self[0]),
                "y": float(contact_pos_on_self[1]),
                "z": float(contact_pos_on_self[2])
            },
            "force": {
                "x": float(force[0]),
                "y": float(force[1]),
                "z": float(force[2])
            },

        "penetration": contact[8],
        "contact_normal": to_vector(contact[7])
        })

    collision_entry = {
        "frame": frame,
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
        "collider_transform": create_transform_data(p, plane_pos, plane_quat, [1.0, 1.0, 1.0]),
        "points":points
    }
    collision_data.append(collision_entry)
    collision_point_entry["points"] = points
    collision_points.append(collision_point_entry)

def record_collision_empty(p, plane_id: int, cube_id:int, empty_collision_points:list[Any]):
    # Get current state after collision
    pos, quat = p.getBasePositionAndOrientation(cube_id)

    collision_point_entry={}
    # Get plane position and orientation
    plane_pos, plane_quat = p.getBasePositionAndOrientation(plane_id)

    collision_point_entry["self_position"] = {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2])
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
    },
    collision_point_entry["self_rotation"] = {
        "x": float(quat[0]),
        "y": float(quat[1]),
        "z": float(quaternion_to_euler(p, quat))

    }
    collision_point_entry["collider_rotation"] = {
        "x": float(plane_quat[0]),
        "y": float(plane_quat[1]),
        "z": float(quaternion_to_euler(p, plane_quat))
    },

    collision_point_entry["points"] = []
    empty_collision_points.append(collision_point_entry)