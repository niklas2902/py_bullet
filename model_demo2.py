import json
import random
import time
from pyexpat import features
from typing import Any

import torch
import math
import pybullet as p
import numpy as np
import tqdm

from model import ImpulesePredictor, NumberContactPointsPredictor, ContactPointsPredictor, CollisionPredictor
from own_physics import calculate_force
from parameters import SceneParameters
from recorder import record_collision, record_collision_empty
from scene_creator import create_scene, random_quaternion
import time

SPRING_CONSTANT = 1000  # N/m
DAMPENING = 0.9
BOUNCINESS_FACTOR = 0.3
MAX_RUNS = 60000
GRAVITY_RUNS = 200
MAX_FRAMES = 20000

all_impulse_predictor = CollisionPredictor(
    input_dim=9,
)
checkpoint = torch.load("all_impulse_model_backup.pth", map_location="cpu")
all_impulse_predictor.load_state_dict(checkpoint['model_state_dict'])
all_impulse_predictor.eval()  # Set to evaluation mode

# Extract normalization stats
feature_stats = checkpoint.get('feature_stats')
target_stats = checkpoint.get('target_stats')


def apply_force(contact_points,
                current_angular_vel, current_linear_vel,
                model,
                prev_angular_vel, prev_linear_vel,
                cube_id,
                plane_id,
                timestep: float):
    for cp in contact_points:
        apply_spring_force(cp, cube_id, current_linear_vel)


def apply_spring_force(normal, penetration, cp, cube_id, current_linear_vel):
    force_vector = calculate_force(normal, penetration, cube_id, current_linear_vel)

    p.applyExternalForce(cube_id, -1, force_vector.tolist(), cp[5], p.WORLD_FRAME)


def apply_impulse_predictor(cube_id, current_linear_vel, relative_pos, relative_rot):
    features = []
    features.extend(current_linear_vel)          # 3 values
    features.extend([0, 0, relative_pos[2]])     # 3 values
    features.extend(relative_rot)               # 3 values (roll, pitch, yaw)

    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, 9)

    if feature_stats is not None:
        features_tensor = (features_tensor - feature_stats['mean']) / feature_stats['std']

    with torch.no_grad():
        predictions = all_impulse_predictor(features_tensor)

    # NEW: num_contacts is now class logits [B, max_contacts+1], use argmax
    num_contacts_pred = predictions["num_contacts"].argmax(dim=-1).item()  # scalar int

    # NEW: contact_points and impulses are already [B, C, 3]
    contact_points = predictions["contact_points"].squeeze(0)  # (4, 3)
    impulses = predictions["impulses"].squeeze(0)              # (4, 3)

    # Denormalize using target_stats (same as before, adjusted for new shape)
    if target_stats is not None:
        for i in range(3):
            contact_points[:, i] = contact_points[:, i] * target_stats["cp_std"][i] + target_stats["cp_mean"][i]
            impulses[:, i]       = impulses[:, i]       * target_stats["imp_std"][i] + target_stats["imp_mean"][i]

    contact_points = contact_points.numpy()
    impulses = impulses.numpy()

    for index in range(num_contacts_pred):
        predicted_point = contact_points[index] + np.array(relative_pos)
        predicted_force = impulses[index]

        p.applyExternalForce(
            cube_id,
            -1,
            predicted_force.tolist(),
            predicted_point.tolist(),
            p.WORLD_FRAME
        )

def main():
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    # Check connection type
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    frame = 0
    plane_id, cube_id, timestep = create_scene(p, True, SceneParameters(random_rotation = True))

    log_id = p.startStateLogging(
        p.STATE_LOGGING_VIDEO_MP4,
        "collision_run.mp4"
    )

    # Disable ALL collisions for plane
    p.setCollisionFilterGroupMask(
        plane_id, -1,
        collisionFilterGroup=1,
        collisionFilterMask=0
    )

    # Disable ALL collisions for cube
    p.setCollisionFilterGroupMask(
        cube_id, -1,
        collisionFilterGroup=1,
        collisionFilterMask=0
    )
    p.resetBaseVelocity(
        cube_id,
        linearVelocity=[0, 0, 0],  # Forward velocity in x-direction
        angularVelocity=[0, 0, 0]  # No rotation
    )

    p.resetBasePositionAndOrientation(cube_id, [0, 0, 2], random_quaternion())
    initial_orientation = p.getQuaternionFromEuler([0.0, 0.5, 0.0])

    prev_linear_vel = [0, 0, 0]
    prev_angular_vel = [0, 0, 0]

    while frame < MAX_FRAMES:
        # Store velocities before simulation step
        start_time = time.time() * 1000
        current_linear_vel, current_angular_vel = p.getBaseVelocity(cube_id)

        # Get positions and orientations
        plane_pos, plane_orn = p.getBasePositionAndOrientation(plane_id)
        cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)

        # Convert to numpy
        plane_pos = np.array(plane_pos)
        cube_pos = np.array(cube_pos)

        # Get relative position in plane's frame
        plane_rot_matrix = np.array(p.getMatrixFromQuaternion(plane_orn)).reshape(3, 3)
        relative_pos = plane_rot_matrix.T @ (cube_pos - plane_pos)

        # Get relative rotation
        plane_orn_inv = p.invertTransform([0, 0, 0], plane_orn)[1]
        relative_quat = p.multiplyTransforms([0, 0, 0], plane_orn_inv,
                                             [0, 0, 0], cube_orn)[1]
        relative_euler = p.getEulerFromQuaternion(relative_quat)

        # Apply predicted forces BEFORE stepping simulation
        apply_impulse_predictor(cube_id, current_linear_vel, relative_pos, relative_euler)

        # Step simulation
        p.stepSimulation()

        # DEBUG: Check velocity after stepping
        vel_after, _ = p.getBaseVelocity(cube_id)

        # Update previous velocities
        prev_linear_vel = current_linear_vel
        prev_angular_vel = current_angular_vel

        frame += 1
        if connection_type == p.GUI:
            time.sleep(timestep - (min(0,time.time() * 1000 - start_time)))
    p.stopStateLogging(log_id)
    p.disconnect()


if __name__ == "__main__":
    main()