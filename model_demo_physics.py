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

from models.model_physical import  WrenchPredictor
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
MAX_FRAMES = 2000

all_impulse_predictor = WrenchPredictor(input_dim=10, width=128)
checkpoint = torch.load("checkpoints/wrench_model_best_phys.pth", map_location="cpu")
state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint['model_state_dict'].items()}
all_impulse_predictor.load_state_dict(state_dict)
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
    features.extend([relative_pos[2]])     # 3 values
    roll, pitch, yaw = relative_rot
    features.extend([math.sin(roll), math.cos(roll),
            math.sin(pitch), math.cos(pitch),
            math.sin(yaw), math.cos(yaw)])               # 3 values (roll, pitch, yaw)

    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, 9)

    with torch.no_grad():
        predictions = all_impulse_predictor(features_tensor)

    # NEW: num_contacts is now class logits [B, max_contacts+1], use argmax
    print(predictions)
    is_collision = is_collision = (torch.sigmoid(predictions["collision_logit"]) > 0.5).item()

    # NEW: contact_points and impulses are already [B, C, 3]
    torque = predictions["torque"].squeeze(0)  # (4, 3)
    force = predictions["force"].squeeze(0)              # (4, 3)

    force = force.numpy()
    torque = torque.numpy()
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    cube_pos = np.array(cube_pos)
    
    if is_collision:
        p.applyExternalForce(
            cube_id, -1,
            force.tolist(),
            cube_pos.tolist(),
            p.WORLD_FRAME,
        )
        p.applyExternalTorque(
            cube_id, -1,
            torque.tolist(),
            p.WORLD_FRAME,
        )

def main():
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    # Check connection type
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    frame = 0
    plane_id,  cube_id, timestep = create_scene(p, True, SceneParameters(random_rotation = True))
    initial_orientation = p.getQuaternionFromEuler([0.0, 0.2, 0.0])

    p.resetBasePositionAndOrientation(
        cube_id,
        p.getBasePositionAndOrientation(cube_id)[0],
        initial_orientation
    )
    p.resetBaseVelocity(
        cube_id,
        linearVelocity=[0, 0, 0],
        angularVelocity=[0, 0, 0]  # No rotation
    )  # Forward velocity in x-direction



    log_id = p.startStateLogging(
        p.STATE_LOGGING_VIDEO_MP4,
        "collision_run_mlp.mp4"
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