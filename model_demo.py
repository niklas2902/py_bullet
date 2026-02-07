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

from model import ImpulesePredictor, NumberContactPointsPredictor, ContactPointsPredictor
from own_physics import calculate_force
from recorder import record_collision, record_collision_empty
from scene_creator import create_scene

SPRING_CONSTANT = 1000 #N/m
DAMPENING = 0.9
BOUNCINESS_FACTOR = 0.3
MAX_RUNS = 20000
GRAVITY_RUNS = 200
MAX_FRAMES = 20000

contact_number_model = NumberContactPointsPredictor()
contact_number_model.load_state_dict(torch.load("number_of_contact_points_model.pth", map_location="cpu"))

contact_points_predictor = ContactPointsPredictor()
checkpoint = torch.load("contact_points_model.pth", map_location="cpu")
contact_points_predictor.load_state_dict(checkpoint['model_state_dict'])


impulse_predictor = ImpulesePredictor()
impulse_predictor.load_state_dict(torch.load("best_impulse_model.pth", map_location="cpu"))


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


def apply_impulse_predictor(cube_id, current_linear_vel, relative_pos, relative_rot, contact_points, number):
    # Extract features for contact points predictor
    features = []
    features.extend(relative_pos)  # x, y, z
    features.extend(relative_rot)  # roll, pitch, yaw

    if len(contact_points) > 4:
        raise Exception("Too many contact points")

    # Convert to tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension

    # Normalize using saved statistics
    if checkpoint.get('feature_stats') is not None:
        feature_mean = checkpoint['feature_stats']['mean']
        feature_std = checkpoint['feature_stats']['std']
        features_tensor = (features_tensor - feature_mean) / feature_std

    # Get predicted contact points (normalized)
    with torch.no_grad():
        predictions = contact_points_predictor(features_tensor)

    # Denormalize predictions using saved statistics
    if checkpoint.get('target_stats') is not None:
        predictions_reshaped = predictions.view(-1, 4, 3)  # (batch, 4 points, 3 coords)

        for coord_idx in range(3):  # x, y, z
            coord_mean = checkpoint['target_stats']['mean'][coord_idx]
            coord_std = checkpoint['target_stats']['std'][coord_idx]

            predictions_reshaped[:, :, coord_idx] = (
                    predictions_reshaped[:, :, coord_idx] * coord_std + coord_mean
            )

        predictions = predictions_reshaped.view(-1, 12)

    # Extract predicted contact points
    predictions = predictions.squeeze(0).numpy()  # Remove batch dimension

    # Now use impulse_predictor to get forces
    # (Assuming impulse_predictor takes different features)
    impulse_features = list(current_linear_vel)
    impulse_features.extend(relative_pos)
    impulse_features.extend(relative_rot)

    forces = impulse_predictor(torch.FloatTensor(impulse_features))

    # Apply forces at predicted contact point positions
    for index in range(int(number)):
        predicted_point = predictions[index * 3: index * 3 + 3]

        # Check if this is a valid prediction (not a padding point)
        if not (predicted_point[0] == -1 and predicted_point[1] == -1 and predicted_point[2] == -1):
            force = forces[index * 3: index * 3 + 3]
            # Apply force at predicted contact point position
            p.applyExternalForce(cube_id, -1, force.tolist(), predicted_point.tolist(), p.WORLD_FRAME)

def main():
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    # Check connection type
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    frame = 0
    plane_id,  cube_id, timestep = create_scene(p, True)
    p.resetBaseVelocity(
        cube_id,
        linearVelocity=[-5, 5, 0],  # Forward velocity in x-direction
        angularVelocity=[0, 0, 0]  # No rotation
    )
    while frame < MAX_FRAMES:
        # Store velocities before simulation step
        current_linear_vel, current_angular_vel = p.getBaseVelocity(cube_id)

        # Step simulation
        p.stepSimulation()

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

        # Get contact points
        features = torch.FloatTensor([relative_pos[0], relative_pos[1], relative_pos[2],relative_euler[0], relative_euler[1], relative_euler[2]])

        pred = contact_number_model(features)
        number = torch.round(torch.clamp(pred, min=0, max=4)).item()

        contact_points = p.getContactPoints(bodyA=cube_id, bodyB=plane_id)
        if number:
            #apply_force(contact_points, current_angular_vel, current_linear_vel,
            #            None, prev_angular_vel, prev_linear_vel, cube_id, plane_id, timestep)
            apply_impulse_predictor(cube_id, current_linear_vel, relative_pos, relative_euler, contact_points, number)

            # DEBUG: Check velocity after applying forces
            vel_after, _ = p.getBaseVelocity(cube_id)

        # Update previous velocities
        prev_linear_vel = current_linear_vel
        prev_angular_vel = current_angular_vel

        frame += 1
        if connection_type == p.GUI:#
            time.sleep(timestep)

    p.disconnect()

if __name__ == "__main__":
    main()

