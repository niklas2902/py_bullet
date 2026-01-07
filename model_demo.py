import json
import random
import time
from typing import Any

import torch
import math
import pybullet as p
import numpy as np
import tqdm

from model import ImpulesePredictor
from own_physics import calculate_force
from recorder import record_collision, record_collision_empty
from scene_creator import create_scene

SPRING_CONSTANT = 1000 #N/m
DAMPENING = 0.9
BOUNCINESS_FACTOR = 0.3
MAX_RUNS = 20000
GRAVITY_RUNS = 200
MAX_FRAMES = 20000

impulse_model = ImpulesePredictor(19)
impulse_model.load_state_dict(torch.load("best_impulse_model.pth", map_location="cpu"))
impulse_model.eval()

def apply_force(contact_points,
                current_angular_vel, current_linear_vel,
                model,
                prev_angular_vel, prev_linear_vel,
                cube_id,
                plane_id,
                 timestep: float):

        for cp in contact_points:
            apply_spring_force(cp, cube_id, current_linear_vel)


def apply_spring_force(cp, cube_id, current_linear_vel):
    force_vector = calculate_force(cp, cube_id, current_linear_vel)

    p.applyExternalForce(cube_id, -1, force_vector.tolist(), cp[5], p.WORLD_FRAME)

def apply_impulse_predictor(cube_id, current_linear_vel, contact_points):
    features = list(current_linear_vel)
    if len(contact_points) > 4:
        raise Exception("Too many contact points")
    for cp in contact_points:
        pen = cp[8]
        normal = cp[7]
        features.extend([pen, normal[0], normal[1], normal[2]])

    for i in range(len(contact_points), 4):
        features.extend([0, 0, 0, 0])

    impulse_model.eval()  # ADD THIS - important!
    with torch.no_grad():
        feature_tensor = torch.FloatTensor(features)

        # FIX: Add batch dimension!
        feature_tensor = feature_tensor.unsqueeze(0)  # Shape: [1, 19]

        forces = impulse_model(feature_tensor)

        # FIX: Remove batch dimension from output
        forces = forces.squeeze(0)  # Shape: [12] instead of [1, 12]

    for index, cp in enumerate(contact_points):
        force = forces[index * 3:index * 3 + 3]
        print(f"--------------Contact Point {index}----------------")
        print(f"penetration: {cp[8]} | normal: {cp[7]} | linear velocity: {current_linear_vel}")
        print(f"feature tensor: {feature_tensor}")
        print(f"features in list: {features}")
        print(f"force:{force}")
        print(f"calculated:{calculate_force(cp, cube_id, current_linear_vel)}")
        #apply_spring_force(cp, cube_id, current_linear_vel)
        p.applyExternalForce(cube_id, -1, force.tolist(), cp[5], p.WORLD_FRAME)

    print("-----------------------------------")

def main():
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    # Check connection type
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    frame = 0
    prev_linear_vel = [0, 0, 0]
    prev_angular_vel = [0, 0, 0]

    plane_id,  cube_id, timestep = create_scene(p, True)
    while frame < MAX_FRAMES:
        # Store velocities before simulation step
        current_linear_vel, current_angular_vel = p.getBaseVelocity(cube_id)

        # Step simulation
        p.stepSimulation()


        # Get contact points
        contact_points = p.getContactPoints(bodyA=cube_id, bodyB=plane_id)
        if contact_points:
            #apply_force(contact_points, current_angular_vel, current_linear_vel,
            #            None, prev_angular_vel, prev_linear_vel, cube_id, plane_id, timestep)
            apply_impulse_predictor(cube_id, current_linear_vel, contact_points)

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

