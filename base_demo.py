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
from parameters import SceneParameters
from recorder import record_collision, record_collision_empty
from scene_creator import create_scene

SPRING_CONSTANT = 1000  # N/m
DAMPENING = 0.9
BOUNCINESS_FACTOR = 0.3
MAX_RUNS = 20000
GRAVITY_RUNS = 200
MAX_FRAMES = 2000


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


def apply_impulse_predictor(cube_id, current_linear_vel, contact_points):
    features = list(current_linear_vel)
    if len(contact_points) > 4:
        raise Exception("Too many contact points")

    # Get the cube's center of mass position (world frame) to compute torque arms
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    cube_pos = np.array(cube_pos)

    # Accumulate net force and net torque from all contact points
    net_force = np.zeros(3)
    net_torque = np.zeros(3)

    for cp in contact_points:
        pen = cp[8]
        normal = cp[7]
        contact_pos_world = np.array(cp[5])
        features.extend([pen, normal[0], normal[1], normal[2]])

        # Compute the force at this contact point
        force_vector = calculate_force(normal, pen, cube_id, current_linear_vel)
        force_vector = np.asarray(force_vector)

        # Accumulate force
        net_force += force_vector

        # Torque = r x F, where r is the lever arm from center of mass to contact point
        r = contact_pos_world - cube_pos
        net_torque += np.cross(r, force_vector)

    for i in range(len(contact_points), 4):
        features.extend([0, 0, 0, 0])

    # Apply single net force at the center of mass (no torque from this call since r=0)
    # and apply the accumulated torque separately
    if len(contact_points) > 0:
        p.applyExternalForce(
            cube_id, -1,
            net_force.tolist(),
            cube_pos.tolist(),
            p.WORLD_FRAME,
        )
        p.applyExternalTorque(
            cube_id, -1,
            net_torque.tolist(),
            p.WORLD_FRAME,
        )

    print("-----------------------------------")


def main():
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    # Check connection type
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    frame = 0
    prev_linear_vel = [0, 0, 0]
    prev_angular_vel = [0, 0, 0]

    log_id = p.startStateLogging(
        p.STATE_LOGGING_VIDEO_MP4,
        "collision_run_gt.mp4"
    )

    plane_id, cube_id, timestep = create_scene(p, True, SceneParameters(random_rotation = True))
    p.resetBaseVelocity(
        cube_id,
        linearVelocity=[0, 0, 0],
        angularVelocity=[0, 0, 0]  # No rotation
    )  # Forward velocity in x-direction
    initial_orientation = p.getQuaternionFromEuler([0.0, 0.5, 0.0])

    p.resetBasePositionAndOrientation(
        cube_id,
        p.getBasePositionAndOrientation(cube_id)[0],
        initial_orientation
    )

    while frame < MAX_FRAMES:
        # Store velocities before simulation step
        current_linear_vel, current_angular_vel = p.getBaseVelocity(cube_id)

        # Step simulation
        p.stepSimulation()

        # Get contact points
        contact_points = p.getContactPoints(bodyA=cube_id, bodyB=plane_id)
        if contact_points:
            # apply_force(contact_points, current_angular_vel, current_linear_vel,
            #            None, prev_angular_vel, prev_linear_vel, cube_id, plane_id, timestep)
            apply_impulse_predictor(cube_id, current_linear_vel, contact_points)

            # DEBUG: Check velocity after applying forces
            vel_after, _ = p.getBaseVelocity(cube_id)

        # Update previous velocities
        prev_linear_vel = current_linear_vel
        prev_angular_vel = current_angular_vel

        frame += 1
        if connection_type == p.GUI:
            time.sleep(timestep)
    p.stopStateLogging(log_id)
    p.disconnect()


if __name__ == "__main__":
    main()