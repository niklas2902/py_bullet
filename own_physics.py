import json
import random
import time
from typing import Any

import torch
import math
import pybullet as p
import numpy as np
import tqdm

from parameters import SceneParameters
from recorder import record_collision, record_collision_empty
from scene_creator import create_scene, random_rotation_an_position

SPRING_CONSTANT = 1000 #N/m
DAMPENING = 0.9
BOUNCINESS_FACTOR = 0.3
MAX_RUNS = 100000
GRAVITY_RUNS = 0
MAX_FRAMES_GRAVITY = 0
MAX_FRAMES_NORMAL = 500
def apply_force(contact_points,
                current_angular_vel, current_linear_vel,
                model,
                prev_angular_vel, prev_linear_vel,
                cube_id,
                plane_id,
                 timestep: float):

        for cp in contact_points:
            apply_spring_force(cp[7], cp[8], cp[5], cube_id, current_linear_vel)


def apply_spring_force(normal, penetration,position, cube_id, current_linear_vel):
    force_vector = calculate_force(normal, penetration, cube_id, current_linear_vel)

    p.applyExternalForce(cube_id, -1, force_vector.tolist(), position, p.WORLD_FRAME)


def calculate_force(normal, penetration, cube_id, current_linear_vel) -> np.ndarray[Any, np.dtype[Any]] | Any:
    mass = p.getDynamicsInfo(cube_id, -1)[0]
    k = SPRING_CONSTANT
    c = 2 * math.sqrt(k * mass) * BOUNCINESS_FACTOR  # critical damping

    penetration = penetration
    normal = normal

    v = np.array(current_linear_vel)
    n = np.array(normal)

    vel_normal = np.dot(v, n)

    spring_force = -k * penetration
    damping_force = -c * vel_normal

    force_mag = spring_force + damping_force

    force_vector = force_mag * n
    return force_vector


def main(should_use_gravity:bool, max_frames:int, parameters: SceneParameters):
    # Connect to PyBullet
    physics_client = p.connect(p.DIRECT)
    # Check connection type
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    collision_data = []
    collision_point_data = []
    collision_point_data_empty = []
    frame = 0
    prev_linear_vel = [0, 0, 0]
    prev_angular_vel = [0, 0, 0]

    plane_id,  cube_id, timestep = create_scene(p, should_use_gravity, parameters)

    if should_use_gravity:
        pos, orn = p.getBasePositionAndOrientation(cube_id)
        p.resetBasePositionAndOrientation(cube_id, [pos[0], pos[1], random.random() * 4 + 1], orn)
    while frame < max_frames:
        # Store velocities before simulation step
        current_linear_vel, current_angular_vel = p.getBaseVelocity(cube_id)

        # Step simulation
        p.stepSimulation()


        # Get contact points
        contact_points = p.getContactPoints(bodyA=cube_id, bodyB=plane_id)
        if contact_points:
            record_collision(p, collision_data, collision_point_data, contact_points, frame, plane_id, prev_angular_vel, current_linear_vel,
                             cube_id)

            apply_force(contact_points, current_angular_vel, current_linear_vel,
                        None, prev_angular_vel, prev_linear_vel, cube_id, plane_id, timestep)
            if not should_use_gravity:
                break


        else:
            record_collision_empty(p, plane_id, cube_id, collision_point_data_empty, current_linear_vel)

        # Update previous velocities
        prev_linear_vel = current_linear_vel
        prev_angular_vel = current_angular_vel

        frame += 1
        if connection_type == p.GUI:#
            time.sleep(timestep)

    # Save collision data to JSON file
    with open(f'logs/collision_data-{time.time()}.json', 'w') as f:
        json.dump(collision_data, f, indent=4)

    num_collision_points = len(collision_point_data)
    random.shuffle(collision_point_data_empty)
    for i in range(num_collision_points):
        if i < len(collision_point_data_empty):
            collision_point_data.append(collision_point_data_empty[i])
    with open(f"logs/collision_points_{time.time()}.json", 'w') as f:
        json.dump(collision_point_data, f, indent=4)


    p.disconnect()

if __name__ == "__main__":

    for x_rot in tqdm.tqdm(range(45), "x"):
        for y_rot in range(45):
            for z_rot in range(45):
                main(False, MAX_FRAMES_NORMAL, SceneParameters((x_rot, y_rot, z_rot), 45.,
                                                               velocity_range=((-10, 10), (-10, 10), (-10, 0)), position_range=(1,1)))

    for x_rot in tqdm.tqdm(range(10), "x"):
        for y_rot in range(10):
            for z_rot in range(10):
                main(False, MAX_FRAMES_NORMAL, SceneParameters((x_rot, y_rot, z_rot), 10.,
                                                               velocity_range=((-10, 10), (-10, 10), (-1, 0)), position_range=(1,1)))

    for x_rot in tqdm.tqdm(range(20), "x"):
        for y_rot in range(20):
            for z_rot in range(20):
                main(False, MAX_FRAMES_NORMAL, SceneParameters((x_rot, y_rot, z_rot), 20.,
                                                               velocity_range=((-10, 10), (-10, 10), (-20, -5)), position_range=(1,1)))


    """
    for x_rot in tqdm.tqdm(range(45), "x"):
        for y_rot in range(45):
            for z_rot in range(45):
                main(False, MAX_FRAMES_NORMAL, [x_rot, y_rot,z_rot], True)
    """

