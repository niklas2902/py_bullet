import json
import multiprocessing
import os
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
from scene_creator import create_scene, reset_scene, random_rotation_an_position

SPRING_CONSTANT = 1000  # N/m
BOUNCINESS_FACTOR = 0.5
MAX_RUNS = 100000
GRAVITY_RUNS = 0
MAX_FRAMES_GRAVITY = 0
MAX_FRAMES_NORMAL = 500
MAX_FRAMES_TO_RECORD = 100
def _contact_point_velocity(cube_id, contact_pos_world, v_com, omega):
    """Velocity at the contact point: v_com + omega × r."""
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    r = np.array(contact_pos_world, dtype=np.float64) - np.array(cube_pos, dtype=np.float64)
    return np.array(v_com, dtype=np.float64) + np.cross(np.array(omega, dtype=np.float64), r)


def apply_force(contact_points,
                current_angular_vel, current_linear_vel,
                model,
                prev_angular_vel, prev_linear_vel,
                cube_id,
                plane_id,
                 timestep: float):

        for cp in contact_points:
            apply_spring_force(cp[7], cp[8], cp[5], cube_id, current_linear_vel, current_angular_vel)


def apply_spring_force(normal, penetration, position, cube_id, current_linear_vel, current_angular_vel):
    v_contact = _contact_point_velocity(cube_id, position, current_linear_vel, current_angular_vel)
    force_vector = calculate_force(normal, penetration, cube_id, v_contact.tolist())
    p.applyExternalForce(cube_id, -1, force_vector.tolist(), position, p.WORLD_FRAME)


def calculate_force(normal, penetration, cube_id, current_linear_vel) -> np.ndarray[Any, np.dtype[Any]] | Any:
    mass = p.getDynamicsInfo(cube_id, -1)[0]
    k = SPRING_CONSTANT
    
    # Clamp mass to avoid sqrt(0) or sqrt(negative) in damping coefficient
    mass = max(mass, 1e-6)
    
    # Clamp penetration to non-positive values — positive penetration is physically meaningless
    # and can cause explosive forces if penetration somehow goes positive
    penetration = min(penetration, 0.0)
    
    c = 2.0 * math.sqrt(k * mass) * BOUNCINESS_FACTOR  # critical damping

    v = np.array(current_linear_vel, dtype=np.float64)
    n = np.array(normal, dtype=np.float64)
    
    # Normalize the contact normal defensively — physics engines can return
    # slightly non-unit normals, which scales forces incorrectly
    n_mag = np.linalg.norm(n)
    if n_mag < 1e-8:
        return np.zeros(3, dtype=np.float64)  # degenerate normal, no force
    n = n / n_mag

    vel_normal = np.dot(v, n)
    
    spring_force = -k * penetration
    damping_force = -c * vel_normal
    force_mag = spring_force + damping_force
    
    # Prevent damping from reversing the spring force direction entirely —
    # this can cause objects to be sucked into surfaces
    if force_mag < 0.0:
        force_mag = 0.0
    
    force_vector = force_mag * n
    return force_vector

def empty_collisions(scene_parameters:SceneParameters):

    physics_client = p.connect(p.DIRECT)
    # Check connection type
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']
    plane_id,  cube_id, timestep = create_scene(p, False, scene_parameters)
    collision_data_empty = []

    for i in tqdm.tqdm(range(10000), "empty collisions"):
        simulate_empty_collisions(p, cube_id, plane_id, collision_data_empty)

    with open(f"logs/collision_points_{time.time()}.json", 'w') as f:
        json.dump(collision_data_empty, f, indent=4)
    p.disconnect()


def simulate_empty_collisions(p, cube_id, plane_id, collision_data_empty):
    # Connect to PyBullet
    pos, orn = p.getBasePositionAndOrientation(cube_id)
    orn = p.getQuaternionFromEuler([
        random.uniform(0, 2*math.pi),
        random.uniform(0, 2*math.pi),
        random.uniform(0, 2*math.pi)
    ])
    p.resetBasePositionAndOrientation(cube_id, [pos[0], pos[1], random.uniform(1, 10)], orn)

    p.resetBasePositionAndOrientation(cube_id, pos, orn)

    p.resetBaseVelocity(
        cube_id,
        linearVelocity=[
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(0, -10)
        ],
        angularVelocity=[
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10)
        ])
    frame = 0

    collision_data_empty = []
    while frame < 2:
        # Store velocities before simulation step
        current_linear_vel, current_angular_vel = p.getBaseVelocity(cube_id)

        # Step simulation
        p.stepSimulation()


        # Get contact points
        record_collision_empty(p, plane_id, cube_id, collision_data_empty, current_linear_vel=current_linear_vel)
        frame += 1

def main(should_use_gravity:bool, max_frames:int, parameters: SceneParameters, _physics_client = None, plane_id=None, cube_id=None):
    if _physics_client is None:
        physics_client = p.connect(p.DIRECT)
    else:
        physics_client = _physics_client
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    collision_data = []
    collision_point_data = []
    collision_point_data_empty = []
    frame = 0
    prev_linear_vel = [0, 0, 0]
    prev_angular_vel = [0, 0, 0]

    owns_bodies = plane_id is None
    if owns_bodies:
        plane_id, cube_id, timestep = create_scene(p, should_use_gravity, parameters)
    else:
        timestep = 1.0 / 240
        reset_scene(p, cube_id, should_use_gravity, parameters)

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
    num_collision_points = min(MAX_FRAMES_TO_RECORD, len(collision_point_data))
    random.shuffle(collision_point_data_empty)
    random.shuffle(collision_point_data)
    collision_point_data = collision_point_data[:MAX_FRAMES_TO_RECORD]
    for i in range(num_collision_points):
        if i < len(collision_point_data_empty):
            collision_point_data.append(collision_point_data_empty[i])
    if _physics_client is None:
        with open(f"logs/collision_points_{time.time()}-{os.getpid()}.json", 'w') as f:
            json.dump(collision_point_data, f, indent=4)
        with open(f'logs/collision_data-{time.time()}-{os.getpid()}.json', 'w') as f:
            random.shuffle(collision_data)
            json.dump(collision_data[:MAX_FRAMES_TO_RECORD], f, indent=4)
        p.disconnect()
    elif owns_bodies:
        p.removeBody(plane_id)
        p.removeBody(cube_id)
    return collision_point_data if collision_point_data else []
        


def simulate_sections(value):
    x_rot, sections = value
    physics_client = p.connect(p.DIRECT)
    print(f"x_rot:", x_rot)
    plane_id, cube_id, _ = create_scene(p, False, SceneParameters(random_rotation=True))
    point_data = []

    for y_rot in range(sections):
        print(f"{x_rot} - 1. y_rot: {y_rot}")
        for z_rot in range(sections):
            for vel in range(20):
                point_data += main(False, MAX_FRAMES_NORMAL,
                                   SceneParameters((x_rot, y_rot, z_rot), sections, velocity_range=((-10, 10), (-10, 10), (-vel * 0.5, -vel * 0.5 +1)), position_range=(0,0), offset = 0),
                                   _physics_client=physics_client, plane_id=plane_id, cube_id=cube_id)
    with open(f"logs/collision_points_{time.time()}-{os.getpid()}.json", 'w') as f:
        json.dump(point_data, f, indent=4)
    point_data = []
    for y_rot in range(sections):
        print(f"{x_rot} - 2. y_rot: {y_rot}")
        for z_rot in range(sections):
            for vel in range(10):
                point_data += main(False, MAX_FRAMES_NORMAL, SceneParameters((x_rot, y_rot, z_rot), sections,
                                                            velocity_range=((-10, 10), (-10, 10), (-vel * 0.1-0.01, 0)), position_range=(0,0), offset = 2.5),
                                                            _physics_client=physics_client, plane_id=plane_id, cube_id=cube_id)
    with open(f"logs/collision_points_{time.time()}-{os.getpid()}.json", 'w') as f:
        json.dump(point_data, f, indent=4)
    point_data = []
    for y_rot in range(sections):
        print(f"{x_rot} - 3. y_rot: {y_rot}")
        for z_rot in range(sections):
            for vel in range(-10,0):
                point_data += main(False, MAX_FRAMES_NORMAL, SceneParameters((x_rot, y_rot, z_rot), sections,
                                                            velocity_range=((-10, 10), (-10, 10), (-vel *0.25, -vel *0.25 - 0.1)), position_range=(0,0), offset = 5),
                                                            _physics_client=physics_client, plane_id=plane_id, cube_id=cube_id)
    with open(f"logs/collision_points_{time.time()}-{os.getpid()}.json", 'w') as f:
        json.dump(point_data, f, indent=4)
    point_data = []
    for y_rot in range(sections):
        print(f"{x_rot} - 4. y_rot: {y_rot}")
        for z_rot in range(sections):
            for vel in range(-20, -5):
                point_data += main(False, MAX_FRAMES_NORMAL, SceneParameters((x_rot, y_rot, z_rot), sections,
                                                            velocity_range=((-10, 10), (-10, 10), (vel, vel +1)), position_range=(0,0), offset = 7.5),
                                                            _physics_client=physics_client, plane_id=plane_id, cube_id=cube_id)
    with open(f"logs/collision_points_{time.time()}-{os.getpid()}.json", 'w') as f:
        json.dump(point_data, f, indent=4)
    point_data = []

    for y_rot in range(sections):
        print(f"{x_rot} - 5. y_rot: {y_rot}")
        for z_rot in range(sections):
            for vel in range(-20, -5):
                point_data += main(False, MAX_FRAMES_NORMAL, SceneParameters((x_rot, y_rot, z_rot), sections,
                                                            velocity_range=((-10, 10), (-10, 10), (vel / 100., (vel +1) / 1000.)), position_range=(0,0), offset = 8),
                                                            _physics_client=physics_client, plane_id=plane_id, cube_id=cube_id)
    with open(f"logs/collision_points_{time.time()}-{os.getpid()}.json", 'w') as f:
        json.dump(point_data, f, indent=4)
    point_data = []

    for y_rot in range(sections):
        print(f"{x_rot} - 6. y_rot: {y_rot}")
        for z_rot in range(sections):
            for vel in range(20):
                point_data += main(False, MAX_FRAMES_NORMAL, SceneParameters((x_rot, y_rot, z_rot), sections,
                                                            velocity_range=((-10, 10), (-10, 10), (-vel * 0.5, -vel * 0.5 +1)), position_range=(-vel / 240 * 100,-vel / 240 * 15), offset = 0),
                                                            _physics_client=physics_client, plane_id=plane_id, cube_id=cube_id)
    with open(f"logs/collision_points_{time.time()}-{os.getpid()}.json", 'w') as f:
        json.dump(point_data, f, indent=4)

    p.removeBody(plane_id)
    p.removeBody(cube_id)
    p.disconnect(physics_client)


if __name__ == "__main__":
    sections = 25
    empty_collisions(SceneParameters(random_rotation = True))
    for i in tqdm.tqdm(range(1000), "gravity runs"):
        main(True, 5000, SceneParameters(random_rotation=True))

    with multiprocessing.Pool(processes=sections) as pool:
        ans = pool.map(simulate_sections, [(x_rot, sections) for x_rot in range(sections)])
    #for x_rot in tqdm.tqdm(range(45), "x"):
    #    for y_rot in range(45):
    #        for z_rot in range(45):
    #            main(False, MAX_FRAMES_NORMAL, [x_rot, y_rot,z_rot], True)

