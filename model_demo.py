import json
import math
import time
from typing import Any

import pybullet as p
import torch

from model import ImpulsePredictor
from scene_creator import create_scene


def main():
    model = ImpulsePredictor(21)
    model.load_state_dict(torch.load("best_impulse_model.pth", map_location="cpu"))
    model.eval()
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    # Check connection type
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    collision_data = []
    frame = 0
    prev_linear_vel = [0, 0, 0]
    prev_angular_vel = [0, 0, 0]

    max_frames, plane_id,  sphere_id, timestep = create_scene(p)
    max_frames = 10000000
    p.changeDynamics(sphere_id, -1,
                     contactProcessingThreshold=0,  # disables solver response
                     restitution=0,
                     lateralFriction=0,
                     spinningFriction=0,
                     rollingFriction=0,
                     contactStiffness=0,
                     contactDamping=0
                     )
    p.setGravity(0, 0, -9.81)

    p.resetBaseVelocity(sphere_id,
                        linearVelocity=[0, 1, -1])

    while frame < max_frames:
        # Store velocities before simulation step
        current_linear_vel, current_angular_vel = p.getBaseVelocity(sphere_id)

        # Step simulation
        p.stepSimulation()

        # Get contact points
        contact_points = p.getContactPoints(bodyA=sphere_id, bodyB=plane_id)

        if contact_points:
            apply_force(contact_points, current_angular_vel, current_linear_vel, model, prev_angular_vel,
                        prev_linear_vel, sphere_id, timestep)

        # Update previous velocities
        prev_linear_vel = current_linear_vel
        prev_angular_vel = current_angular_vel

        frame += 1
        if connection_type == p.GUI:#
            time.sleep(timestep)

    # Save collision data to JSON file
    with open(f'logs/collision_data-{time.time()}.json', 'w') as f:
        json.dump(collision_data, f, indent=4)


    p.disconnect()


def apply_force(contact_points, current_angular_vel, current_linear_vel, model: ImpulsePredictor,
                prev_angular_vel, prev_linear_vel, cube_id, timestep: float):

    for cp in contact_points:

        # --------------------
        # 0. Extract positions
        # --------------------
        cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
        cube_roll, cube_pitch, cube_yaw = p.getEulerFromQuaternion(cube_orn)

        collider_id = cp[2]  # bodyUniqueIdB
        collider_pos, collider_orn = p.getBasePositionAndOrientation(collider_id)
        col_roll, col_pitch, col_yaw = p.getEulerFromQuaternion(collider_orn)

        # Contact point in world space
        contact_point = cp[5]  # = contactPositionOnA
        cx, cy, cz = contact_point

        # -------------------------------
        # 1. Build model input (ADD POS)
        # -------------------------------

        # 1. cube velocity
        lin, ang = p.getBaseVelocity(cube_id)

        # Final feature vector (example):
        features = [
            # cube linear vel
            lin[0], lin[1], lin[2],

            # cube angular vel
            ang[0], ang[1], ang[2],
            cx, cy, cz,
            cube_pos[0], cube_pos[1], cube_pos[2],
            collider_pos[0], collider_pos[1], collider_pos[2],


            # cube orientation (encoded)
            math.sin(cube_roll), math.cos(cube_roll),
            math.sin(cube_pitch), math.cos(cube_pitch),
            math.sin(cube_yaw), math.cos(cube_yaw),
        ]

        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        print(x.shape)

        with torch.no_grad():
            pred = model(x)[0]

        linear_impulse = pred[:3].numpy()

        # Force = impulse (per step)
        force = linear_impulse

        # -------------------------------
        # 3. Apply forces at the contact
        # -------------------------------
        p.applyExternalForce(
            objectUniqueId=cube_id,
            linkIndex=-1,
            forceObj=force,
            posObj=contact_point,   # world position
            flags=p.WORLD_FRAME
        )




if __name__ == "__main__":
    main()