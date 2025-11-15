import json
import math
import time
from typing import Any

import pybullet as p
import torch

from model import ComplexImpulsePredictor
from scene_creator import create_scene


def main():
    model = ComplexImpulsePredictor(12)
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
    # Give initial downward velocity (since gravity is off)
    p.resetBaseVelocity(sphere_id,
                        linearVelocity=[0, 0, 0])

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


def apply_force(contact_points, current_angular_vel, current_linear_vel, model: ComplexImpulsePredictor,
                prev_angular_vel: list[int] | Any, prev_linear_vel: list[int] | Any, sphere_id, timestep: float):
    cp = contact_points[0]

    # -------------------------------
    # 1. Build model input (18 dims)
    # -------------------------------
    # Example feature vector — adjust to match your training dataset
    # 1. Get velocities
    lin, ang = p.getBaseVelocity(sphere_id)

    # 2. Get orientation → Euler
    pos, orn = p.getBasePositionAndOrientation(sphere_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)

    # 3. Build SAME EXACT feature vector used in training
    features = [
        lin[0], lin[1], lin[2],
        ang[0], ang[1], ang[2],
        math.sin(roll), math.cos(roll),
        math.sin(pitch), math.cos(pitch),
        math.sin(yaw), math.cos(yaw),
    ]

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)[0]

    linear_impulse = pred[:3].numpy()
    angular_impulse = pred[3:].numpy()

    # -------------------------------
    # 2. Convert impulses → forces
    # -------------------------------
    force = linear_impulse
    torque = angular_impulse
    print("Force:", force)

    # -------------------------------
    # 3. Apply forces
    # -------------------------------
    pos, _ = p.getBasePositionAndOrientation(sphere_id)
    p.applyExternalForce(
        objectUniqueId=sphere_id,
        linkIndex=-1,
        forceObj=force,
        posObj=pos,
        flags=p.WORLD_FRAME
    )


if __name__ == "__main__":
    main()