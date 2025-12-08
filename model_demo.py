import json
import time
import torch
import math
import pybullet as p

from model import ImpulsePredictor
from scene_creator import create_scene


def apply_force(contact_points,
                current_angular_vel, current_linear_vel,
                model,
                prev_angular_vel, prev_linear_vel,
                cube_id,
                plane_id,
                 timestep: float):
    # DEBUG: Check if function is even being called
    print(f"\n=== APPLY_FORCE CALLED: {len(contact_points)} contacts ===")

    # Wake up the object
    p.changeDynamics(cube_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)

    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
    cube_roll, cube_pitch, cube_yaw = p.getEulerFromQuaternion(cube_orn)
    collider_pos, collider_orientation = p.getBasePositionAndOrientation(plane_id)
    lin, ang = prev_linear_vel, prev_angular_vel

    # DEBUG: Print current state
    print(f"Cube position: {cube_pos}")
    print(f"Cube velocity: {lin}")

    # Check mass
    mass_info = p.getDynamicsInfo(cube_id, -1)
    mass = mass_info[0]
    print(f"Cube mass: {mass}")

    base_features = [
        lin[0], lin[1], lin[2],
        ang[0], ang[1], ang[2],
        cube_pos[0], cube_pos[1], cube_pos[2],
        collider_pos[0], collider_pos[1], collider_pos[2],
        math.sin(cube_roll), math.cos(cube_roll),
        math.sin(cube_pitch), math.cos(cube_pitch),
        math.sin(cube_yaw), math.cos(cube_yaw),
    ]

    point_features = []
    for cp in contact_points[:4]:
        cx, cy, cz = cp[5]
        point_features.extend([cx, cy, cz])

    num_points = len(contact_points)
    for _ in range(4 - num_points):
        point_features.extend([0, 0, 0])

    features = base_features + point_features
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(x)[0]

    print(f"Model output range: [{pred.min().item():.6f}, {pred.max().item():.6f}]")

    for i, cp in enumerate(contact_points):
        contact_point = cp[5]
        idx = i * 6
        linear_impulse = pred[idx:idx + 3].cpu().numpy()
        angular_impulse = pred[idx + 3:idx + 6].cpu().numpy()

        # Convert impulse to force
        force = linear_impulse
        torque = angular_impulse

        print(f"\nContact {i} at {contact_point}:")
        print(f"  Linear impulse: {linear_impulse}")
        print(f"  Force (impulse/dt): {force}")
        print(f"  Force magnitude: {(force[0] ** 2 + force[1] ** 2 + force[2] ** 2) ** 0.5:.6f}")

        # Apply forces
        p.applyExternalForce(
            objectUniqueId=cube_id,
            linkIndex=-1,
            forceObj=force.tolist(),
            posObj=contact_point,
            flags=p.WORLD_FRAME
        )

        p.applyExternalTorque(
            objectUniqueId=cube_id,
            linkIndex=-1,
            torqueObj=torque.tolist(),
            flags=p.WORLD_FRAME
        )


def main():
    model = ImpulsePredictor(30)
    model.load_state_dict(torch.load("best_impulse_model.pth", map_location="cpu"))
    model.eval()

    physics_client = p.connect(p.GUI)
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    collision_data = []
    frame = 0
    prev_linear_vel = [0, 0, 0]
    prev_angular_vel = [0, 0, 0]

    plane_id, sphere_id, timestep = create_scene(p)
    max_frames = 10000000

    # DEBUG: Print timestep
    print(f"Timestep: {timestep}")

    p.changeDynamics(sphere_id, -1,
                     contactProcessingThreshold=0,
                     restitution=0,
                     lateralFriction=0,
                     spinningFriction=0,
                     rollingFriction=0,
                     contactStiffness=0,
                     contactDamping=0
                     )
    p.setGravity(0, 0, -9.81)

    p.resetBaseVelocity(sphere_id, linearVelocity=[0, 1, -1])

    contact_detected = False

    while frame < max_frames:
        current_linear_vel, current_angular_vel = p.getBaseVelocity(sphere_id)

        # Get contact points
        contact_points = p.getContactPoints(bodyA=sphere_id, bodyB=plane_id)

        if contact_points:
            apply_force(contact_points, current_angular_vel, current_linear_vel,
                        model, prev_angular_vel, prev_linear_vel, sphere_id, plane_id, timestep)

            # DEBUG: Check velocity after applying forces
            vel_after, _ = p.getBaseVelocity(sphere_id)
            print(f"Velocity BEFORE step: {vel_after}")

        # Step simulation
        p.stepSimulation()

        if contact_points:
            vel_after_step, _ = p.getBaseVelocity(sphere_id)
            print(f"Velocity AFTER step: {vel_after_step}")

        prev_linear_vel = current_linear_vel
        prev_angular_vel = current_angular_vel

        frame += 1
        if connection_type == p.GUI:
            time.sleep(timestep)

    with open(f'logs/collision_data-{time.time()}.json', 'w') as f:
        json.dump(collision_data, f, indent=4)

    p.disconnect()


if __name__ == "__main__":
    main()
