import json
import time

import pybullet as p
import tqdm

from recorder import record_collision
from scene_creator import create_scene


def main():
    # Connect to PyBullet
    physics_client = p.connect(p.DIRECT)
    # Check connection type
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    collision_data = []
    frame = 0
    prev_linear_vel = [0, 0, 0]
    prev_angular_vel = [0, 0, 0]

    max_frames, plane_id,  sphere_id, timestep = create_scene(p)

    while frame < max_frames:
        # Store velocities before simulation step
        current_linear_vel, current_angular_vel = p.getBaseVelocity(sphere_id)

        # Step simulation
        p.stepSimulation()

        # Get contact points
        contact_points = p.getContactPoints(bodyA=sphere_id, bodyB=plane_id)

        if contact_points:
            record_collision(p, collision_data, contact_points, frame, plane_id, prev_angular_vel, prev_linear_vel,
                             sphere_id)

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

if __name__ == "__main__":
    for i in tqdm.tqdm(range(1000000), "Simulation runs:"):
        main()