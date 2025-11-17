import random
from typing import Any

import pybullet_data


def create_scene(p, should_use_gravity:bool = False) -> Any :
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # No gravity
    if should_use_gravity:
        p.setGravity(0, 0, -9.81)
    else:
        p.setGravity(0, 0, 0)

    # Load plane
    plane_id = p.loadURDF("plane.urdf")

    # Make the plane bouncy
    p.changeDynamics(plane_id, -1, restitution=0.9)

    # Create sphere
    sphere_radius = 0.3
    col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
    vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius,
                                 rgbaColor=[0.2, 0.2, 0.8, 1.0])

    sphere_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0, 0, 2.0],
        baseOrientation=[0, 0, 0, 1]
    )

    # Make sphere bouncy
    p.changeDynamics(sphere_id, -1, restitution=0.9)

    # Give initial downward velocity (since gravity is off)
    p.resetBaseVelocity(sphere_id,
                        linearVelocity=[random.uniform(10, -10), random.uniform(10, -10), random.uniform(-1, -10)])

    timestep = 1.0 / 240.0
    p.setTimeStep(timestep)

    max_frames = 1000  # Run for limited frames for testing
    return  max_frames, plane_id, sphere_id, timestep
