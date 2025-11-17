import math
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
    cube_size = 0.5  # full edge length of the cube
    half = cube_size / 2

    col_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half, half, half]
    )
    vis_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half, half, half],
        rgbaColor=[0.8, 0.2, 0.2, 1]
    )

    cube_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0, 0, 2.0],
        baseOrientation=random_quaternion()
    )

    # Make sphere bouncy
    p.changeDynamics(cube_id, -1, restitution=0.9)

    # Give initial downward velocity (since gravity is off)
    p.resetBaseVelocity(cube_id,
                        linearVelocity=[random.uniform(10, -10), random.uniform(10, -10), random.uniform(-1, -10)],
                        angularVelocity=random_angular_velocity())

    timestep = 1.0 / 240.0
    p.setTimeStep(timestep)

    max_frames = 1000  # Run for limited frames for testing
    return  max_frames, plane_id, cube_id, timestep


def random_angular_velocity(strength=5.0):
    return [
        (random.random() * 2 - 1) * strength,
        (random.random() * 2 - 1) * strength,
        (random.random() * 2 - 1) * strength,
    ]


def random_quaternion():
    # Uniform random quaternion
    u1 = random.random()
    u2 = random.random()
    u3 = random.random()

    q = [
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1)     * math.sin(2 * math.pi * u3),
        math.sqrt(u1)     * math.cos(2 * math.pi * u3),
    ]
    return q
