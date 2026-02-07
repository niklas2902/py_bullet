import math
import random
from typing import Any

import pybullet_data

from utils import get_global_vertex_positions


def create_scene(p, should_use_gravity: bool = False) -> Any:
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

    # Create cube
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
        basePosition=[0, 0, 0.5],
        baseOrientation=random_quaternion()
    )

    global_vertex_positions = get_global_vertex_positions(cube_id)
    min_z = get_min_z(global_vertex_positions)
    pos, orn = p.getBasePositionAndOrientation(cube_id)
    p.resetBasePositionAndOrientation(cube_id, [pos[0], pos[1], pos[2] + min_z], orn)


    # Make cube bouncy
    p.changeDynamics(cube_id, -1, restitution=0.5)

    # **DISABLE COLLISION RESPONSE but KEEP COLLISION DETECTION**
    p.setCollisionFilterPair(plane_id, cube_id, -1, -1, enableCollision=1)
    # Set contact processing to only report contacts, not resolve them
    p.changeDynamics(plane_id, -1, contactProcessingThreshold=0)
    p.changeDynamics(cube_id, -1, contactProcessingThreshold=0)

    # Make both objects have zero contact stiffness and damping
    p.changeDynamics(plane_id, -1, contactStiffness=0, contactDamping=0)
    p.changeDynamics(cube_id, -1, contactStiffness=0, contactDamping=0)

    # Give initial downward velocity (since gravity is off)
    p.resetBaseVelocity(cube_id,
                        linearVelocity=[random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 0)],
                        angularVelocity=random_angular_velocity())

    timestep = 1.0 / 240
    p.setTimeStep(timestep)
    return plane_id, cube_id, timestep


def get_min_z(global_vertex_positions):
    z = 999
    for vertex in global_vertex_positions:
        z = min(z, vertex[2])
    return z

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
