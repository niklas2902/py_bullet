import math
import random
from typing import Any

import numpy as np
import pybullet_data

from parameters import SceneParameters
from utils import get_global_vertex_positions


def create_scene(p, should_use_gravity: bool = False, parameters:SceneParameters = SceneParameters()) -> Any:
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # No gravity
    if should_use_gravity:
        p.setGravity(0, 0, -9.81)
    else:
        p.setGravity(0, 0, 0)

    # Load plane
    plane_id = p.loadURDF(
        "/home/niklas/Documents/privat/repositories/py_bullet/blender_models/bunny.urdf",
        basePosition=[0, 0, 0.5],
        baseOrientation=random_quaternion()
    )


    # Make the plane bouncy
    p.changeDynamics(plane_id, -1, restitution=0.5)

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

    #cube_id = p.createMultiBody(
    #    baseMass=1.0,
    #    baseCollisionShapeIndex=col_id,
    #    baseVisualShapeIndex=vis_id,
    #    basePosition=[0, 0, 0.5],
    #    baseOrientation=random_quaternion()
    #)

    sphere_radius = parameters.spawn_radius
    spawn_angles = parameters.spawn_angles
    theta = spawn_angles.theta * (math.pi / 180)
    phi = spawn_angles.phi * (math.pi / 180)
    x = sphere_radius * math.sin(phi) * math.cos(theta)
    y = sphere_radius * math.sin(phi) * math.sin(theta)
    z = sphere_radius * math.cos(phi)
    spawn_point = np.array([x,y,z])

    cube_id = p.loadURDF(
        "/home/niklas/Documents/privat/repositories/py_bullet/blender_models/bunny.urdf",
        basePosition=[x, y, z],
        baseOrientation=random_quaternion()
    )

    random_rotation_and_position(cube_id, plane_id, p, parameters)

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
                        linearVelocity=np.array([0,0,random.uniform(-sphere_radius, sphere_radius)]) - spawn_point,
                        angularVelocity=random_angular_velocity())
    p.resetBaseVelocity(plane_id, linearVelocity=[0,0,0], angularVelocity=random_angular_velocity())

    timestep = 1.0 / 240
    p.setTimeStep(timestep)
    return plane_id, cube_id, timestep


def reset_scene(p, cube_id,  plane_id, should_use_gravity: bool, parameters: SceneParameters):
    if should_use_gravity:
        p.setGravity(0, 0, -9.81)
    else:
        p.setGravity(0, 0, 0)
    random_rotation_and_position(cube_id, plane_id , p, parameters)
    p.resetBaseVelocity(cube_id,
                        linearVelocity=[random.uniform(parameters.velocity_range[0][0], parameters.velocity_range[0][1]),
                                        random.uniform(parameters.velocity_range[1][0], parameters.velocity_range[1][1]),
                                        random.uniform(parameters.velocity_range[2][0], parameters.velocity_range[2][1])],
                        angularVelocity=random_angular_velocity())

def normalized(v):
    normalized_v = v / np.sqrt(np.sum(v**2))
    return normalized_v
def random_rotation_and_position(cube_id, plane_id, p, parameters:SceneParameters):
        
    pos, rot = p.getBasePositionAndOrientation(cube_id)
    if parameters.random_rotation:
        rot = rot
    else:
        rot =  p.getQuaternionFromEuler([math.radians(parameters.rotation_parts[0] / parameters.rotation_fidelity * 360. + parameters.offset ),
                                        math.radians(parameters.rotation_parts[1] / parameters.rotation_fidelity * 360. + parameters.offset ),
                                        math.radians(parameters.rotation_parts[2] / parameters.rotation_fidelity * 360. + parameters.offset)])
    p.resetBasePositionAndOrientation(cube_id, pos, rot)
    pts = p.getClosestPoints(bodyA=cube_id, bodyB=plane_id, distance=3)
    nearest = min(pts, key=lambda c: c[8]) if pts else None
    dist = (np.array(nearest[6]) - np.array(nearest[5]))
    normalized_dist = np.array([0,0,0]) # normalized(dist)

    if parameters.random_rotation:
        p.resetBasePositionAndOrientation(cube_id, np.array(pos) + dist - normalized_dist * 0.1, rot)
    else:
        p.resetBasePositionAndOrientation(cube_id, np.array(pos) + dist - normalized_dist * 0.1, rot)


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
