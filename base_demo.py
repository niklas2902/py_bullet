import math
import time

import pybullet as p
import numpy as np
from parameters import SceneParameters
from scene_creator import create_scene
from own_physics import calculate_force, _contact_point_velocity

MAX_FRAMES = 2000

REST_LINEAR_DAMPING = 10.0  # N·s/m  — opposes linear velocity while in contact
REST_ANGULAR_DAMPING = 1.2  # N·m·s/rad — opposes angular velocity while in contact
FORCE_CLAMPING_START = 9.9
TORQUE_CLAMPING_START = 0.1

def apply_impulse_predictor(cube_id, current_linear_vel, current_angular_vel, contact_points):
    """
    Net-force / net-torque application: sum all contact contributions, then make
    a single applyExternalForce + applyExternalTorque call.
    """
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    cube_pos = np.array(cube_pos)

    n_contacts = len(contact_points)
    if n_contacts == 0:
        return

    net_force = np.zeros(3)
    net_torque = np.zeros(3)

    for cp in contact_points:
        pen = cp[8]
        normal = cp[7]
        contact_pos_world = np.array(cp[5])

        v_contact = _contact_point_velocity(
            cube_id, contact_pos_world, current_linear_vel, current_angular_vel
        )

        force_vector = np.asarray(calculate_force(normal, pen, cube_id, v_contact.tolist()))
        force_vector = force_vector / n_contacts

        net_force += force_vector
        r = contact_pos_world - cube_pos
        net_torque += np.cross(r, force_vector)
    print(np.linalg.norm(net_force))
    #if(np.linalg.norm(net_force) < FORCE_CLAMPING_START):
    #    net_force  += -REST_LINEAR_DAMPING  * np.array(current_linear_vel)
    
    #if(np.linalg.norm(net_torque) < TORQUE_CLAMPING_START):
    #    net_torque += -REST_ANGULAR_DAMPING * np.array(current_angular_vel)

    p.applyExternalForce(cube_id, -1, net_force.tolist(), cube_pos.tolist(), p.WORLD_FRAME)
    p.applyExternalTorque(cube_id, -1, net_torque.tolist(), p.WORLD_FRAME)


def _disable_default_contact_response(body_id):
    """
    Make PyBullet's built-in contact solver effectively a no-op for this body,
    so our custom spring force is the only normal response.
    """
    for link in range(-1, p.getNumJoints(body_id)):
        p.changeDynamics(
            body_id, link,
            restitution=0.0,
            lateralFriction=0.0,
            contactStiffness=1e-9,
            contactDamping=1e-9,
        )


def main():
    physics_client = p.connect(p.GUI)
    connection_type = p.getConnectionInfo(physics_client)['connectionMethod']

    p.setTimeStep(1.0 / 240.0)
    p.setPhysicsEngineParameter(numSolverIterations=150)

    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "collision_run_gt.mp4")

    plane_id, cube_id, timestep = create_scene(p, False, SceneParameters(random_rotation=True, velocity_range=((0,0), (0,0), (-3,-5))))

    _disable_default_contact_response(cube_id)
    _disable_default_contact_response(plane_id)

    frame = 0
    while frame < MAX_FRAMES:
        p.stepSimulation()

        current_linear_vel, current_angular_vel = p.getBaseVelocity(cube_id)
        contact_points = p.getContactPoints(bodyA=cube_id, bodyB=plane_id)

        if contact_points:
            apply_impulse_predictor(
                cube_id, current_linear_vel, current_angular_vel, contact_points
            )

            contact_points_plane = p.getContactPoints(bodyA=plane_id, bodyB=cube_id)
            current_linear_vel_plane, current_angular_vel_plane = p.getBaseVelocity(plane_id)
            apply_impulse_predictor(
                plane_id, current_linear_vel_plane, current_angular_vel_plane, contact_points_plane
            )



        if frame % 30 == 0:
            pos, _ = p.getBasePositionAndOrientation(cube_id)
            max_pen = max((c[8] for c in contact_points), default=0.0)
            print(f"f, mass=0={frame:4d} z={pos[2]:+.3f} vz={current_linear_vel[2]:+.3f} "
                  f"n={len(contact_points):2d} pen={max_pen:.4f}")

        frame += 1
        if connection_type == p.GUI:
            time.sleep(timestep)

    p.stopStateLogging(log_id)
    p.disconnect()


if __name__ == "__main__":
    main()
