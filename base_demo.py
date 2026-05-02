import time

import pybullet as p
import numpy as np
from parameters import SceneParameters
from scene_creator import create_scene
from own_physics import calculate_force, _contact_point_velocity

MAX_FRAMES = 2000


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

    plane_id, cube_id, timestep = create_scene(p, True, SceneParameters(random_rotation=True))

    _disable_default_contact_response(cube_id)
    _disable_default_contact_response(plane_id)

    p.resetBaseVelocity(cube_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    initial_orientation = p.getQuaternionFromEuler([0.0, 0.5, 0.0])
    p.resetBasePositionAndOrientation(
        cube_id,
        p.getBasePositionAndOrientation(cube_id)[0],
        initial_orientation,
    )

    frame = 0
    while frame < MAX_FRAMES:
        p.stepSimulation()

        current_linear_vel, current_angular_vel = p.getBaseVelocity(cube_id)
        contact_points = p.getContactPoints(bodyA=cube_id, bodyB=plane_id)

        if contact_points:
            apply_impulse_predictor(
                cube_id, current_linear_vel, current_angular_vel, contact_points
            )

        if frame % 30 == 0:
            pos, _ = p.getBasePositionAndOrientation(cube_id)
            max_pen = max((c[8] for c in contact_points), default=0.0)
            print(f"f={frame:4d} z={pos[2]:+.3f} vz={current_linear_vel[2]:+.3f} "
                  f"n={len(contact_points):2d} pen={max_pen:.4f}")

        frame += 1
        if connection_type == p.GUI:
            time.sleep(timestep)

    p.stopStateLogging(log_id)
    p.disconnect()


if __name__ == "__main__":
    main()
