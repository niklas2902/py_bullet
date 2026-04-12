import time

import torch
import numpy as np
import pybullet as p

from models.gnn_model import GNNCollisionPredictor, build_cube_edges
from parameters import SceneParameters
from scene_creator import create_scene

MAX_FRAMES = 5000

# ------------------------------------------------------------------
# Load model + stats
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GNNCollisionPredictor(
    node_dim=128,
    edge_dim=64,
    gnn_layers=4,
    head_width=128,
    num_heads=4,
    max_contacts=4,
    dropout=0.0,           # no dropout at inference
    fully_connected=True,
    plane_normal=(0.0, 1.0, 0.0),
    plane_offset=0.0,
).to(device)

checkpoint = torch.load("checkpoints/best_gnn_collision.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Target normalization stats (saved during training)
target_stats = checkpoint.get("target_stats")

# Edge index (built once, reused every frame)
edge_index = build_cube_edges(fully_connected=True).to(device)


# ------------------------------------------------------------------
# Inference helper
# ------------------------------------------------------------------
def apply_impulse_predictor(cube_id, relative_pos, relative_euler):
    start = time.time()

    # Build input tensors — just position + euler, batch size 1
    position = torch.tensor([[0.0, 0.0, relative_pos[2]]], dtype=torch.float32, device=device)
    euler = torch.tensor([list(relative_euler)], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(
            position=position,
            euler=euler,
            edge_index=edge_index,
        )

    # Count prediction
    num_contacts_pred = out["num_contacts"].argmax(dim=-1).item()

    # Contact points and impulses: [1, 4, 3] → [4, 3]
    contact_points = out["contact_points"].squeeze(0)
    impulses = out["impulses"].squeeze(0)

    # Denormalize back to world units
    if target_stats is not None:
        cp_mean = target_stats["cp_mean"].to(device)
        cp_std = target_stats["cp_std"].to(device)
        imp_mean = target_stats["imp_mean"].to(device)
        imp_std = target_stats["imp_std"].to(device)

        contact_points = contact_points * cp_std + cp_mean
        impulses = impulses * imp_std + imp_mean

    contact_points = contact_points.cpu().numpy()  # [4, 3]
    impulses = impulses.cpu().numpy()                # [4, 3]


    # Apply forces at predicted contact points
    for i in range(num_contacts_pred):
        predicted_point = contact_points[i] + np.array(relative_pos)
        predicted_force = impulses[i]

        p.applyExternalForce(
            cube_id, -1,
            predicted_force.tolist(),
            predicted_point.tolist(),
            p.WORLD_FRAME,
        )


# ------------------------------------------------------------------
# Main simulation loop
# ------------------------------------------------------------------
def main():
    physics_client = p.connect(p.GUI)
    connection_type = p.getConnectionInfo(physics_client)["connectionMethod"]

    frame = 0
    
    plane_id, cube_id, timestep = create_scene(p, True, SceneParameters(random_rotation = True))

    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "collision_run.mp4")

    # Disable collisions (we predict contact forces ourselves)
    p.setCollisionFilterGroupMask(plane_id, -1, collisionFilterGroup=1, collisionFilterMask=0)
    p.setCollisionFilterGroupMask(cube_id, -1, collisionFilterGroup=1, collisionFilterMask=0)

    p.resetBaseVelocity(cube_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    initial_orientation = p.getQuaternionFromEuler([0.0, 0.5, 0.0])
    p.resetBasePositionAndOrientation(
        cube_id,
        p.getBasePositionAndOrientation(cube_id)[0],
        initial_orientation,
    )

    while frame < MAX_FRAMES:
        start_time = time.time() * 1000

        # Get positions and orientations
        plane_pos, plane_orn = p.getBasePositionAndOrientation(plane_id)
        cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)

        plane_pos = np.array(plane_pos)
        cube_pos = np.array(cube_pos)

        # Relative position in plane's frame
        plane_rot_matrix = np.array(p.getMatrixFromQuaternion(plane_orn)).reshape(3, 3)
        relative_pos = plane_rot_matrix.T @ (cube_pos - plane_pos)

        # Relative rotation
        plane_orn_inv = p.invertTransform([0, 0, 0], plane_orn)[1]
        relative_quat = p.multiplyTransforms([0, 0, 0], plane_orn_inv, [0, 0, 0], cube_orn)[1]
        relative_euler = p.getEulerFromQuaternion(relative_quat)

        # Predict and apply forces
        apply_impulse_predictor(cube_id, relative_pos, relative_euler)

        p.stepSimulation()

        frame += 1
        if connection_type == p.GUI:
            elapsed_ms = time.time() * 1000 - start_time
            sleep_time = max(0, timestep - elapsed_ms / 1000)
            time.sleep(sleep_time)

    p.stopStateLogging(log_id)
    p.disconnect()


if __name__ == "__main__":
    main()