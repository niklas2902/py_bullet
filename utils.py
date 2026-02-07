# Convert quaternion to rotation matrix
import numpy as np
import pybullet as p

def get_global_vertex_positions(cube_id):
    pos, orn = p.getBasePositionAndOrientation(cube_id)
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    # Define local vertex positions for a unit cube centered at origin
    # Adjust the size if your cube has different dimensions
    half_extents = [0.5, 0.5, 0.5]  # for a 1x1x1 cube
    local_vertices = np.array([
        [-half_extents[0], -half_extents[1], -half_extents[2]],
        [ half_extents[0], -half_extents[1], -half_extents[2]],
        [ half_extents[0],  half_extents[1], -half_extents[2]],
        [-half_extents[0],  half_extents[1], -half_extents[2]],
        [-half_extents[0], -half_extents[1],  half_extents[2]],
        [ half_extents[0], -half_extents[1],  half_extents[2]],
        [ half_extents[0],  half_extents[1],  half_extents[2]],
        [-half_extents[0],  half_extents[1],  half_extents[2]],
    ])

    # Transform vertices to global coordinates
    global_vertices = []
    for vertex in local_vertices:
        # Rotate then translate
        global_vertex = rot_matrix @ vertex + np.array(pos)
        global_vertices.append(global_vertex)

    global_vertices = np.array(global_vertices)
    return global_vertices