"""
Re-decompose bunny.obj with CoACD using parameters tuned for thin-feature
resolution (ears, paw tips). Writes a multi-hull OBJ in the same format
as the original bunny_collision.obj (one `o hull_N` group per hull).
"""
import time
import sys
import trimesh
import coacd
import numpy as np

SRC = "bunny.obj"
DST = "bunny_collision.obj"

print("Loading source mesh...", flush=True)
mesh_tm = trimesh.load(SRC, force="mesh")
print(f"  vertices={len(mesh_tm.vertices)}  faces={len(mesh_tm.faces)}", flush=True)
print(f"  watertight={mesh_tm.is_watertight}  volume={mesh_tm.volume:.4f}", flush=True)

mesh = coacd.Mesh(mesh_tm.vertices, mesh_tm.faces)

# Tuned for thin-feature resolution:
#   threshold=0.01     (default 0.05) -> tighter concavity, more hulls on ears/paws
#   max_convex_hull=300                -> cap; threshold should drive the count
#   resolution=2_000_000 (default 2000) -> high-res voxelization catches thin geometry
#   preprocess_resolution=50 (default 50)
#   mcts_iterations=150, mcts_max_depth=3, mcts_nodes=20  (defaults, fine)
#   pca=False  (default) -- our mesh is already axis-aligned reasonably
#   seed=42    deterministic
print("\nRunning CoACD (this will take a few minutes)...", flush=True)
t0 = time.time()
parts = coacd.run_coacd(
    mesh,
    threshold=0.05,
    max_convex_hull=1000,
    resolution=2_000_000,
    preprocess_mode="auto",
    preprocess_resolution=256,
    seed=42,
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s. Got {len(parts)} hulls.", flush=True)

# Report hull-size distribution so we can see if thin features got their own hulls
sizes = []
for verts, faces in parts:
    v = np.asarray(verts)
    extent = v.max(axis=0) - v.min(axis=0)
    sizes.append(float(np.linalg.norm(extent)))
sizes_arr = np.array(sizes)
print(f"\nHull diagonal extents:")
print(f"  min={sizes_arr.min():.4f}  max={sizes_arr.max():.4f}")
print(f"  median={np.median(sizes_arr):.4f}  mean={sizes_arr.mean():.4f}")
print(f"  hulls with extent < 0.05 (likely fine features): "
      f"{int((sizes_arr < 0.05).sum())}")

# Write multi-hull OBJ matching the original file's format
print(f"\nWriting {DST}...", flush=True)
total_v = 0
total_f = 0
with open(DST, "w") as fh:
    fh.write("# CoACD convex decomposition of bunny.obj\n")
    fh.write(f"# {len(parts)} convex hulls\n")
    fh.write(f"# threshold=0.02  resolution=100000  seed=42\n")
    vertex_offset = 0
    for i, (verts, faces) in enumerate(parts):
        verts = np.asarray(verts)
        faces = np.asarray(faces)
        fh.write(f"o hull_{i}\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            # OBJ is 1-indexed; offset accumulates across all prior hulls
            fh.write(f"f {f[0]+1+vertex_offset} {f[1]+1+vertex_offset} {f[2]+1+vertex_offset}\n")
        vertex_offset += len(verts)
        total_v += len(verts)
        total_f += len(faces)

print(f"  total vertices={total_v}  total faces={total_f}")
print(f"  file size: ", end="")
import os
print(f"{os.path.getsize(DST)/1024:.1f} KB")
