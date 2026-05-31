"""
Benchmark: Collision Resolution & Force Computation
====================================================
Measures wall-clock time for:
  1. Contact point lookup (p.getContactPoints)
  2. Force computation    (calculate_force only, summed over contacts)
  3. Force application    (applyExternalForce + applyExternalTorque)
  4. Full resolution      (compute + apply for all contacts)
  5. Full frame           (step + query + resolve)

Mirrors the physics of base_demo.py: step first, then read contacts and
apply net force + torque using _contact_point_velocity (angular velocity
included). PyBullet's built-in contact response is disabled.

Usage:
    python benchmark_collision.py [--frames 2000] [--warmup 50] [--repeats 3] [--target-fps 60]
"""

import argparse
import statistics
import time
from typing import List

import numpy as np
import pybullet as p

from own_physics import calculate_force, _contact_point_velocity
from parameters import SceneParameters
from scene_creator import create_scene


# ------------------------------------------------------------------
# formatting
# ------------------------------------------------------------------
def print_section(label: str, times_s: List[float], target_fps: int):
    if not times_s:
        print(f"{label}:")
        print("  (no samples collected)\n")
        return

    times_ms = [t * 1e3 for t in times_s]
    mean = statistics.mean(times_ms)
    median = statistics.median(times_ms)
    p99 = sorted(times_ms)[int(len(times_ms) * 0.99)]
    qps = 1000.0 / mean if mean > 0 else float("inf")
    frame_budget_ms = 1000.0 / target_fps
    pct = (mean / frame_budget_ms) * 100.0

    print(f"{label}:")
    print(f"  Mean:   {mean:.3f} ms")
    print(f"  Median: {median:.3f} ms")
    print(f"  P99:    {p99:.3f} ms")
    print(f"  Budget: ~{qps:.0f} queries per second")
    print(f"  At {target_fps} FPS: {pct:.1f}% of frame budget")
    print()


# ------------------------------------------------------------------
# micro-benchmark: calculate_force (synthetic inputs)
# ------------------------------------------------------------------
def bench_calculate_force_micro(cube_id, samples: int) -> List[float]:
    normals = [tuple(np.random.randn(3).tolist()) for _ in range(samples)]
    penetrations = np.random.uniform(0.0, 0.05, size=samples).tolist()
    velocities = [tuple(np.random.randn(3).tolist()) for _ in range(samples)]

    times = []
    for i in range(samples):
        t0 = time.perf_counter()
        calculate_force(normals[i], penetrations[i], cube_id, velocities[i])
        times.append(time.perf_counter() - t0)
    return times


# ------------------------------------------------------------------
# disable PyBullet's built-in contact solver (mirrors base_demo.py)
# ------------------------------------------------------------------
def _disable_default_contact_response(body_id):
    for link in range(-1, p.getNumJoints(body_id)):
        p.changeDynamics(
            body_id, link,
            restitution=0.0,
            lateralFriction=0.0,
            contactStiffness=1e-9,
            contactDamping=1e-9,
        )


# ------------------------------------------------------------------
# full simulation benchmark
# ------------------------------------------------------------------
def bench_simulation(frames: int, warmup: int) -> dict:
    client = p.connect(p.DIRECT)
    plane_id, cube_id, timestep = create_scene(
        p, True, SceneParameters(random_rotation=True)
    )
    _disable_default_contact_response(cube_id)
    _disable_default_contact_response(plane_id)

    initial_orientation = p.getQuaternionFromEuler([0.0, 0.2, 0.0])
    p.resetBasePositionAndOrientation(
        cube_id,
        p.getBasePositionAndOrientation(cube_id)[0],
        initial_orientation,
    )
    p.resetBaseVelocity(cube_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    timings = {
        "contact_lookup": [],
        "force_compute": [],
        "force_apply": [],
        "full_resolve": [],
        "full_frame": [],
    }

    for frame in range(frames + warmup):
        recording = frame >= warmup

        frame_start = time.perf_counter()

        # --- Step simulation first (like base_demo.py) ---
        p.stepSimulation()

        # --- Get velocities (linear + angular) ---
        current_linear_vel, current_angular_vel = p.getBaseVelocity(cube_id)

        # --- Contact point lookup ---
        t1 = time.perf_counter()
        contact_points = p.getContactPoints(bodyA=cube_id, bodyB=plane_id)
        t2 = time.perf_counter()

        # --- Net force + torque (mirrors apply_impulse_predictor) ---
        resolve_start = time.perf_counter()
        compute_total = 0.0
        apply_total = 0.0

        if contact_points:
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            cube_pos = np.array(cube_pos)
            n_contacts = len(contact_points)

            net_force = np.zeros(3)
            net_torque = np.zeros(3)

            for cp in contact_points:
                contact_pos_world = np.array(cp[5])
                v_contact = _contact_point_velocity(
                    cube_id, contact_pos_world, current_linear_vel, current_angular_vel
                )

                tc0 = time.perf_counter()
                force_vector = np.asarray(
                    calculate_force(cp[7], cp[8], cube_id, v_contact.tolist())
                ) / n_contacts
                tc1 = time.perf_counter()

                net_force += force_vector
                net_torque += np.cross(contact_pos_world - cube_pos, force_vector)
                compute_total += tc1 - tc0

            ta0 = time.perf_counter()
            p.applyExternalForce(cube_id, -1, net_force.tolist(), cube_pos.tolist(), p.WORLD_FRAME)
            p.applyExternalTorque(cube_id, -1, net_torque.tolist(), p.WORLD_FRAME)
            ta1 = time.perf_counter()
            apply_total = ta1 - ta0

        resolve_end = time.perf_counter()
        frame_end = time.perf_counter()

        if recording:
            timings["contact_lookup"].append(t2 - t1)
            timings["full_frame"].append(frame_end - frame_start)

            if contact_points:
                timings["force_compute"].append(compute_total)
                timings["force_apply"].append(apply_total)
                timings["full_resolve"].append(resolve_end - resolve_start)

    p.disconnect()
    return timings


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark collision resolution")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force-samples", type=int, default=10_000)
    parser.add_argument("--target-fps", type=int, default=60)
    args = parser.parse_args()

    print()
    print("=" * 50)
    print("  Collision & Force Computation Benchmark")
    print("=" * 50)
    print()

    # --- Micro-benchmark: calculate_force ---
    print(f"Running calculate_force micro-benchmark ({args.force_samples} calls) ...")
    client = p.connect(p.DIRECT)
    _, cube_id, _ = create_scene(p, True, SceneParameters(random_rotation=True))
    force_times = bench_calculate_force_micro(cube_id, args.force_samples)
    p.disconnect()
    print()
    print_section(
        "Single force computation (calculate_force, synthetic)",
        force_times,
        args.target_fps,
    )

    # --- Full simulation ---
    print(
        f"Running full simulation ({args.frames} frames x {args.repeats} repeats, "
        f"{args.warmup} warmup) ..."
    )

    merged = {
        "contact_lookup": [],
        "force_compute": [],
        "force_apply": [],
        "full_resolve": [],
        "full_frame": [],
    }

    for run in range(1, args.repeats + 1):
        print(f"  Run {run}/{args.repeats} ...", end=" ", flush=True)
        t = bench_simulation(args.frames, args.warmup)
        for key in merged:
            merged[key].extend(t[key])
        print("done")

    print()
    print("-" * 50)
    print()

    fps = args.target_fps
    print_section("Single contact lookup (getContactPoints)", merged["contact_lookup"], fps)
    print_section("Single force computation (calculate_force, in-sim)", merged["force_compute"], fps)
    print_section("Single force application (applyExternalForce)", merged["force_apply"], fps)
    print_section("Full collision resolution (compute + apply, all contacts)", merged["full_resolve"], fps)
    print_section("Full frame (detect + query + resolve + step)", merged["full_frame"], fps)

    # --- Summary ---
    total_frames = len(merged["full_frame"])
    contact_frames = len(merged["full_resolve"])
    print("-" * 50)
    print(f"  Total frames measured : {total_frames}")
    print(f"  Frames with contacts : {contact_frames} ({100 * contact_frames / total_frames:.1f}%)")
    print()


if __name__ == "__main__":
    main()