"""
Speed up (or slow down) a video so its total duration becomes exactly TARGET_DURATION seconds.
Uses moviepy to re-encode the video with an adjusted speed factor.
"""
from moviepy import VideoFileClip
import os

TARGET_DURATION = 10.0  # seconds

files = [
    "collision_run_gnn.mp4",
    "collision_run_gt.mp4",
    "collision_run_mlp.mp4",
    "collision_run_transfomer.mp4",
]

os.makedirs("videos", exist_ok=True)

for filename in files:
    input_path = f"{filename}"
    output_path = f"videos/{filename}"

    print(f"\n{'='*60}")
    print(f"Processing: {filename}")

    clip = VideoFileClip(input_path)
    print(f"  Original duration: {clip.duration:.2f}s")

    speed_factor = clip.duration / TARGET_DURATION
    print(f"  Speed factor: {speed_factor:.4f}x")

    fast_clip = clip.with_speed_scaled(speed_factor)
    print(f"  New duration: {fast_clip.duration:.2f}s")

    fast_clip.write_videofile(
        output_path,
        codec="libx264",
        fps=60,
        preset="medium",
        bitrate="2000k",
        audio=False,
        logger="bar",
    )

    clip.close()
    fast_clip.close()
    print(f"  Saved to {output_path}")

print(f"\n{'='*60}")
print("All done!")