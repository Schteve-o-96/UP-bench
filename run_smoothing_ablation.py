import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

import matplotlib.pyplot as plt
import numpy as np

from auv_planning.planning.ASTAR_2025 import AStarPlanner
from auv_planning.wave_env import _obstacle_from_actor, HOVERINGAUV_COLLISION_THRESHOLD_M as _COLLISION_THRESHOLD
from external.evaluation.baselines.stress_scenarios import GOAL_ZONE_NAME, vertical_wall_ablation
from external.evaluation.plotting import plot_ablation_sweep
from run_offline_stress_experiment import _path_collision_check
_GRID_RESOLUTION = 0.5
_MAX_STEPS = 6000

_FIELDNAMES = [
    "lateral_offset", "plan_found", "planning_duration",
    "smoothed_path_valid", "smoothed_path_violations", "smoothed_path_min_clearance", "margin",
]


def run_one(lateral_offset: float) -> dict:
    scene = vertical_wall_ablation(lateral_offset).sample(seed=0).scene
    start = scene.robots[0].position.as_array()
    goal = next(r for r in scene.regions if r.name == GOAL_ZONE_NAME).position.as_array()
    obstacles = [_obstacle_from_actor(a) for a in scene.actors]

    planner = AStarPlanner(grid_resolution=_GRID_RESOLUTION, max_steps=_MAX_STEPS,
                            collision_threshold=_COLLISION_THRESHOLD)

    t0 = time.time()
    path = planner.plan_path(start, goal, obstacles)
    planning_duration = time.time() - t0

    if path is None:
        return {
            "lateral_offset": lateral_offset, "plan_found": False, "planning_duration": planning_duration,
            "smoothed_path_valid": None, "smoothed_path_violations": None,
            "smoothed_path_min_clearance": None, "margin": None,
        }

    smoothed = planner.smooth_path(path, smoothing_factor=1.0, num_points=200)
    violations, min_clearance = _path_collision_check(obstacles, smoothed, planner.collision_threshold)
    margin = min_clearance - _COLLISION_THRESHOLD if np.isfinite(min_clearance) else None
    return {
        "lateral_offset": lateral_offset, "plan_found": True, "planning_duration": planning_duration,
        "smoothed_path_valid": violations == 0, "smoothed_path_violations": violations,
        "smoothed_path_min_clearance": min_clearance if np.isfinite(min_clearance) else None,
        "margin": margin,
    }


def main(lo: float, hi: float, n: int, out: Path) -> None:
    offsets = np.linspace(lo, hi, n)
    rows = []
    print(f"  {'lateral_offset':>10s}  {'plan':>6s}  {'violations':>10s}  {'min_clearance':>13s}  {'margin':>8s}")
    for w in offsets:
        row = run_one(float(w))
        rows.append(row)
        plan_str = "yes" if row["plan_found"] else "NO"
        viol = row["smoothed_path_violations"]
        clr = row["smoothed_path_min_clearance"]
        margin = row["margin"]
        print(f"  {w:10.2f}  {plan_str:>6s}  {('-' if viol is None else viol):>10}  "
              f"{('-' if clr is None else f'{clr:.3f}'):>13}  {('-' if margin is None else f'{margin:+.3f}'):>8}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ax = plot_ablation_sweep(
        [row["lateral_offset"] for row in rows], [row["margin"] for row in rows],
        xlabel="lateral_offset (m) — wall placement relative to the AUV's approach line",
        ylabel="smoothed-path clearance margin (m)\n(negative = plan pre-invalidated before execution)",
    )
    ax.set_title("Smoothing-defect threshold — vertical_wall_blockade, A*", fontsize=10, color="#0b0b0b")
    fig_out = out.with_suffix(".png")
    ax.figure.savefig(fig_out, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"  wrote {fig_out}")

    crossing = None
    for a, b in zip(rows, rows[1:]):
        if a["margin"] is not None and b["margin"] is not None and a["margin"] * b["margin"] < 0:
            crossing = (a["lateral_offset"], b["lateral_offset"])
            break
    print(f"\n{len(rows)} points written to {out}")
    if crossing:
        print(f"  margin crosses zero between lateral_offset={crossing[0]:.2f} and {crossing[1]:.2f} "
              f"-- the threshold where the smoothed plan starts pre-invalidating itself")
    else:
        print("  margin never crosses zero in this range -- widen --lo/--hi to bracket the threshold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lo", type=float, default=-5.0, help="most negative lateral_offset to sweep (m)")
    parser.add_argument("--hi", type=float, default=5.0, help="most positive lateral_offset to sweep (m)")
    parser.add_argument("--n", type=int, default=25, help="number of points across [lo, hi]")
    parser.add_argument("--out", type=str, default=str(_RESULTS_DIR / "smoothing_ablation.csv"))
    args = parser.parse_args()
    main(args.lo, args.hi, args.n, Path(args.out))
