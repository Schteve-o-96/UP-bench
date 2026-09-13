import os
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

import numpy as np

from auv_planning.planning.base import BasePlanner
from auv_planning.planning.ASTAR_2025 import AStarPlanner
from auv_planning.planning.RRT_2025 import RRTPlanner
from auv_planning.planning.DJS_2025 import DijkstraPlanner
from auv_planning.wave_env import WaveEnvironment, HOVERINGAUV_COLLISION_THRESHOLD_M
from external.evaluation.baselines.stress_scenarios import (
    OFFLINE_STRESS_SCENARIOS, DIFFICULTY_EXTRACTORS, DIFFICULTY_AXIS_LABELS,
    FAMILY_SEED_OFFSETS, FAMILY_SEED_STRIDE,
)
from _experiment_common import (
    install_native_target_sampling, install_native_target_pool_reuse, sync_planner_bounds, existing_keys,
    fast_forward_episodes,
)

_PLAN_METHODS = {
    AStarPlanner: "plan_path",
    RRTPlanner: "plan_path",
    DijkstraPlanner: "plan_path",
}

PLANNER_FACTORIES = {
    "astar": lambda max_step: AStarPlanner(
        grid_resolution=0.5, max_steps=max_step, collision_threshold=HOVERINGAUV_COLLISION_THRESHOLD_M,
    ),
    "rrt": lambda max_step: RRTPlanner(
        grid_resolution=0.5, max_steps=max_step, collision_threshold=HOVERINGAUV_COLLISION_THRESHOLD_M,
    ),
    "djs": lambda max_step: DijkstraPlanner(
        grid_resolution=0.5, max_steps=max_step, collision_threshold=HOVERINGAUV_COLLISION_THRESHOLD_M,
    ),
}

_FIELDNAMES = [
    "planner", "archetype_family", "difficulty_axis", "difficulty_value", "episode",
    "plan_found", "reached_goal", "timed_out", "collisions",
    "path_length", "step_count", "elapsed_sim_time", "planning_duration", "execution_duration",
    "energy", "smoothness",
    "smoothed_path_valid", "smoothed_path_violations", "smoothed_path_min_clearance",
    "raw_path_valid", "raw_path_violations", "raw_path_min_clearance",
]

_instrumented = False


def _path_collision_check(obstacles, path, collision_threshold):
    violations = 0
    min_clearance = float("inf")
    for pos in path:
        pos_arr = np.asarray(pos, dtype=float)
        clearance = min((obs.distance_to_surface(pos_arr) for obs in obstacles), default=float("inf"))
        min_clearance = min(min_clearance, clearance)
        if clearance < collision_threshold:
            violations += 1
    return violations, min_clearance


def _print_episode_progress(self, record):
    family = getattr(self, "_wave_family", None)
    label = f"{family or '?'}/{getattr(self, '_wave_planner_key', '?')}"
    n = record["episode"]
    total = getattr(self, "_wave_num_episodes", None)
    offset = getattr(self, "_wave_episode_offset", 0)
    denom = f"/{total + offset}" if total is not None else ""
    if not record["plan_found"]:
        status = "plan_fail"
    elif record["reached_goal"]:
        status = "success"
    else:
        status = "timeout"
    collisions_note = f", collisions={record['collisions']}" if record["plan_found"] else ""
    difficulty = record.get("difficulty_value")
    difficulty_note = ""
    if difficulty is not None:
        axis_label = DIFFICULTY_AXIS_LABELS.get(family, "difficulty")
        difficulty_note = f", {axis_label}={difficulty:.2f}"
    print(f"    [{label}] episode {n}{denom}: {status}{collisions_note}{difficulty_note}")


def _current_difficulty_value(self) -> float | None:
    family = getattr(self, "_wave_family", None)
    env = getattr(self, "_wave_env", None)
    extractor = DIFFICULTY_EXTRACTORS.get(family)
    if extractor is None or env is None:
        return None
    return extractor(env.scene)


def _episode_number(self):
    return len(self.episode_records) + 1 + getattr(self, "_wave_episode_offset", 0)


def _instrumented_plan(original):
    def wrapped(self, start, goal, obstacles):
        t0 = time.time()
        result = original(self, start, goal, obstacles)
        planning_duration = time.time() - t0
        if result is None:
            record = {
                "episode": _episode_number(self),
                "difficulty_value": _current_difficulty_value(self),
                "plan_found": False,
                "reached_goal": False,
                "timed_out": False,
                "collisions": 0,
                "path_length": 0.0,
                "step_count": 0,
                "elapsed_sim_time": 0.0,
                "planning_duration": planning_duration,
                "execution_duration": None,
                "energy": None,
                "smoothness": None,
                "smoothed_path_valid": None,
                "smoothed_path_violations": None,
                "smoothed_path_min_clearance": None,
                "raw_path_valid": None,
                "raw_path_violations": None,
                "raw_path_min_clearance": None,
            }
            self.episode_records.append(record)
            _print_episode_progress(self, record)
        else:
            raw_violations, raw_min_clearance = _path_collision_check(
                obstacles, result, self.collision_threshold,
            )
            self._pending_planning_duration = planning_duration
            self._pending_obstacles = obstacles
            self._pending_raw_valid = raw_violations == 0
            self._pending_raw_violations = raw_violations
            self._pending_raw_min_clearance = raw_min_clearance
        return result
    return wrapped


def _instrumented_smooth_path(original):
    def wrapped(self, path, smoothing_factor=1.0, num_points=200):
        smoothed = original(self, path, smoothing_factor=smoothing_factor, num_points=num_points)
        obstacles = getattr(self, "_pending_obstacles", None)
        if smoothed is not None and obstacles is not None:
            violations, min_clearance = _path_collision_check(obstacles, smoothed, self.collision_threshold)
            self._pending_smoothed_valid = violations == 0
            self._pending_smoothed_violations = violations
            self._pending_smoothed_min_clearance = min_clearance
        return smoothed
    return wrapped


def _instrumented_control_loop(original):
    def wrapped(self, env, start_pos, goal_pos, path, desired_speed):
        t0 = time.time()
        reached_goal, step_count, total_path_length, collisions, energy, smoothness = original(
            self, env, start_pos, goal_pos, path, desired_speed,
        )
        execution_duration = time.time() - t0
        planning_duration = getattr(self, "_pending_planning_duration", None)
        smoothed_valid = getattr(self, "_pending_smoothed_valid", None)
        smoothed_violations = getattr(self, "_pending_smoothed_violations", None)
        smoothed_min_clearance = getattr(self, "_pending_smoothed_min_clearance", None)
        raw_valid = getattr(self, "_pending_raw_valid", None)
        raw_violations = getattr(self, "_pending_raw_violations", None)
        raw_min_clearance = getattr(self, "_pending_raw_min_clearance", None)
        self._pending_planning_duration = None
        self._pending_obstacles = None
        self._pending_smoothed_valid = None
        self._pending_smoothed_violations = None
        self._pending_smoothed_min_clearance = None
        self._pending_raw_valid = None
        self._pending_raw_violations = None
        self._pending_raw_min_clearance = None
        record = {
            "episode": _episode_number(self),
            "difficulty_value": _current_difficulty_value(self),
            "plan_found": True,
            "reached_goal": bool(reached_goal),
            "timed_out": bool(not reached_goal and step_count >= self.max_steps),
            "collisions": int(collisions),
            "path_length": float(total_path_length),
            "step_count": int(step_count),
            "elapsed_sim_time": step_count / self.ticks_per_sec,
            "planning_duration": planning_duration,
            "execution_duration": execution_duration,
            "energy": float(energy),
            "smoothness": float(smoothness),
            "smoothed_path_valid": smoothed_valid,
            "smoothed_path_violations": smoothed_violations,
            "smoothed_path_min_clearance": (
                None if smoothed_min_clearance is None or not np.isfinite(smoothed_min_clearance)
                else float(smoothed_min_clearance)
            ),
            "raw_path_valid": raw_valid,
            "raw_path_violations": raw_violations,
            "raw_path_min_clearance": (
                None if raw_min_clearance is None or not np.isfinite(raw_min_clearance)
                else float(raw_min_clearance)
            ),
        }
        self.episode_records.append(record)
        _print_episode_progress(self, record)
        return reached_goal, step_count, total_path_length, collisions, energy, smoothness
    return wrapped


def _instrumented_update_episode_features(original):
    def wrapped(self):
        original(self)
        extractor = DIFFICULTY_EXTRACTORS.get(self.scenario_name)
        if extractor is not None:
            difficulty = extractor(self.scene)
            axis_label = DIFFICULTY_AXIS_LABELS.get(self.scenario_name, "difficulty")
            print(f"    [{self.scenario_name}] sampled scene: {axis_label}={difficulty:.2f}")
    return wrapped


def install_episode_instrumentation():
    global _instrumented
    if _instrumented:
        return
    BasePlanner.control_loop = _instrumented_control_loop(BasePlanner.control_loop)
    BasePlanner.smooth_path = _instrumented_smooth_path(BasePlanner.smooth_path)
    WaveEnvironment._update_episode_features = _instrumented_update_episode_features(
        WaveEnvironment._update_episode_features
    )
    for cls, method_name in _PLAN_METHODS.items():
        setattr(cls, method_name, _instrumented_plan(getattr(cls, method_name)))
    _instrumented = True


def run_family_planner(
    family, builder, planner_key, max_step, seed, num_episodes, n_targets, show, verbose, debug_draw,
    combo_index=None, total_combos=None, skip_episodes=0,
):
    combo_note = f"[{combo_index}/{total_combos}] " if combo_index is not None else ""
    print(f"{combo_note}[{family}] planner={planner_key} sampling scenario and initializing HoloOcean...")
    env = WaveEnvironment(
        builder, seed=seed, n_targets=n_targets, show_viewport=show, verbose=verbose, debug_draw=debug_draw,
    )
    try:
        if skip_episodes:
            print(f"{combo_note}[{family}] planner={planner_key} fast-forwarding {skip_episodes} episode(s) "
                  f"to reproduce episode {skip_episodes + 1} of a longer run with the same --seed...")
            fast_forward_episodes(env, skip_episodes)
        planner = PLANNER_FACTORIES[planner_key](max_step)
        sync_planner_bounds(planner, env)
        planner.episode_records = []
        planner._wave_env = env
        planner._wave_family = family
        planner._wave_planner_key = planner_key
        planner._wave_num_episodes = num_episodes
        planner._wave_episode_offset = skip_episodes
        planner.stop_after_successes = None
        planner.train(env, num_episodes=num_episodes)
        records = planner.episode_records
        print(f"  -> {len(records)} episodes recorded")
        return records
    finally:
        env.close(f"{family}/{planner_key}")


def main(planner_keys, families, seed, num_episodes, max_step, n_targets, show, verbose, debug_draw, out, extend,
         skip_episodes=0):
    install_episode_instrumentation()
    install_native_target_sampling()
    install_native_target_pool_reuse()

    scenarios = {f: b for f, b in OFFLINE_STRESS_SCENARIOS.items() if f in families}
    if not scenarios:
        raise SystemExit(f"no families match families={families}")

    base_seed = seed
    existing_combos = (
        existing_keys(out, _FIELDNAMES, lambda row: (row["archetype_family"], row["planner"]))
        if extend else set()
    )

    total_combos = len(scenarios) * len(planner_keys)
    combo_index = 0
    all_records = []
    skipped = []
    for family, builder in scenarios.items():
        family_seed = base_seed + FAMILY_SEED_OFFSETS[family] * FAMILY_SEED_STRIDE
        for planner_key in planner_keys:
            combo_index += 1
            if (family, planner_key) in existing_combos:
                skipped.append((family, planner_key))
                print(f"[{combo_index}/{total_combos}] [{family}] planner={planner_key} "
                      f"already present in {out} -- skipping (--extend)")
                continue
            records = run_family_planner(
                family, builder, planner_key, max_step, family_seed, num_episodes, n_targets, show, verbose, debug_draw,
                combo_index=combo_index, total_combos=total_combos, skip_episodes=skip_episodes,
            )
            for record in records:
                record["planner"] = planner_key
                record["archetype_family"] = family
                record["difficulty_axis"] = DIFFICULTY_AXIS_LABELS.get(family)
            all_records.extend(records)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if extend else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not (extend and out.stat().st_size > 0):
            writer.writeheader()
        for record in all_records:
            writer.writerow({k: record.get(k) for k in _FIELDNAMES})

    if skipped:
        print(f"\n  skipped {len(skipped)} (family, planner) combo(s) already present in {out}: {skipped}")
    print(f"\n{len(all_records)} {'new ' if extend else ''}episodes across {len(scenarios)} families x "
          f"{len(planner_keys)} planners {'appended to' if extend else 'written to'} {out}")
    for family in scenarios:
        for planner_key in planner_keys:
            rows = [r for r in all_records if r["archetype_family"] == family and r["planner"] == planner_key]
            n_plan_fail = sum(1 for r in rows if not r["plan_found"])
            n_success = sum(1 for r in rows if r["reached_goal"])
            n_timeout = sum(1 for r in rows if r["timed_out"])
            n_success_with_collision = sum(1 for r in rows if r["reached_goal"] and r["collisions"] > 0)
            planning_durations = [r["planning_duration"] for r in rows if r["planning_duration"] is not None]
            avg_plan_s = sum(planning_durations) / len(planning_durations) if planning_durations else float("nan")
            print(f"  {family:32s} {planner_key:8s} n={len(rows):3d}  "
                  f"plan_fail={n_plan_fail:3d}  success={n_success:3d}  timeout={n_timeout:3d}  "
                  f"success_with_collision={n_success_with_collision:3d}  avg_plan_s={avg_plan_s:6.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planners", type=str, default=None,
                         help=f"comma-separated planner keys (default: all of {sorted(PLANNER_FACTORIES)})")
    parser.add_argument("--families", type=str, default=None,
                         help="comma-separated archetype families (default: all four)")
    parser.add_argument("--seed", type=int, default=0, help="base scenario sampling seed")
    parser.add_argument("--num_episodes", type=int, default=15, help="episodes per (family, planner)")
    parser.add_argument("--n_targets", type=int, default=10)
    parser.add_argument("--max_step", type=int, default=12500,
                         help="ticks per episode, applied uniformly across every family for "
                              "comparability -- raised from the original 6000 default after confirming "
                              "some upbench_native_mirror episodes need more time to reach a genuinely "
                              "farther, uniform-random target (not a planning failure: two previously-"
                              "timed-out A* episodes both succeeded at max_step=20000, finishing at "
                              "7598/8307 ticks), the same pattern RRT showed on horizontal_wall_blockade")
    parser.add_argument("--show", action="store_true", help="show simulation windows")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug_draw", action="store_true")
    parser.add_argument("--out", type=str, default=str(_RESULTS_DIR / "offline_stress_experiment.csv"))
    parser.add_argument("--extend", action="store_true",
                         help="append to --out instead of overwriting it, skipping any (family, planner) "
                              "combo already present -- default is to overwrite, as before")
    parser.add_argument("--skip_episodes", type=int, default=0,
                         help="fast-forward this many episodes (scene resample + goal advance, no "
                              "planning/execution) before running the real episode(s). To reproduce a "
                              "single failing episode from a larger run, rerun with the *same* --seed, "
                              "--num_episodes 1, and --skip_episodes set to (failing episode's CSV "
                              "'episode' number - 1) -- this lands on the same scene and, for families "
                              "like upbench_native_mirror whose goal pool is drawn once and consumed in "
                              "sequence across episodes, the same goal too")
    args = parser.parse_args()

    planner_keys = args.planners.split(",") if args.planners else list(PLANNER_FACTORIES)
    unknown_planners = [p for p in planner_keys if p not in PLANNER_FACTORIES]
    if unknown_planners:
        raise SystemExit(f"unknown planner key(s): {unknown_planners} — choices are {sorted(PLANNER_FACTORIES)}")

    all_families = sorted(OFFLINE_STRESS_SCENARIOS)
    families = args.families.split(",") if args.families else all_families
    unknown_families = [f for f in families if f not in all_families]
    if unknown_families:
        raise SystemExit(f"unknown family/families: {unknown_families} — choices are {all_families}")

    main(
        planner_keys, families, args.seed, args.num_episodes, args.max_step,
        args.n_targets, args.show, args.verbose, args.debug_draw, Path(args.out), args.extend,
        skip_episodes=args.skip_episodes,
    )
