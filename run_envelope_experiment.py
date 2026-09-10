import os
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

from auv_planning.planning.base import BasePlanner
from auv_planning.planning.base import BasePlanner
from auv_planning.planning.Theta_2025 import RSAPPlanner
from auv_planning.wave_env import WaveEnvironment, HOVERINGAUV_COLLISION_THRESHOLD_M
from external.evaluation.baselines.stress_scenarios import (
    ALL_STRESS_SCENARIOS, DIFFICULTY_EXTRACTORS, DIFFICULTY_AXIS_LABELS,
    FAMILY_SEED_OFFSETS, FAMILY_SEED_STRIDE,
)
from _experiment_common import install_native_target_sampling, sync_planner_bounds, existing_keys

SCENARIOS = dict(ALL_STRESS_SCENARIOS)

_FIELDNAMES = [
    "archetype", "archetype_family", "difficulty_axis", "difficulty_value", "episode",
    "min_obstacle_standoff", "current_magnitude", "spawn_yaw_offset",
    "closest_approach", "reached_goal", "timed_out", "collisions",
    "path_length", "step_count", "elapsed_sim_time", "wall_clock_seconds",
]

_instrumented = False


def _family_for_scenario_name(name: str) -> str:
    return name[: -len("_current")] if name.endswith("_current") else name


def install_difficulty_logging():
    global _instrumented
    if _instrumented:
        return
    original_reset = WaveEnvironment.reset

    def instrumented_reset(self):
        original_reset(self)
        family = _family_for_scenario_name(self.scenario_name)
        extractor = DIFFICULTY_EXTRACTORS.get(family)
        value = extractor(self.scene) if extractor is not None else None
        if not hasattr(self, "_wave_difficulty_log"):
            self._wave_difficulty_log = []
        self._wave_difficulty_log.append(value)

        n = len(self._wave_difficulty_log)
        total = getattr(self, "_wave_num_episodes", None)
        denom = f"/{total}" if total is not None else ""
        name = getattr(self, "_wave_archetype_name", self.scenario_name)
        value_note = f" — difficulty={value:.3g}" if value is not None else ""
        print(f"    [{name}] episode {n}{denom}{value_note}")

    WaveEnvironment.reset = instrumented_reset
    _instrumented = True


def run_archetype(
    name, builder, *, seed, num_episodes, n_targets, max_step, show, verbose, debug_draw,
    combo_index=None, total_combos=None,
):
    combo_note = f"[{combo_index}/{total_combos}] " if combo_index is not None else ""
    print(f"{combo_note}[{name}] sampling scenario and initializing HoloOcean...")
    env = WaveEnvironment(
        builder, seed=seed, n_targets=n_targets, show_viewport=show, verbose=verbose, debug_draw=debug_draw,
    )
    env._wave_archetype_name = name
    env._wave_num_episodes = num_episodes
    try:
        planner = RSAPPlanner(
            grid_resolution=0.5, max_steps=max_step,
            collision_threshold=HOVERINGAUV_COLLISION_THRESHOLD_M, stop_after_successes=None,
        )
        sync_planner_bounds(planner, env)
        planner.train(env, num_episodes=num_episodes)
        records = planner.episode_records
        difficulty_log = getattr(env, "_wave_difficulty_log", [])
        family = _family_for_scenario_name(name)
        axis_label = DIFFICULTY_AXIS_LABELS.get(family)
        for record, value in zip(records, difficulty_log):
            record["archetype_family"] = family
            record["difficulty_axis"] = axis_label
            record["difficulty_value"] = value
        print(f"[{name}] {len(records)} episodes collected")
        return records
    finally:
        env.close(name)


def main(scenarios, seed, num_episodes, n_targets, max_step, show, verbose, debug_draw, out, extend):
    install_difficulty_logging()
    install_native_target_sampling()

    base_seed = seed
    existing = existing_keys(out, _FIELDNAMES, lambda row: row["archetype"]) if extend else set()

    total_combos = len(scenarios)
    all_records = []
    skipped = []
    for i, name in enumerate(scenarios):
        if name in existing:
            skipped.append(name)
            print(f"[{i + 1}/{total_combos}] [{name}] already present in {out} -- skipping (--extend)")
            continue
        records = run_archetype(
            name, SCENARIOS[name],
            seed=base_seed + FAMILY_SEED_OFFSETS[name] * FAMILY_SEED_STRIDE, num_episodes=num_episodes,
            n_targets=n_targets,
            max_step=max_step, show=show, verbose=verbose, debug_draw=debug_draw,
            combo_index=i + 1, total_combos=total_combos,
        )
        all_records.extend(records)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if extend else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not (extend and out.stat().st_size > 0):
            writer.writeheader()
        for record in all_records:
            writer.writerow({k: record.get(k) for k in _FIELDNAMES})

    if skipped:
        print(f"\n  skipped {len(skipped)} archetype(s) already present in {out}: {skipped}")
    print(f"\n{len(all_records)} {'new ' if extend else ''}episodes across {len(scenarios)} archetypes "
          f"{'appended to' if extend else 'written to'} {out}")
    for name in scenarios:
        rows = [r for r in all_records if r["archetype"] == name]
        n_success = sum(1 for r in rows if r["reached_goal"])
        n_timeout = sum(1 for r in rows if r["timed_out"] and not r["reached_goal"])
        print(f"  {name:32s} n={len(rows):3d}  success={n_success:3d}  timeout={n_timeout:3d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=str, default=None,
                         help="comma-separated archetype names (default: all "
                              f"{len(SCENARIOS)} registered scenarios — {sorted(SCENARIOS)})")
    parser.add_argument("--seed", type=int, default=0, help="base scenario sampling seed")
    parser.add_argument("--num_episodes", type=int, default=50, help="episodes per archetype")
    parser.add_argument("--n_targets", type=int, default=10)
    parser.add_argument("--max_step", type=int, default=12500)
    parser.add_argument("--show", action="store_true", help="show simulation windows")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug_draw", action="store_true")
    parser.add_argument("--out", type=str, default=str(_RESULTS_DIR / "rsap_envelope_experiment.csv"))
    parser.add_argument("--extend", action="store_true",
                         help="append to --out instead of overwriting it, skipping any archetype "
                              "already present -- default is to overwrite, as before")
    args = parser.parse_args()

    scenario_names = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
    unknown = [n for n in scenario_names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario name(s): {unknown} — choices are {sorted(SCENARIOS)}")

    main(
        scenario_names, args.seed, args.num_episodes, args.n_targets, args.max_step,
        args.show, args.verbose, args.debug_draw, Path(args.out), args.extend,
    )
