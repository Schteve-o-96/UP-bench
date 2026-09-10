import csv
import random
from pathlib import Path

from auv_planning.wave_env import WaveEnvironment

_native_target_sampling_installed = False


def install_native_target_sampling():
    global _native_target_sampling_installed
    if _native_target_sampling_installed:
        return
    original = WaveEnvironment._sample_random_targets

    def patched(self, n):
        if self.scenario_name != "upbench_native_mirror":
            return original(self, n)
        random.seed(self._episode_seed())
        bounds = self.scene.scene_bounds
        return [
            [
                float(random.randint(int(bounds.x[0]), int(bounds.x[1]))),
                float(random.randint(int(bounds.y[0]), int(bounds.y[1]))),
                float(random.randint(int(bounds.z[0]), int(bounds.z[1]))),
            ]
            for _ in range(n)
        ]

    WaveEnvironment._sample_random_targets = patched
    _native_target_sampling_installed = True


def sync_planner_bounds(planner, env) -> None:
    bounds = env.scene.scene_bounds
    planner.x_min, planner.x_max = bounds.x
    planner.y_min, planner.y_max = bounds.y
    planner.z_min, planner.z_max = bounds.z
    planner.nx = int((planner.x_max - planner.x_min) / planner.grid_resolution)
    planner.ny = int((planner.y_max - planner.y_min) / planner.grid_resolution)
    planner.nz = int((planner.z_max - planner.z_min) / planner.grid_resolution)


def existing_keys(out: Path, fieldnames, key_fn):
    if not out.exists():
        return set()
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fieldnames:
            raise SystemExit(
                f"{out} already exists with a different column schema -- refusing to extend it.\n"
                f"  existing: {reader.fieldnames}\n  expected: {fieldnames}\n"
                "Back up or remove the old file first, or write to a different --out path."
            )
        return {key_fn(row) for row in reader}
