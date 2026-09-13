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

        def candidate_fn():
            return [
                float(random.randint(int(bounds.x[0]), int(bounds.x[1]))),
                float(random.randint(int(bounds.y[0]), int(bounds.y[1]))),
                float(random.randint(int(bounds.z[0]), int(bounds.z[1]))),
            ]

        return [self._sample_avoiding_obstacles(candidate_fn) for _ in range(n)]

    WaveEnvironment._sample_random_targets = patched
    _native_target_sampling_installed = True


_native_target_pool_reuse_installed = False


def install_native_target_pool_reuse():
    global _native_target_pool_reuse_installed
    if _native_target_pool_reuse_installed:
        return
    original_refresh = WaveEnvironment._refresh_targets_for_new_episode
    original_choose = WaveEnvironment.choose_next_target

    def patched_refresh(self):
        if self.scenario_name != "upbench_native_mirror":
            return original_refresh(self)
        self._native_mirror_reset_count = getattr(self, "_native_mirror_reset_count", 0) + 1
        return

    def patched_choose(self):
        if self.scenario_name != "upbench_native_mirror":
            return original_choose(self)
        reset_count = getattr(self, "_native_mirror_reset_count", 0)
        if getattr(self, "_native_mirror_target_drawn_at", None) == reset_count:
            return self.current_target
        self._native_mirror_target_drawn_at = reset_count
        return original_choose(self)

    WaveEnvironment._refresh_targets_for_new_episode = patched_refresh
    WaveEnvironment.choose_next_target = patched_choose
    _native_target_pool_reuse_installed = True


def fast_forward_episodes(env, n, target_advances_per_episode=1):
    for _ in range(n):
        env.reset()
        for _ in range(target_advances_per_episode):
            env.set_current_target(env.choose_next_target())


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
