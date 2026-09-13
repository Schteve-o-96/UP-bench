import logging
import random
from itertools import chain

import holoocean
import numpy as np
from wave_dsl.concrete import ConcreteBox, ConcreteSphere, ConcreteSphereRegion

from .planning.geometry import BoxShape, Obstacle, SphereShape


GOAL_ZONE_NAME = "goal_zone"

_HOVERINGAUV_MASS_KG = 31.02
_WATER_DENSITY = 997.0
_COEFFICIENT_OF_DRAG = 0.8
_AREA_OF_DRAG = 0.5

HOVERINGAUV_COLLISION_THRESHOLD_M = 0.38


_EPISODE_SEED_STRIDE = 1000


def _obstacle_from_actor(actor) -> Obstacle:
    shape = actor.shape
    rotation = actor.orientation.as_rotation_matrix()
    if isinstance(shape, ConcreteSphere):
        return Obstacle(position=actor.position.as_array(), shape=SphereShape(shape.radius), rotation=rotation)
    if isinstance(shape, ConcreteBox):
        extents = np.array([shape.lx, shape.ly, shape.lz])
        return Obstacle(position=actor.position.as_array(), shape=BoxShape(extents), rotation=rotation)
    raise ValueError(
        f"Unsupported obstacle shape for up-bench integration: {type(shape).__name__} "
        "(only sphere/box are supported for now)"
    )


class WaveEnvironment:
    def __init__(self, scenario_builder_fn, seed, n_targets=10, show_viewport=False, verbose=False,
                 debug_draw=False):
        logging.getLogger("wave_dsl").setLevel(logging.WARNING)

        if "Ocean" not in holoocean.installed_packages():
            holoocean.install("Ocean")

        self.debug_draw = debug_draw

        self.scenario = scenario_builder_fn()
        self.scenario_name = scenario_builder_fn.__name__
        self._initial_seed = seed
        self._episode_index = 0
        self.scene = self.scenario.sample(seed=seed).scene
        self.agent_name = self.scene.agents[0].name

        self._plan = self._build_plan()
        self.obstacles = [_obstacle_from_actor(a) for a in self.scene.actors]
        self._update_episode_features()

        self.env = holoocean.make(scenario_cfg=self._plan.config, show_viewport=show_viewport, verbose=verbose)
        self._spawn_props()

        self._needs_resample = False

        self.pose = np.zeros((4, 4))
        self.prev_location = self.pose[0:3, 3]
        self.location = self.pose[0:3, 3]
        self.rotation = np.zeros((3,))
        self.velocity = np.zeros((3,))
        self.lasers = np.zeros((14,))
        self.observation_space = [self.pose, self.rotation, self.velocity, self.lasers]
        self.observation_space = [item for sublist in self.observation_space for item in sublist.flatten()]

        self.targets = self._sample_random_targets(n_targets)
        self.choosen_targets = []
        self.current_target = self.choose_next_target()

    def _build_plan(self, scene=None):
        plan = self.scenario.adapter.prepare(scene if scene is not None else self.scene)
        plan.config["ticks_per_sec"] = 100
        return plan

    def _episode_seed(self):
        return self._initial_seed + self._episode_index * _EPISODE_SEED_STRIDE

    def _resample_scene(self):
        self._episode_index += 1
        seed = self._episode_seed()
        self.scene = self.scenario.sample(seed=seed).scene
        self._plan = self._build_plan(self.scene)
        self.obstacles = [_obstacle_from_actor(a) for a in self.scene.actors]
        self._update_episode_features()

    def _update_episode_features(self):
        spawn = self.scene.agents[0].position.as_array()
        self.min_obstacle_standoff = min(
            (obs.distance_to_surface(spawn) for obs in self.obstacles), default=float("inf"),
        )
        if self.scene.currents:
            current_vel = self.scenario.adapter.current_velocity(self.scene, spawn)
            self.current_magnitude_at_start = float(np.linalg.norm(current_vel))
        else:
            self.current_magnitude_at_start = 0.0
        self.spawn_yaw_offset = self.scene.agents[0].orientation.yaw

    def _spawn_props(self):
        for method, args in self._plan.env_commands:
            getattr(self.env, method)(*args)

    def _sample_random_targets(self, n):
        random.seed(self._episode_seed())
        goal_region = next((r for r in self.scene.regions if r.name == GOAL_ZONE_NAME), None)
        return [
            self._sample_avoiding_obstacles(lambda: self._sample_target_candidate(goal_region))
            for _ in range(n)
        ]

    def _sample_avoiding_obstacles(self, candidate_fn, max_attempts=1000):
        candidate = candidate_fn()
        for _ in range(max_attempts):
            if not any(
                obs.distance_to_surface(candidate) < HOVERINGAUV_COLLISION_THRESHOLD_M
                for obs in self.obstacles
            ):
                return candidate
            candidate = candidate_fn()
        return candidate

    def _sample_target_candidate(self, goal_region):
        bounds = self.scene.scene_bounds
        if goal_region is not None and isinstance(goal_region.shape, ConcreteSphereRegion):
            center = goal_region.position.as_array()
            radius = goal_region.shape.radius
            direction = np.array([random.gauss(0.0, 1.0) for _ in range(3)])
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
            r = radius * random.random() ** (1.0 / 3.0)  # uniform within the sphere's volume
            candidate = center + direction * r
        else:
            candidate = np.array([random.uniform(*bounds.x), random.uniform(*bounds.y), random.uniform(*bounds.z)])
        lo = [bounds.x[0], bounds.y[0], bounds.z[0]]
        hi = [bounds.x[1], bounds.y[1], bounds.z[1]]
        return np.clip(candidate, lo, hi).tolist()

    def choose_next_target(self):
        while True:
            target = random.choice(self.targets)
            if target not in self.choosen_targets:
                self.choosen_targets.append(target)
                return target

    def get_current_target(self):
        return self.current_target

    def set_current_target(self, target):
        self.current_target = target

    def draw_targets(self):
        for i in self.targets:
            if i == self.current_target:
                self.env.draw_point(i, color=[0, 255, 0], thickness=5, lifetime=0)
            else:
                self.env.draw_point(i, color=[255, 255, 0], thickness=5, lifetime=0)

    def _refresh_targets_for_new_episode(self):
        self.targets = self._sample_random_targets(len(self.targets))
        self.choosen_targets = []
        self.current_target = self.choose_next_target()

    def reset(self):
        if self._needs_resample:
            self._resample_scene()
        else:
            self._needs_resample = True

        self.env.reset()

        agent_cfg = self._plan.config["agents"][0]
        self.env.agents[self.agent_name].teleport(
            location=agent_cfg["location"], rotation=agent_cfg["rotation"],
        )
        self._spawn_props()

        self._refresh_targets_for_new_episode()

        bounds = self.scene.scene_bounds
        center = [(bounds.x[0] + bounds.x[1]) / 2, (bounds.y[0] + bounds.y[1]) / 2, (bounds.z[0] + bounds.z[1]) / 2]
        extent = [(bounds.x[1] - bounds.x[0]) / 2, (bounds.y[1] - bounds.y[0]) / 2, (bounds.z[1] - bounds.z[0]) / 2]
        self.env.draw_box(center=center, extent=extent, thickness=50, lifetime=0)
        self.draw_targets()
        if self.debug_draw and self.scene.currents:
            self.scenario.adapter.draw_currents(self.env, self.scene, lifetime=0)

    def _current_disturbance_accel(self) -> np.ndarray:
        if not self.scene.currents:
            current_vel = np.zeros(3)
        else:
            current_vel = self.scenario.adapter.current_velocity(self.scene, self.location)
        relative_vel = self.velocity - current_vel
        speed = np.linalg.norm(relative_vel)
        if speed < 1e-9:
            return np.zeros(3)
        drag_force = (
            -0.5 * _WATER_DENSITY * speed ** 2
            * _COEFFICIENT_OF_DRAG * _AREA_OF_DRAG
            * (relative_vel / speed)
        )
        return drag_force / _HOVERINGAUV_MASS_KG

    def tick(self, action):
        action = np.array(action, dtype=float)
        action[0:3] += self._current_disturbance_accel()
        self.env.act(self.agent_name, action)
        state = self.env.tick()
        if self._plan.tick_plan:
            pos = np.array(state["PoseSensor"])[0:3, 3]
            positions = {self.agent_name: pos}
            for command_fn in self._plan.tick_plan:
                for method, args in command_fn(positions):
                    getattr(self.env, method)(*args)
        return state

    def update_state(self, states):
        sensors = ["PoseSensor", "VelocitySensor", "RotationSensor", "HorizontalRangeSensor", "UpRangeSensor",
                   "DownRangeSensor", "UpInclinedRangeSensor", "DownInclinedRangeSensor"]

        if all(element in states for element in sensors):
            self.pose = states["PoseSensor"]
            self.rotation = states["RotationSensor"]
            self.velocity = states['VelocitySensor']
            self.lasers = list(chain.from_iterable([states[key] for key in ["HorizontalRangeSensor", "UpRangeSensor",
                                                                            "DownRangeSensor",
                                                                            "UpInclinedRangeSensor",
                                                                            "DownInclinedRangeSensor"]]))
            self.pose = np.array(self.pose)
            self.location = self.pose[0:3, 3]
            self.rotation = np.array(self.rotation) + 180
            self.velocity = np.array(self.velocity)
            self.lasers = np.array(self.lasers)
            self.observation_space = [self.pose, self.rotation, self.velocity, self.lasers]
            self.observation_space = [item for sublist in self.observation_space for item in sublist.flatten()]

    def close(self, label: str = "") -> None:
        close = getattr(self.env, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as e:
            prefix = f"[{label}] " if label else ""
            logging.warning(f"{prefix}env.close() raised {e!r} — continuing anyway")
