from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class ObstacleShape(ABC):
    """A shape's geometry in its own local frame (centered at the origin, unrotated).

    Add new obstacle kinds (cylinder, cone, mesh, ...) by subclassing this with a
    matching entry in `from_concrete_shape` below — nothing else in the planners
    needs to change, since they only ever go through `Obstacle.contains_point` /
    `distance_to_surface` / `bounding_radius`.
    """

    @abstractmethod
    def contains_point(self, local_point: np.ndarray) -> bool: ...

    @abstractmethod
    def distance_to_surface(self, local_point: np.ndarray) -> float: ...

    @abstractmethod
    def bounding_radius(self) -> float: ...


@dataclass
class SphereShape(ObstacleShape):
    radius: float

    def contains_point(self, local_point: np.ndarray) -> bool:
        return float(np.linalg.norm(local_point)) <= self.radius

    def distance_to_surface(self, local_point: np.ndarray) -> float:
        return max(0.0, float(np.linalg.norm(local_point)) - self.radius)

    def bounding_radius(self) -> float:
        return self.radius


@dataclass
class BoxShape(ObstacleShape):
    extents: np.ndarray  # full size (lx, ly, lz)

    def __post_init__(self):
        self.extents = np.asarray(self.extents, dtype=float)

    def contains_point(self, local_point: np.ndarray) -> bool:
        return bool(np.all(np.abs(local_point) <= self.extents / 2.0))

    def distance_to_surface(self, local_point: np.ndarray) -> float:
        half_extents = self.extents / 2.0
        clamped = np.clip(local_point, -half_extents, half_extents)
        return float(np.linalg.norm(local_point - clamped))

    def bounding_radius(self) -> float:
        return float(np.linalg.norm(self.extents / 2.0))


@dataclass
class Obstacle:
    """A positioned, optionally-oriented obstacle: `shape` carries the geometry,
    `position`/`rotation` carry where it sits in the world."""

    position: np.ndarray
    shape: ObstacleShape
    rotation: np.ndarray | None = None  # 3x3 world-from-local; None = axis-aligned

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.rotation is not None:
            self.rotation = np.asarray(self.rotation, dtype=float)

    def __iter__(self):
        return iter(self.position)

    def __getitem__(self, i):
        return self.position[i]

    def _to_local(self, point) -> np.ndarray:
        offset = np.asarray(point, dtype=float) - self.position
        if self.rotation is None:
            return offset
        return self.rotation.T @ offset

    def contains_point(self, point) -> bool:
        return self.shape.contains_point(self._to_local(point))

    def distance_to_surface(self, point) -> float:
        return self.shape.distance_to_surface(self._to_local(point))

    def bounding_radius(self) -> float:
        return self.shape.bounding_radius()
