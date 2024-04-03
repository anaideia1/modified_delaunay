from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """
    Dataclass representing points on the plane.
    """
    x: float
    y: float


@dataclass(frozen=True)
class Triangle:
    """
    Dataclass representing triangle on the plane.
    Used for saving resulted triangulation.
    """
    point1: Point
    point2: Point
    point3: Point


@dataclass
class Hull:
    """
    Dataclass representing hulls on the plane. Used for saving convex hulls.
    """
    points: list[Point]