from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
     x: int
     y: int


@dataclass(frozen=True)
class Triangle:
    point1: Point
    point2: Point
    point3: Point


@dataclass
class Hull:
    points: set[Point]