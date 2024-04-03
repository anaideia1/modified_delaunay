import numpy as np

from figures import Point, Triangle


class FileInput:
    def __init__(self, _file_name):
        self.file_name = _file_name

    @staticmethod
    def _str_to_point(line: str) -> Point:
        coords = [int(item) for item in line.split()]
        return Point(*coords)

    def read_points(self) -> set[Point]:
        points = set()
        with open(self.file_name, "r") as f:
            for line in f:
                points.add(self._str_to_point(line))

        return points


class FileOutput:
    def __init__(self, _file_name):
        self.file_name = _file_name

    @staticmethod
    def _triangle_to_str(triangle: Triangle) -> str:
        p1, p2, p3 = triangle.point1, triangle.point1, triangle.point1
        return f'{p1.x} {p1.y} {p2.x} {p2.y} {p3.x} {p3.y}'

    def write_triangles(self, triangles: list[Triangle]) -> None:
        with open(self.file_name, "w") as f:
            for triangle in triangles:
                f.write(f'{self._triangle_to_str(triangle)}\n')


class FileGenerator:
    def __init__(
            self, _file_name: str, num: int = 100, coord_range: int = 100
    ) -> None:
        self.file_name = _file_name
        self.number_of_points = num
        self.coordinates_range = coord_range

    def _generate_points(self) -> set[Point]:
        datapoints = np.random.randint(
            1, self.coordinates_range,
            size=(self.number_of_points, 2)
        )
        return {Point(*point) for point in datapoints}

    def write_points(self, points: list[Point] = None) -> None:
        points = self._generate_points() if points is None else points

        with open(self.file_name, "w") as f:
            for point in points:
                f.write(f'{point.x} {point.y}\n')