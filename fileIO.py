from figures import Point, Triangle


class FileInput:
    def __init__(self, _file_name):
        self.file_name = _file_name

    @staticmethod
    def str_to_point(line: str) -> Point:
        coords = line.split()
        return Point(*coords)

    def read_points(self) -> list[Point]:
        points = []
        with open(self.file_name, "r") as f:
            for line in f:
                points.append(self.str_to_point(line))

        return points


class FileOutput:
    def __init__(self, _file_name):
        self.file_name = _file_name

    @staticmethod
    def triangle_to_str(triangle: Triangle) -> str:
        return str(triangle)

    def write_triangles(self, triangles: list[Triangle]) -> None:
        with open(self.file_name, "w") as f:
            for triangle in triangles:
                f.write(self.triangle_to_str(triangle))