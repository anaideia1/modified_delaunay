import numpy as np

from abc import ABC, abstractmethod
from math import atan2, pi
from matplotlib import pyplot as plt

from figures import Point, Hull


class GrahamScanInterface(ABC):
    """
    Interface class for Dependency injection for different GrahamScan classes.
    """
    @abstractmethod
    def __init__(self, _datapoints):
        ...

    @abstractmethod
    def execute(self):
        ...


class GrahamScan(GrahamScanInterface):
    """
    Class for building convex hull from set of points.
    """
    def __init__(self, _datapoints: set[Point] = None):
        self.datapoints = set() if _datapoints is None else _datapoints
        self.anchor_point = None

    def _set_anchor_point(self) -> Point:
        """
        Find and set anchor point to self.anchor_point for Graham scan
        It's lowermost point (or leftmost if there are few with the same height).
        :return: newly assigned anchor point
        """
        for point in self.datapoints:
            if (
                    self.anchor_point is None
                    or point.y < self.anchor_point.y
                    or point.y == self.anchor_point.y
                    and point.x < self.anchor_point.x
            ):
                self.anchor_point = point

        return self.anchor_point

    @staticmethod
    def _polar_angle(center: Point, point: Point) -> float:
        """
        Measure polar angle for point with origin in center point.
        :param center: Center point for polar coordinated
        :param point: point, those polar angle we measure
        :return: polar angle for point with origin in center point
        """
        y_span = point.y - center.y
        x_span = point.x - center.x
        if not y_span and not x_span:
            return pi
        return atan2(y_span,x_span)

    def _get_sorted_by_angle_datapoints(self) -> list[Point]:
        """
        Sort all datapoints by polar angle with coordinates origin in anchor
        point in ascending order. If polar angle same sort by x-coordinate also
        in ascending order.
        :return: sorted list with points
        """
        self._set_anchor_point()
        datapoints_angles = [
            [point.x,point.y, self._polar_angle(self.anchor_point, point)]
            for point in self.datapoints
        ]
        datapoints_angles = np.array(datapoints_angles)
        datapoints_angles = datapoints_angles[np.lexsort(
            (datapoints_angles[:,0], datapoints_angles[:,2])
        )]
        return [Point(*point) for point in datapoints_angles[:,(0,1)]]

    @staticmethod
    def _is_counter_cw(a: Point, b: Point, c: Point, d: Point) -> bool:
        """
        Decide if vector (a, c) has counter clockwise orientation to vector
        (a, b). If they on the same line (nor clockwise, counter clockwise),
        then compare vectors' orientation (d, c) and (d, b).
        :param a: point added before previous
        :param b: point added previously
        :param c: current point in adding cycle
        :param d: point added before point before previous
        :return: condition for adding new point by vectors orientation
        """
        res = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
        if res == 0:
            res = (b.x - d.x) * (c.y - d.y) - (c.x - d.x) * (b.y - d.y)
        return res < 0

    def execute(self) -> Hull:
        """
        Main function of this class.
        Return Hull instance with points, which is convex hull for provided
        during class instance initialization datapoints.
        :return: Convex Hull
        """
        sorted_datapoints = self._get_sorted_by_angle_datapoints()
        convex_hull_points = [self.anchor_point, sorted_datapoints[0],
                              sorted_datapoints[1]]
        for point in sorted_datapoints[2:]:
            while (
                    len(convex_hull_points) > 2
                    and self._is_counter_cw(
                        convex_hull_points[-2],
                        convex_hull_points[-1],
                        point,
                        convex_hull_points[-3]
                    )
            ):
                del convex_hull_points[-1]  # backtrack
            convex_hull_points.append(point)

        return Hull(points=convex_hull_points[:-1])


class RecurGrahamScan:
    """
    Class for building convex hull RECURSIVELY from set of points,
    Calculate one convex hull, delete current convex hull's points from set of
    initial points and do it again, while it's still possible (>2 point left).
    """
    def __init__(
            self,
            _graham_scan: GrahamScanInterface,
            _datapoints: set[Point] = None
    ) -> None:
        self.graham_scan_class = _graham_scan
        self.datapoints = set() if _datapoints is None else _datapoints

    def execute(self, make_plot: bool = False) -> list[Hull]:
        """
        Main function of this class.
        Return list of Hull instances with points, which is recursive convex
        hulls for provided during class instance initialization datapoints.
        :param make_plot: enabling drawing resulting convex hulls on plot
        :return: list of convex hull
        """
        convex_hulls = []
        remaining_points = self.datapoints.copy()

        while len(remaining_points) >= 3:
            convex_hull = self.graham_scan_class(remaining_points).execute()
            convex_hulls.append(convex_hull)
            remaining_points -= set(convex_hull.points)

        last_hull = Hull(points=list(remaining_points))
        convex_hulls.append(last_hull)

        if make_plot:
            points = np.array([(point.x, point.y) for point in self.datapoints])
            plt.scatter(points[:, 0], points[:, 1])
            for hull in convex_hulls[:-1]:
                hull_array = [(point.x, point.y) for point in hull.points]
                hull_array.append(hull_array[0])
                hull_array = np.array(hull_array)
                plt.plot(hull_array[:, 0], hull_array[:, 1], c='r')

        return convex_hulls


if __name__ == '__main__':
    number_of_datapoints = 50
    datapoints = np.random.randint(1,100,size=(number_of_datapoints,2))
    datapoints = {Point(*point) for point in datapoints}

    convex_hulls = RecurGrahamScan(GrahamScan, datapoints).execute(make_plot=True)
    print(f'{convex_hulls = }')
    plt.show()

