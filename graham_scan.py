import numpy as np

from abc import ABC, abstractmethod
from math import atan2, pi
from matplotlib import pyplot as plt
from matplotlib import animation

from figures import Point


class GrahamScanInterface(ABC):
    @abstractmethod
    def __init__(self, _datapoints):
        ...

    @abstractmethod
    def execute(self):
        ...


class GrahamScan(GrahamScanInterface):
    def __init__(self, _datapoints=None):
        self.datapoints = [] if _datapoints is None else _datapoints
        self.anchor_point = None

    def _set_anchor_point(self):
        self.anchor_point = self.datapoints[0]
        for point in self.datapoints:
            if (
                    point[1] < self.anchor_point[1]
                    or point[1] == self.anchor_point[1]
                    and point[0] < self.anchor_point[0]
            ):
                self.anchor_point = point

    @staticmethod
    def _polar_angle(p0, p1):
        y_span=p1[1]-p0[1]
        x_span=p1[0]-p0[0]
        if not y_span and not x_span:
            return pi
        return atan2(y_span,x_span)

    def _get_sorted_by_angle_datapoints(self):
        self._set_anchor_point()
        datapoints_angles = [
            [point[0],point[1], self._polar_angle(self.anchor_point, point)]
            for point in self.datapoints
        ]
        datapoints_angles = np.array(datapoints_angles)
        datapoints_angles = datapoints_angles[np.lexsort(
            (datapoints_angles[:,0], datapoints_angles[:,2])
        )]
        return datapoints_angles[:,(0,1)]

    @staticmethod
    def _is_counter_cw(a, b, c, d):
        res = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if res == 0:
            res = (b[0] - d[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - d[1])
        return res < 0

    def execute(self):
        sorted_datapoints = self._get_sorted_by_angle_datapoints()
        convex_hull = [self.anchor_point, sorted_datapoints[0], sorted_datapoints[1]]
        for point in sorted_datapoints[2:]:
            while len(convex_hull) > 2 and self._is_counter_cw(convex_hull[-2],convex_hull[-1], point, convex_hull[-3]):
                del convex_hull[-1] # backtrack
            convex_hull.append(point)
        return convex_hull


class RecurGrahamScan:
    def __init__(self, _graham_scan: GrahamScanInterface, _datapoints=None):
        self.graham_scan_class = _graham_scan
        self.datapoints = [] if _datapoints is None else _datapoints

    def _subtract_convex_hull(self, remaining_points, last_hull):
        remaining_set = set([tuple(x) for x in remaining_points])
        hull_set = set([tuple(x) for x in last_hull])
        return np.array(list(remaining_set - hull_set))

    def execute(self, make_plot=False):
        convex_hulls = []
        remaining_points = self.datapoints.copy()

        while len(remaining_points) >= 3:
            convex_hull = self.graham_scan_class(remaining_points).execute()
            convex_hulls.append(convex_hull)
            convex_hull = np.array(convex_hull)
            remaining_points = self._subtract_convex_hull(
                remaining_points, convex_hull
            )
        convex_hulls.append(remaining_points)

        if make_plot:
            plt.scatter(self.datapoints[:, 0], self.datapoints[:, 1])
            for hull in convex_hulls[:-1]:
                hull = np.array(hull)
                plt.plot(hull[:, 0], hull[:, 1], c='r')


        return convex_hulls


if __name__ == '__main__':
    number_of_datapoints = 50
    datapoints = np.random.randint(1,100,size=(number_of_datapoints,2))
    datapoints = np.array(list(set([tuple(x) for x in datapoints])))

    convex_hulls = RecurGrahamScan(GrahamScan, datapoints).execute(make_plot=True)
    print(f'{convex_hulls = }')
    plt.show()

