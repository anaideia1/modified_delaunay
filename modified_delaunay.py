import numpy as np

from matplotlib import pyplot as plt

from graham_scan import RecurGrahamScan, GrahamScan
from hull_delaunay import (
    HullDelaunay,
    InternalHullDelaunay,
    HullDelaunayInterface,
)
from figures import Point, Triangle, Hull



class ModifiedDelaunayTriangulation:
    """
    Main class for modified Delaunay triangulation algorithm
    """
    def __init__(
            self,
            _hull_delaunay: HullDelaunayInterface,
            _internal_hull_delaunay: HullDelaunayInterface,
            _points: set[Point] = None
    ) -> None:
        self.hull_delaunay_class = _hull_delaunay
        self.internal_hull_delaunay_class = _internal_hull_delaunay
        self.datapoints = [] if _points is None else _points

    def execute(self, make_plot: bool = False) -> list[Triangle]:
        """
        General function, which  recursively build convex hulls for set of
        points, then make initial triangles between each two consecutive
        hulls, after that process and transform them for meeting Delaunay
        criteria and making plot, if it's needed.
        :param make_plot: indicate if it needed to make plot
        :return: list of triangles, which met Delaunay criteria between hulls
        """
        res_triangles = []

        convex_hulls = RecurGrahamScan(
            GrahamScan, self.datapoints
        ).execute(make_plot=make_plot)

        for i in range(1, len(convex_hulls)):
            hull_class = None
            if i == len(convex_hulls) - 1:
                hull_class = self.internal_hull_delaunay_class
            else:
                hull_class = self.hull_delaunay_class

            triangles = hull_class(
                convex_hulls[i - 1], convex_hulls[i]
            ).execute(make_plot=make_plot)

            res_triangles.extend(triangles)

        return [Triangle(*triangle) for triangle in res_triangles]


if __name__ == '__main__':
    number_of_datapoints = 100
    datapoints = np.random.randint(1,100,size=(number_of_datapoints,2))
    datapoints = {Point(*point) for point in datapoints}
    ModifiedDelaunayTriangulation(
        HullDelaunay, InternalHullDelaunay, datapoints
    ).execute(make_plot=True)

    plt.show()
