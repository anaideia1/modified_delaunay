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
        res_triangles = []

        convex_hulls = RecurGrahamScan(
            GrahamScan, self.datapoints
        ).execute(make_plot=make_plot)

        # TODO optimize then rebuild with dataclasses
        for i in range(1, len(convex_hulls)):
            # hull_out = np.array(list(
            #     dict.fromkeys(list([tuple(x) for x in convex_hulls[i - 1].points]))
            # ))
            # hull_in = np.array(list(
            #     dict.fromkeys(list([tuple(x) for x in convex_hulls[i]]))
            # ))

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
    print(f'{datapoints = }')
    ModifiedDelaunayTriangulation(
        HullDelaunay, InternalHullDelaunay, datapoints
    ).execute(make_plot=True)

    plt.show()
