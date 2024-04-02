import operator
import numpy as np

from functools import reduce
from collections import Counter
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

from graham_scan import RecurGrahamScan, GrahamScan


class HullDelaunayInterface(ABC):
    @abstractmethod
    def __init__(self, _datapoints):
        ...

    @abstractmethod
    def execute(self):
        ...


class HullDelaunay(HullDelaunayInterface):
    def __init__(self, _outer_hull, _inner_hull):
        self.inner_hull = _inner_hull
        self.outer_hull = _outer_hull

    @staticmethod
    def _point_is_right(point, vector_start, vector_end):
        res = (
            (vector_end[0] - vector_start[0]) * (point[1] - vector_start[1])
            - (point[0] - vector_start[0]) * (vector_end[1] - vector_start[1])
        )
        return res < 0

    def _point_is_visible(self, point, left, current, right):
        return (
            self._point_is_right(point, left, current)
            or self._point_is_right(point, current, right)
        )

    def _get_outer_visible_points(
            self, left, current, right, first_ind=None, last_ind=0
    ):
        res = []
        last_ind = last_ind - 1 if last_ind else 0
        finded = False
        first_ind_int = first_ind if first_ind else 0

        while last_ind < len(self.outer_hull) + first_ind_int + 1:
            ind = last_ind % len(self.outer_hull)
            if self._point_is_visible(
                    self.outer_hull[ind], left, current, right
            ):
                if first_ind is None:
                    first_ind = ind
                res.append(self.outer_hull[ind])
                finded = True
            elif finded:
                break
            last_ind += 1

        return res, first_ind, last_ind

    def _get_initial_traignles_list(self):
        res_triangles = []
        first_ind, last_ind = None, 0

        for ind in range(1 - len(self.inner_hull), 1):
            visible_points, first_ind, last_ind = self._get_outer_visible_points(
                left=self.inner_hull[ind - 1],
                current=self.inner_hull[ind],
                right=self.inner_hull[ind + 1],
                first_ind=first_ind,
                last_ind=last_ind,
            )
            curr_triangles = []
            if visible_points:
                curr_triangles = [
                    (
                        self.inner_hull[ind],
                        visible_points[outer_ind],
                        visible_points[outer_ind + 1]
                    )
                    for outer_ind, item in enumerate(visible_points[:-1])
                ]
                curr_triangles.append((
                    self.inner_hull[ind + 1],
                    self.inner_hull[ind],
                    visible_points[-1],
                ))
            res_triangles.extend(curr_triangles)

        return res_triangles

    def _check_convex(self, quad):
        turns_right = [
            self._point_is_right(quad[j], quad[j+1], quad[j+2])
            for j in range(1-len(quad), 1)
        ]
        return len(set(turns_right)) == 1

    @staticmethod
    def _check_delaunay_criteria(quad):
        criteria = None
        p0, p1, p2, p3 = quad[0], quad[1], quad[2], quad[3]

        cos_a = ((p0[0] - p1[0]) * (p0[0] - p3[0])
                + (p0[1] - p1[1]) * (p0[1] - p3[1]))
        cos_b = ((p2[0] - p1[0]) * (p2[0] - p3[0])
                + (p2[1] - p1[1]) * (p2[1] - p3[1]))

        if cos_a >= 0 and cos_b >= 0:
            criteria = True
        elif cos_a < 0 and cos_b < 0:
            criteria =  False
        else:
            sin_a = ((p0[0] - p1[0]) * (p0[1] - p3[1])
                     - (p0[0] - p3[0]) * (p0[1] - p1[1]))
            sin_b = ((p2[0] - p1[0]) * (p2[1] - p3[1])
                     - (p2[0] - p3[0]) * (p2[1] - p1[1]))
            criteria = (sin_a * cos_b + cos_a * sin_b) <= 0
        return criteria

    def _delaunay_transform_triangles(self, quad):
        quad = [np.array(item) for item in quad]
        return (quad[0], quad[1], quad[2]), (quad[2], quad[3], quad[0])

    def _manage_triangles_order(self, tri1, tri2, next_tri):
        res = None

        points = [tuple(x) for x in [*tri1, *next_tri]]
        common = [
            elem for elem, count in Counter(points).items() if count > 1
        ]

        if len(common) == 2:
            res = (tri2, tri1)
        else:
            res = (tri1, tri2)

        return res

    def _single_delaunay_transformation_with_backsteps(self, initial_triangles, index):
        tri1, tri2 = initial_triangles[index], initial_triangles[index + 1]
        points = [tuple(x) for x in [*tri1, *tri2]]
        common = [
            elem for elem, count in Counter(points).items() if count > 1
        ]
        not_common = [
            item for item in points if item not in common
        ]
        if len(common) == 2:
            ord_quad = reduce(operator.add, zip(not_common, common))
            if (
                    self._check_convex(ord_quad)
                    and not self._check_delaunay_criteria(ord_quad)
            ):
                print(f'{ord_quad = }')
                new_triangles = self._delaunay_transform_triangles(ord_quad)
                ordered_new_triangles = self._manage_triangles_order(
                    new_triangles[0],
                    new_triangles[1],
                    initial_triangles[(index + 2) % len(initial_triangles)]
                )
                initial_triangles[index] = ordered_new_triangles[0]
                initial_triangles[index + 1] = ordered_new_triangles[1]
                self._single_delaunay_transformation_with_backsteps(
                    initial_triangles, index - 1
                )


    def _delaunay_transformation(self, initial_triangles):
        for ind in range(-1, len(initial_triangles) - 1):
            self._single_delaunay_transformation_with_backsteps(
                initial_triangles, ind
            )
        return initial_triangles

    def execute(self, make_plot=False):
        triangles = self._get_initial_traignles_list()

        transformed_triangles = self._delaunay_transformation(triangles)

        if make_plot:
            for tri in transformed_triangles:
                tri_plot = np.array(tri)
                tri_plot = np.append(tri_plot, tri_plot[:1], axis=0)
                plt.plot(tri_plot[:, 0], tri_plot[:, 1], c='g')

        return transformed_triangles


class InternalHullDelaunay(HullDelaunay):
    def _two_points_initial_split(self, first, sec):
        start_point, switch_point = None, None

        prev_orient = self._point_is_right(self.outer_hull[0], first, sec)
        for i in range(1, 2 * len(self.outer_hull)):
            curr_orient = self._point_is_right(
                self.outer_hull[i % len(self.outer_hull)], first, sec
            )
            if curr_orient != prev_orient:
                if curr_orient:
                    start_point = i
                else:
                    switch_point = i
            if start_point is not None and switch_point is not None:
                if start_point > switch_point:
                    start_point -= len(self.outer_hull)
                break
            prev_orient = curr_orient

        return start_point, switch_point

    def _two_points_initial_triangles(self):
        first, sec = self.inner_hull[0], self.inner_hull[1]
        if first[0] > sec[0] or first[0] == sec[0] and first[1] > sec[1]:
            first, sec = sec, first

        start_index, switch_index = self._two_points_initial_split(first, sec)

        right_triangles = [
            (
                sec,
                self.outer_hull[ind],
                self.outer_hull[ind + 1]
            )
            for ind in range(start_index, switch_index)
        ]
        right_triangles.append(
            (sec, self.outer_hull[switch_index], first)
        )
        left_triangles = [
            (
                first,
                self.outer_hull[ind % len(self.outer_hull)],
                self.outer_hull[(ind + 1) % len(self.outer_hull)]
            )
            for ind in range(
                switch_index, start_index + len(self.outer_hull)
            )
        ]
        left_triangles.append(
            (first, self.outer_hull[start_index], sec)
        )

        return right_triangles + left_triangles

    def _get_initial_traignles_list(self):
        triangles = []
        inner_points_number = len(self.inner_hull)
        if inner_points_number == 0:
            triangles = [
                (
                    self.outer_hull[0],
                    self.outer_hull[ind],
                    self.outer_hull[ind + 1]
                )
                for ind in range(1, len(self.outer_hull) - 1)
            ]
        elif inner_points_number == 1:
            triangles = [
                (
                    self.inner_hull[0],
                    self.outer_hull[ind],
                    self.outer_hull[ind + 1]
                )
                for ind in range(-1, len(self.outer_hull) - 1)
            ]
        elif inner_points_number == 2:
            triangles = self._two_points_initial_triangles()

        return triangles


class ModifiedDelaunayTriangulation:
    def __init__(self, _hull_delaunay: HullDelaunayInterface, _internal_hull_delaunay: HullDelaunayInterface, _points=None):
        self.hull_delaunay_class = _hull_delaunay
        self.internal_hull_delaunay_class = _internal_hull_delaunay
        self.datapoints = [] if _points is None else _points

    def execute(self, make_plot=False):
        res_triangles = []

        convex_hulls = RecurGrahamScan(
            GrahamScan, self.datapoints
        ).execute(make_plot=make_plot)

        # TODO optimize then rebuild with dataclasses
        for i in range(1, len(convex_hulls)):
            hull_out = np.array(list(
                dict.fromkeys(list([tuple(x) for x in convex_hulls[i - 1]]))
            ))
            hull_in = np.array(list(
                dict.fromkeys(list([tuple(x) for x in convex_hulls[i]]))
            ))

            hull_class = None
            if i == len(convex_hulls) - 1:
                hull_class = self.internal_hull_delaunay_class
            else:
                hull_class = self.hull_delaunay_class

            triangles = hull_class(
                hull_out, hull_in
            ).execute(make_plot=make_plot)

            res_triangles.extend(triangles)

        return res_triangles


if __name__ == '__main__':
    number_of_datapoints = 100
    datapoints = np.random.randint(1,100,size=(number_of_datapoints,2))
    datapoints = np.array(list(set([tuple(x) for x in datapoints])))

    ModifiedDelaunayTriangulation(
        HullDelaunay, InternalHullDelaunay, datapoints
    ).execute(make_plot=True)

    plt.show()
