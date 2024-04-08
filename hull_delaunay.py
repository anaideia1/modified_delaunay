import operator

import numpy as np

from functools import reduce
from collections import Counter
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

from figures import Point, Triangle, Hull


class HullDelaunayInterface(ABC):
    """
    Interface class for Dependency injection for different HullDelaunay
    algorith classes.
    """
    @abstractmethod
    def __init__(self, _outer_hull: Hull, _inner_hull: Hull):
        ...

    @abstractmethod
    def execute(self):
        ...


class HullDelaunay(HullDelaunayInterface):
    """
    Class for building triangulation between point of two convex hulls,
    which met delaunay criteria and properties.
    """
    def __init__(self, _outer_hull: Hull, _inner_hull: Hull):
        self.inner_hull = _inner_hull.points
        self.outer_hull = _outer_hull.points

    @staticmethod
    def _point_is_right(
            point: Point, vector_start: Point, vector_end: Point
    ) -> bool:
        """
        Decide is point right or not relative to vector.
        :param point: point, which position we decide
        :param vector_start: start of vector
        :param vector_end: end of vector
        :return: True if point is right else False
        """
        res = (
            (vector_end.x - vector_start.x) * (point.y - vector_start.y)
            - (point.x - vector_start.x) * (vector_end.y - vector_start.y)
        )
        return res < 0

    def _point_is_visible(
            self, point: Point, left: Point, current: Point, right: Point
    ) -> bool:
        """
        Decide is outer hull point is visible for current point of inner hull
        based on vectors with neighboring inner hull points.
        :param point: outer hull point
        :param left: left inner hull neighboring inner hull point
        :param current: current inner hull point
        :param right: right inner hull neighboring inner hull point
        :return: True if point is visible for current else False
        """
        return (
            self._point_is_right(point, left, current)
            or self._point_is_right(point, current, right)
        )

    def _get_outer_visible_points(
            self, left: Point, current: Point, right: Point,
            first_ind: int = None, last_ind: int = 0
    ):
        """
        Find all outer hull points which is visible for this current point
        Do not add points which was already cut off by connecting line of
        previous inner hull point and their visible points.
        :param left: left inner hull neighboring inner hull point
        :param current: current inner hull point
        :param right: right inner hull neighboring inner hull point
        :param first_ind: leftmost index of cut off points on prev steps
        :param last_ind: rightmost index of cut off points on prev steps
        :return: all outer hull still visible points
        """
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

    def _get_initial_traignles_list(self) -> list[tuple[Point, Point, Point]]:
        """
        Build and return initial triangles using all possible visible points
        for every inner hull point with no overlay.
        :return: initial triangulation for points of two hull
        """
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

    def _check_convex(self, quad: list[Point]) -> bool:
        """
        Decide if quad polygon is convex.
        :param quad: list of points of a quad
        :return: True if convex else False
        """
        turns_right = [
            self._point_is_right(quad[j], quad[j+1], quad[j+2])
            for j in range(1-len(quad), 1)
        ]
        return len(set(turns_right)) == 1

    @staticmethod
    def _check_delaunay_criteria(quad: list[Point]) -> bool:
        """
        Decide if quad polygon built on two triangles met delaunay criteria.
        :param quad: list of points of a quad built on two triangles
                    (starts with uncommon point)
        :return: True if met criteria else False
        """
        criteria = None
        p0, p1, p2, p3 = quad[0], quad[1], quad[2], quad[3]

        cos_a = ((p0.x - p1.x) * (p0.x - p3.x)
                + (p0.y - p1.y) * (p0.y - p3.y))
        cos_b = ((p2.x - p1.x) * (p2.x - p3.x)
                + (p2.y - p1.y) * (p2.y - p3.y))

        if cos_a >= 0 and cos_b >= 0:
            criteria = True
        elif cos_a < 0 and cos_b < 0:
            criteria =  False
        else:
            sin_a = ((p0.x - p1.x) * (p0.y - p3.y)
                     - (p0.x - p3.x) * (p0.y - p1.y))
            sin_b = ((p2.x - p1.x) * (p2.y - p3.y)
                     - (p2.x - p3.x) * (p2.y - p1.y))
            criteria = (sin_a * cos_b + cos_a * sin_b) <= 0
        return criteria

    @staticmethod
    def _delaunay_transform_triangles(
            quad: list[Point]
    ) -> tuple[tuple[Point, Point, Point], tuple[Point, Point, Point]]:
        """
        Return two rebuilt triangles with reversed common and uncommon points.
        :param quad: initial quad
        :return: two new triangles
        """
        return (quad[0], quad[1], quad[2]), (quad[2], quad[3], quad[0])

    @staticmethod
    def _manage_triangles_order(
            tri1: tuple[Point, Point, Point],
            tri2: tuple[Point, Point, Point],
            next_tri: tuple[Point, Point, Point]
    ) -> tuple[tuple[Point, Point, Point], tuple[Point, Point, Point]]:
        """
        Decide second position in the pair of new rebuilt triangles for
        right computing order.
        :param tri1: first of new triangles
        :param tri2: second of new triangles
        :param next_tri: next triangle in the list
        :return: return same triangles in the right order
        """
        res = None

        points = [*tri1, *next_tri]
        common = [
            elem for elem, count in Counter(points).items() if count > 1
        ]

        if len(common) == 2:
            res = (tri2, tri1)
        else:
            res = (tri1, tri2)

        return res

    def _single_delaunay_transformation_with_backsteps(
            self,
            initial_triangles: list[tuple[Point, Point, Point]],
            index: int
    ) -> None:
        """
        Get from initial_triangles list pair of triangles by index
        (index, index+1) and transform triangles, if it's needed for them
        meeting Delaunay criteria.
        :param initial_triangles: list of triangles we should process
        :param index: index of first of pair triangles in initial_triangles
        :return: None, but change initial_triangles list
        """
        tri1, tri2 = initial_triangles[index], initial_triangles[index + 1]
        points = [*tri1, *tri2]
        common = sorted(
            [elem for elem, count in Counter(points).items() if count > 1],
            key=lambda point: point.y, reverse=True
        )
        not_common = sorted(
            [item for item in points if item not in common],
            key= lambda point: point.x
        )
        if len(common) == 2:
            ord_quad = reduce(operator.add, zip(not_common, common))
            if (
                    self._check_convex(ord_quad)
                    and not self._check_delaunay_criteria(ord_quad)
            ):
                new_triangles = self._delaunay_transform_triangles(ord_quad)
                ordered_new_triangles = self._manage_triangles_order(
                    new_triangles[0],
                    new_triangles[1],
                    initial_triangles[(index + 2) % len(initial_triangles)]
                )
                initial_triangles[index] = ordered_new_triangles[0]
                initial_triangles[index + 1] = ordered_new_triangles[1]
                try:
                    self._single_delaunay_transformation_with_backsteps(
                        initial_triangles, index - 1
                    )
                except IndexError:
                    pass


    def _delaunay_transformation(
            self, initial_triangles: list[tuple[Point, Point, Point]]
    ) -> list[tuple[Point, Point, Point]]:
        """
        Go through all triangles in initial_triangles and transform them,
        if it's needed for meeting Delaunay criteria.
        :param initial_triangles: list of triangles we should process
        :return: list of triangles, which met Delaunay criteria
        """
        for ind in range(-1, len(initial_triangles) - 1):
            self._single_delaunay_transformation_with_backsteps(
                initial_triangles, ind
            )
        return initial_triangles

    def execute(
            self, make_plot: bool = False
    ) -> list[tuple[Point, Point, Point]]:
        """
        General function, which make initial triangles between two consecutive
        hulls, then process and transform them and making plot, if it's needed.
        :param make_plot: indicate if it needed to make plot
        :return: list of triangles, which met Delaunay criteria between hulls
        """
        triangles = self._get_initial_traignles_list()

        transformed_triangles = self._delaunay_transformation(triangles)

        if make_plot:
            for tri in transformed_triangles:
                tri_plot = np.array([(p.x, p.y) for p in tri])
                tri_plot = np.append(tri_plot, tri_plot[:1], axis=0)
                plt.plot(tri_plot[:, 0], tri_plot[:, 1], c='g')

        return transformed_triangles


class InternalHullDelaunay(HullDelaunay):
    def _two_points_initial_split(self, first: Point, sec: Point) -> (int, int):
        """
        In case if there are two points in last 'hull' method for splitting
        outer hull between two points for future initial triangles building
        without overlay.
        :param first: leftmost point from two last points
        :param sec: rightmost point from two last points
        :return: start and switch point which indicates splitting borders
        of slices of outer_hull points
        """
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
                start_point = start_point % len(self.outer_hull)
                switch_point = switch_point % len(self.outer_hull)
                if start_point > switch_point:
                    start_point -= len(self.outer_hull)
                break
            prev_orient = curr_orient

        return start_point, switch_point

    def _two_points_initial_triangles(self) -> list[tuple[Point, Point, Point]]:
        """
        In case if there are two points in last 'hull' method for initial
        triangles building
        without overlay.
        :return: initial triangles for special case
        """
        first, sec = self.inner_hull[0], self.inner_hull[1]
        if first.x > sec.x or first.x == sec.x and first.y > sec.y:
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

    def _get_initial_traignles_list(self) -> list[tuple[Point, Point, Point]]:
        """
        Overwritten base class method for building initial triangles inside
        last convex hull and extra points.
        :return: initial triangles list for last hull
        """
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