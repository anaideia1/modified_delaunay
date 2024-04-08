import os.path
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

from fileIO import FileInput, FileGenerator, FileOutput
from hull_delaunay import HullDelaunay, InternalHullDelaunay
from modified_delaunay import ModifiedDelaunayTriangulation

if __name__ == '__main__':
    results = []
    step = 300
    start, finish = step, 100 * step

    while start <= finish:
        cur_res = []
        for i in range(10):
            points = FileGenerator('', start, start).generate_points()
            start_time = datetime.now()

            triangles = ModifiedDelaunayTriangulation(
                HullDelaunay, InternalHullDelaunay, points
            ).execute()

            end_time = datetime.now()
            cur_res.append((end_time - start_time).total_seconds())
        results.append([start, sum(cur_res)/len(cur_res)])
        start += step

    print(results)
    results = np.array(results)
    plt.scatter(results[:, 0], results[:, 1])
    plt.plot(results[:, 0], results[:, 1], c='g')
    plt.show()
