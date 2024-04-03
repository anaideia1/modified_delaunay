import os.path

from matplotlib import pyplot as plt

from fileIO import FileInput, FileGenerator, FileOutput
from hull_delaunay import HullDelaunay, InternalHullDelaunay
from modified_delaunay import ModifiedDelaunayTriangulation

if __name__ == '__main__':
    input_file_name = 'input'
    output_file_name = 'output'
    if not os.path.isfile(input_file_name):
        FileGenerator(input_file_name).write_points()

    points = FileInput(input_file_name).read_points()

    triangles = ModifiedDelaunayTriangulation(
        HullDelaunay, InternalHullDelaunay, points
    ).execute(make_plot=True)

    FileOutput(output_file_name).write_triangles(triangles)
    plt.show()