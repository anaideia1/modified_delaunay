Modified Delaunay algorithm

First step:
Splits all points on a plane with Graham Scan to recursive convex hulls (one in another).

Second step:
Building initial triangles in space between two nearby hulls on their points.

Third step:
Rebuild respective initial triangles between hulls in a way to make sure Delaunay criteria is met.

Fourth step:
Build initial triangles for innermost hull.

Fifth: 
Reconstruct some triangles if needed.

