from math import sqrt
import numpy as np
import pyvista as pv

phi = (1 + sqrt(5)) / 2
def f(x, y, z):
    return (
        (18+30*phi) * x**4 * y**4
        + (-36 - 30*phi + 44*phi**2 - 10*phi**3) * x**2
        + (-18 - 24*phi + 10*phi**2) * x**6
        + (3 + 5*phi) * x**8
        + (36 + 60*phi) * x**2 * y**2 * z**4
        + (12 + 20*phi) * x**2 * y**6
        + phi
        + (-16 + 8*phi**4 - 8*phi**8 + 16*phi**12) * x**2 * y**4 * z**4
        + (8 * phi**8) * y**2 * z**8
        + (-18 - 24*phi + 10*phi**2) * z**6
        + (-8*phi**4 - 16*phi**8) * x**6 * z**4
        + (16*phi**4 + 8*phi**8) * x**6 * y**4
        + (-8*phi**4) * y**8 * z**2
        + (-18 - 24*phi + 10*phi**2) * y**6
        + (12 + 20*phi) * x**2 * z**6
        + (36 + 60*phi) * x**4 * y**2 * z**2
        + (36 + 60*phi) * x**2 * y**4 * z**2
        + (8 + 16*phi**4 - 16*phi**8 - 8*phi**12) * x**2 * y**2 * z**6
        + (-54 - 72*phi + 30*phi**2) * y**4 * z**2
        + (-8*phi**4) * x**8 * y**2
        + (16*phi**4 + 8*phi**8) * y**6 * z**4
        + (12 + 20*phi) * y**2 * z**6
        + (3 + 5*phi) * z**8
        + (-8*phi**4) * x**2 * z**8
        + (39 + 41*phi - 37*phi**2 + 5*phi**3) * z**4
        + (-54 - 72*phi + 30*phi**2) * x**2 * y**4
        + (8 + 16*phi**4 - 16*phi**8 - 8*phi**12) * x**6 * y**2 * z**2
        + (-54 - 72*phi + 30*phi**2) * x**2 * z**4
        + (12 + 20*phi) * x**6 * z**2
        + (-16 + 8*phi**4 - 8*phi**8 + 16*phi**12) * x**4 * y**2 * z**4
        + (16*phi**4 + 8*phi**8) * x**4 * z**6
        + (39 + 41*phi - 37*phi**2 + 5*phi**3) * y**4
        + (-36 - 30*phi + 44*phi**2 - 10*phi**3) * z**2
        + (8*phi**8) * x**2 * y**8
        + (12 + 20*phi) * y**6 * z**2
        + (8*phi**8) * x**8 * z**2
        + (-36 - 30*phi + 44*phi**2 - 10*phi**3) * y**2
        + (12 + 20*phi) * x**6 * y**2
        + (-8*phi**4 - 16*phi**8) * y**4 * z**6
        + (-16 + 8*phi**4 - 8*phi**8 + 16*phi**12) * x**4 * y**4 * z**2
        + (78 + 82*phi - 74*phi**2 + 10*phi**3) * x**2 * z**2
        + (18 + 30*phi) * x**4 * z**4
        + (-8*phi**4 - 16*phi**8) * x**4 * y**6
        + (-54 - 72*phi + 30*phi**2) * x**4 * y**2
        + (-54 - 72*phi + 30*phi**2) * x**4 * z**2
        + (-54 - 72*phi + 30*phi**2) * y**2 * z**4
        + (78 + 82*phi - 74*phi**2 + 10*phi**3) * x**2 * y**2
        + (-108 - 144*phi + 60*phi**2) * x**2 * y**2 * z**2
        + (18 + 30*phi) * y**4 * z**4
        + (3 + 5*phi) * y**8
        + (78 + 82*phi - 74*phi**2 + 10*phi**3) * y**2 * z**2
        + (8 + 16*phi**4 - 16*phi**8 - 8*phi**12) * x**2 * y**6 * z**2
        + (39 + 41*phi - 37*phi**2 + 5*phi**3) * x**4
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-2):2:300j, (-2):2:300j, (-2):2:300j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)

values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F") 
# compute one isosurface
isosurf = grid.contour(isosurfaces=[0])

mesh = isosurf.extract_geometry()
lengths = np.linalg.norm(mesh.points, axis=1)
toremove = lengths > sqrt((5+sqrt(5))/2)
mesh2, idx = mesh.remove_points(toremove)
mesh2.plot(smooth_shading=True, color="hotpink")

