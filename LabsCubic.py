# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv

def labs_cubic(x, y, z):
    return x**3 + 3*x**2 - 3*x*y**2 + 3*y**2 - 4 + z**3 + 3*z**2


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-6):5:100j, (-6):6:100j, (-6):4:100j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = labs_cubic(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")
poly = grid.extract_geometry()

# n = 100
# x_min = -6 
# y_min = -6 
# z_min = -6
# grid = pv.UniformGrid(
#     dims=(n, n, n),
#     spacing=(abs(x_min)/n*11/6, abs(y_min)/n*2, abs(z_min)/n*10/6),
#     origin=(x_min, y_min, z_min),
# )
# X, Y, Z = grid.points.T

# sample and plot
mesh = grid.contour(1, method="contour", rng=[1, 0])
dist = np.linalg.norm(mesh.points, axis=1)
mesh.plot(
    scalars=dist, smooth_shading=True, specular=5,
    cmap="plasma", show_scalar_bar=False
)