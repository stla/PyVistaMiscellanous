# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv

def labs_cubic(x, y, z):
    return x**3 + 3*x**2 - 3*x*y**2 + 3*y**2 - 4 + z**3 + 3*z**2


n = 100
x_min = -6 
y_min = -6 
z_min = -6
grid = pv.UniformGrid(
    dims=(n, n, n),
    spacing=(abs(x_min)/n*2, abs(y_min)/n*2, abs(z_min)/n*2),
    origin=(x_min, y_min, z_min)
)
X, Y, Z = grid.points.T

values = labs_cubic(X, Y, Z)
mesh = grid.contour(1, values, method='flying_edges', rng=[0.2, 0])

# sample and plot
dist = np.linalg.norm(mesh.points, axis=1)
mesh.plot(
    scalars=dist, smooth_shading=True, specular=5,
    cmap="plasma", show_scalar_bar=False
)


