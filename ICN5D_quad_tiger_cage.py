# -*- coding: utf-8 -*-
import os
from math import cos, sin
import numpy as np
import pyvista as pv


def f(x, y, z, a, b):
    return (
        (np.sqrt((np.sqrt(x*x + (y*sin(b) + a*cos(b))**2) - 2)**2) - 1)**2 
        + (np.sqrt((np.sqrt(z*z + (y*cos(b) - a*sin(b))**2) - 2)**2) - 1)**2 
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-5):5:250j, (-5):5:250j, (-5):5:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)


# a varies from 0 to pi
a_ = np.linspace(-4.5, 4.5, 120)

for i, a in enumerate(a_):
    values = f(X, Y, Z, a, 0.785)
    grid.point_data["values"] = values.ravel(order="F")
    # compute the isosurface f(x, y, z) = 0
    isosurf = grid.contour(isosurfaces=[0.4**2])
    # make a polydata mesh
    mesh = isosurf.extract_geometry()
    # scalars = distances
    mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)
    #
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size=[512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position((12, 12, 12))
#    pltr.camera.zoom(1)
    pltr.add_mesh(
        mesh,
        smooth_shading=True,
        specular=15,
        cmap="viridis",
        log_scale=False,
        show_scalar_bar=False,
        flip_scalars=False,
    )
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=9 -o ICN5D_quad_tiger_cage.gif"
)
