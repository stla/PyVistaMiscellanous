# -*- coding: utf-8 -*-
import os
from math import pi, cos, sin
import numpy as np
import pyvista as pv


def f(x, y, z, q):
    return (
        (np.sqrt((np.sqrt(x**2 + (z*cos(q))**2) - 4)**2 + (z*sin(q))**2) - 2)**2 
          + (np.sqrt(y**2) - 2)**2 - 0.75
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-10):10:250j, (-10):10:250j, (-10):10:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)


# q varies from 0 to pi
q_ = np.linspace(0, 2*pi, 90)

for i, q in enumerate(q_):
    values = f(X, Y, Z, q)
    grid.point_data["values"] = values.ravel(order="F")
    # compute the isosurface f(x, y, z) = 0
    isosurf = grid.contour(isosurfaces=[0])
    # make a polydata mesh
    mesh = isosurf.extract_geometry()
    # scalars = distances
    mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)
    #
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size=[512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position((24, 18, 20))
    pltr.camera.zoom(1.2)
    pltr.add_mesh(
        mesh,
        smooth_shading=True,
        specular=15,
        cmap="turbo",
        log_scale=False,
        show_scalar_bar=False,
        flip_scalars=False,
    )
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=9 -o ICN5D_4Dtori2.gif"
)
