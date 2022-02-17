# -*- coding: utf-8 -*-
import os
from math import pi, cos, sin, atan
import numpy as np
import pyvista as pv


def f(x, y, z, a):
    return (
        (np.sqrt(x*x + y*y + (z*cos(a))**2) - 4)**2 + (z*sin(a))**2 -0.75**2
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-8):8:250j, (-8):8:250j, (-8):8:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)


# a varies from 0 to pi
ls = np.linspace(atan(-20), atan(20), 88)
a_ = np.concatenate(([0], ls+pi/2, [pi]))

for i, a in enumerate(a_):
    values = f(X, Y, Z, a)
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
    pltr.set_position((20, 0, 20))
    #pltr.camera.roll = 90
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
    "gifski --frames=zzpic*.png --fps=9 -o ICN5D_torus_to_sphere.gif"
)
