# -*- coding: utf-8 -*-
import os
from math import pi, sqrt, cos, sin
import numpy as np
import pyvista as pv


def f(x, y, z, t, a=0):
    cos_t = cos(t)
    sin_t = sin(t)
    return (
        (
            (
                x ** 2
                + y ** 2
                + (z * cos_t - a * sin_t) ** 2
                + (z * sin_t + a * cos_t) ** 2
                + 145 / 3
            )
            ** 2
            - 4 * (9 * (z * cos_t - a * sin_t) ** 2 + 16 * (z * sin_t + a * cos_t) ** 2)
        )
        ** 2
        * (
            (
                x ** 2
                + y ** 2
                + (z * cos_t - a * sin_t) ** 2
                + (z * sin_t + a * cos_t) ** 2
                + 145 / 3
            )
            ** 2
            + 296 * (x ** 2 + y ** 2)
            - 4 * (9 * (z * cos_t - a * sin_t) ** 2 + 16 * (z * sin_t + a * cos_t) ** 2)
        )
        - 16
        * (x ** 2 + y ** 2)
        * (
            x ** 2
            + y ** 2
            + (z * cos_t - a * sin_t) ** 2
            + (z * sin_t + a * cos_t) ** 2
            + 145 / 3
        )
        ** 2
        * (
            37
            * (
                x ** 2
                + y ** 2
                + (z * cos_t - a * sin_t) ** 2
                + (z * sin_t + a * cos_t) ** 2
                + 145 / 3
            )
            ** 2
            - 1369 * (x ** 2 + y ** 2)
            - 7
            * (225 * (z * cos_t - a * sin_t) ** 2 + 448 * (z * sin_t + a * cos_t) ** 2)
        )
        - (16 * sqrt(3))
        / 9
        * (x ** 3 - 3 * x * y ** 2)
        * (
            110
            * (
                x ** 2
                + y ** 2
                + (z * cos_t - a * sin_t) ** 2
                + (z * sin_t + a * cos_t) ** 2
                + 145 / 3
            )
            ** 3
            - 148
            * (
                x ** 2
                + y ** 2
                + (z * cos_t - a * sin_t) ** 2
                + (z * sin_t + a * cos_t) ** 2
                + 145 / 3
            )
            * (
                110 * x ** 2
                + 110 * y ** 2
                - 297 * (z * cos_t - a * sin_t) ** 2
                + 480 * (z * sin_t + a * cos_t) ** 2
            )
        )
        - 64
        * (x ** 2 + y ** 2)
        * (
            3
            * (4096 * (z * sin_t + a * cos_t) ** 4 + 729 * (z * cos_t - a * sin_t) ** 4)
            + 168
            * (15 * (z * cos_t - a * sin_t) ** 2 - 22 * (z * sin_t + a * cos_t) ** 2)
            * (x ** 2 + y ** 2)
        )
        + 64
        * (
            12100 / 27 * (x ** 3 - 3 * x * y ** 2) ** 2
            - 7056 * (3 * x ** 2 * y - y ** 3) ** 2
        )
        - 592240896 * (z * cos_t - a * sin_t) ** 2 * (z * sin_t + a * cos_t) ** 2
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-8):8:250j, (-8):8:250j, (-8):8:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)


# t varies from 0 to pi/2
t_ = np.linspace(0, pi / 2, 60)

for i, t in enumerate(t_):
    values = f(X, Y, Z, t)
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
    pltr.set_position((40, 15, -20))
    pltr.camera.roll = 90
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
    "magick convert -dispose previous -delay 1x9 -duplicate 1,-2-1 zzpic*.png ICN5D_cage2.gif"
)
