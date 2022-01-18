# -*- coding: utf-8 -*-
import os
from math import pi, sqrt, cos, sin
import numpy as np
import pyvista as pv

def f(x, y, z, t):
    cos_t = cos(t)
    sin_t = sin(t)
    return (
        (
            ((x * cos_t)**2 + y**2 + z**2 + (x * sin_t)**2 + 145 / 3)
           **2
            - 4 * (9 * z**2 + 16 * (x * sin_t)**2)
        )
       **2
        * (
            ((x * cos_t)**2 + y**2 + z**2 + (x * sin_t)**2 + 145 / 3)
           **2
            + 296 * ((x * cos_t)**2 + y**2)
            - 4 * (9 * z**2 + 16 * (x * sin_t)**2)
        )
        - 16
        * ((x * cos_t)**2 + y**2)
        * ((x * cos_t)**2 + y**2 + z**2 + (x * sin_t)**2 + 145 / 3)**2
        * (
            37
            * ((x * cos_t)**2 + y**2 + z**2 + (x * sin_t)**2 + 145 / 3)
           **2
            - 1369 * ((x * cos_t)**2 + y**2)
            - 7 * (225 * z**2 + 448 * (x * sin_t)**2)
        )
        - 16
        * sqrt(3)
        / 9
        * ((x * cos_t)**3 - 3 * (x * cos_t) * y**2)
        * (
            110
            * ((x * cos_t)**2 + y**2 + z**2 + (x * sin_t)**2 + 145 / 3)
           **3
            - 148
            * ((x * cos_t)**2 + y**2 + z**2 + (x * sin_t)**2 + 145 / 3)
            * (
                110 * (x * cos_t)**2
                + 110 * y**2
                - 297 * z**2
                + 480 * (x * sin_t)**2
            )
        )
        - 64
        * ((x * cos_t)**2 + y**2)
        * (
            3 * (729 * z**4 + 4096 * (x * sin_t)**4)
            + 168
            * ((x * cos_t)**2 + y**2)
            * (15 * z**2 - 22 * (x * sin_t)**2)
        )
        + 64
        * (
            12100 / 27 * ((x * cos_t)**3 - 3 * (x * cos_t) * y**2)**2
            - 7056 * (3 * (x * cos_t)**2 * y - y**3)**2
        )
        - 592240896 * z**2 * (x * sin_t)**2
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-8):9:250j, (-9):9:250j, (-6):6:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)


# t varies from 0 to pi/2
t_ = np.linspace(0, pi/2, 60)

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
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position((40, 15, -20))
    pltr.camera.roll = 90
    pltr.camera.zoom(1.2)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=15, cmap="turbo", log_scale=False,
        show_scalar_bar=False, flip_scalars=False 
    )
    pltr.show(screenshot=pngname)

os.system(
    "magick convert -dispose previous -delay 1x9 -duplicate 1,-2-1 zzpic*.png ICN5D_cage.gif"    
)

