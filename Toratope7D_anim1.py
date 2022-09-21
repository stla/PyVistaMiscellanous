# -*- coding: utf-8 -*-
import os
from math import pi, cos, sin
import numpy as np
import pyvista as pv

def f(x, y, z, a):
    return (
        (np.sqrt(
            (np.sqrt((np.sqrt((x*sin(a))**2 + (z*cos(a))**2) - 5)**2) - 2.5)**2
                + (y*sin(a))**2) - 1.25)**2 
            + (np.sqrt((np.sqrt((z*sin(a))**2 + (x*cos(a))**2) - 2.5)**2
                       + (y*cos(a))**2) - 1.25)**2 
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-10):10:350j, (-3):3:250j, (-10):10:350j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)


###############################################################################
# a = 1
# values = f(X, Y, Z, a)
# grid.point_data["values"] = values.ravel(order="F")
# # compute the isosurface f(x, y, z) = 0
# isosurf = grid.contour(isosurfaces=[0.25])
# # make a polydata mesh
# mesh = isosurf.extract_geometry()
# # scalars = distances
# mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)
# #
# pltr = pv.Plotter(window_size=[512, 512])
# pltr.background_color = "#363940"
# pltr.set_focus(mesh.center)
# pltr.set_position((24, 18, 20))
# #pltr.camera.roll = 90
# pltr.camera.zoom(0.8)
# pltr.add_mesh(
#     mesh,
#     smooth_shading=True,
#     specular=15,
#     cmap="turbo",
#     log_scale=False,
#     show_scalar_bar=False,
#     flip_scalars=False,
# )
# pltr.show()



# a varies from 0 to pi
a_ = np.linspace(0, pi, 180, endpoint=False)

for i, a in enumerate(a_):
    values = f(X, Y, Z, a)
    grid.point_data["values"] = values.ravel(order="F")
    # compute the isosurface f(x, y, z) = 0
    isosurf = grid.contour(isosurfaces=[0.25])
    # make a polydata mesh
    mesh = isosurf.extract_geometry()
    # scalars = distances
    mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)
    #
    pngname = "zz/zzpic%03d.png" % i
    pltr = pv.Plotter(window_size=[512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position((24, 18, 20))
    #pltr.camera.roll = 90
    pltr.camera.zoom(0.8)
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
    "convert zz/zzpic*.png -delay 1x10 toratope7D_anim1.gif"
)
