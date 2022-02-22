# -*- coding: utf-8 -*-
import os
from math import cos, sin, pi
import numpy as np
import pyvista as pv


def f(x, y, z, a, b, c, d, t):
    return (
        (np.sqrt(
            (np.sqrt(
                (np.sqrt((x*sin(b) + a*cos(b))**2 
                          + ((z*cos(d) - (y*cos(c) - (x*cos(b) - a*sin(b))
                                          * sin(c))*sin(d))*sin(t))**2) 
                 - 10)**2) - 5)**2 + 
            (np.sqrt((y*sin(c) + (x*cos(b) - a*sin(b))*cos(c))**2) - 5)**2) 
            - 2.5)**2 + (np.sqrt(
                (np.sqrt((z*sin(d) 
                          + (y*cos(c) - (x*cos(b) - a*sin(b)) 
                             * sin(c))*cos(d))**2 
                         + ((z*cos(d) - (y*cos(c) 
                                         - (x*cos(b) 
                                            - a*sin(b))*sin(c)) 
                             * sin(d))*cos(t))**2) - 5)**2) - 2.5)**2
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-20):20:300j, (-20):20:300j, (-20):20:300j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)


# b varies from 0 to 2pi
b_ = np.linspace(0, 2*pi, 120)

for i, b in enumerate(b_):
    values = f(X, Y, Z, 5, b, pi/4, pi/2, 0)
    grid.point_data["values"] = values.ravel(order="F")
    # compute the isosurface f(x, y, z) = 0
    isosurf = grid.contour(isosurfaces=[1])
    # make a polydata mesh
    mesh = isosurf.extract_geometry()
    # scalars = distances
    mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)
    #
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size=[512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus((0,0,0))
    pltr.set_position((47, 47, 47))
#    pltr.camera.zoom(1)
    pltr.add_mesh(
        mesh,
        smooth_shading=True,
        specular=15,
        cmap="inferno",
        log_scale=False,
        show_scalar_bar=False,
        flip_scalars=False,
    )
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=9 -o ICN5D_topo_changes.gif"
)
