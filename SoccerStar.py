# -*- coding: utf-8 -*-
import os
from math import pi, cos, sin, sqrt
import pyvista as pv
import numpy as np


def satellite(t, R, alpha=3*pi/4, k=4):
    return R * np.array([
        cos(alpha) * cos(t) * cos(k*t) - sin(t) * sin(k*t),
        cos(alpha) * sin(t) * cos(k*t) + cos(t) * sin(k*t),
        sin(alpha) * cos(k*t)
    ])

def soccer_star(x, y, z):
    a1 = 100
    a2 = 100
    a3 = 100
    a4 = 100
    u = x * x + y * y + z * z
    v = (
        -z
        * (2 * x + z)
        * (
            x**4
            - x * x * z * z
            + z**4
            + 2 * (x**3 * z - x * z**3)
            + 5 * (y**4 - y * y * z * z)
            + 10 * (x * y * y * z - x * x * y * y)
        )
    )
    w = (
        (4 * x * x + z * z - 6 * x * z)
        * (
            z**4
            - 2 * z**3 * x
            - x * x * z * z
            + 2 * z * x**3
            + x**4
            - 25 * y * y * z * z
            - 30 * x * y * y * z
            - 10 * x * x * y * y
            + 5 * y**4
        )
        * (
            z**4
            + 8 * z**3 * x
            + 14 * x * x * z * z
            - 8 * z * x**3
            + x**4
            - 10 * y * y * z * z
            - 10 * x * x * y * y
            + 5 * y**4
        )
    )
    return (
        1
        + (
            (128565 + 115200 * sqrt(5)) / 1295029 * a3
            + (49231296000 * sqrt(5) - 93078919125) / 15386239549 * a4
            - a1
            - 3 * a2
            - 3
        )
        * u
        + (
            (-230400 * sqrt(5) - 257130) / 1295029 * a3
            + (238926989250 - 126373248000 * sqrt(5)) / 15386239549 * a4
            + 3 * a1
            + 8 * a2
            + 3
        )
        * u
        * u
        + (
            (115200 * sqrt(5) + 128565) / 1295029 * a3
            + (91097280000 * sqrt(5) - 172232645625) / 15386239549 * a4
            - 3 * a1
            - 6 * a2
            - 1
        )
        * u
        * u
        * u
        + (a3 + (121075 - 51200 * sqrt(5)) / 11881 * a4) * v
        + ((102400 * sqrt(5) - 242150) / 11881 - 2 * a3) * u * v
        + a1 * u**4
        + a2 * u**5
        + a3 * u * u * v
        + a4 * w
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-1.5):1.5:250j, (-1.5):1.5:250j, (-1.5):1.5:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = soccer_star(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")

isosurf = grid.contour(isosurfaces = [0])
mesh = isosurf.extract_geometry()
#pngname = "zzpic%03d.png" % i
pltr = pv.Plotter(window_size = [512, 512], off_screen=False)
pltr.background_color = "#363940"
pltr.set_focus(mesh.center)
pltr.set_position((7, 4, 2))
pltr.camera.zoom(1.3)
pltr.add_mesh(
    mesh, smooth_shading=True, specular=0.2, cmap="viridis", 
    log_scale=False, show_scalar_bar=False, flip_scalars=False 
)
pltr.show()

nframes = 180
t_ = np.linspace(0, 2*pi, nframes+1)[:nframes]
for i, t in enumerate(t_):
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position(satellite(t, 6))
    #pltr.camera.zoom(1)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=0.2, cmap="turbo", 
        log_scale=False, show_scalar_bar=False, flip_scalars=False 
    )
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=10 -o SoccerStar.gif"    
)
