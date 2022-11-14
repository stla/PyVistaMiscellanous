# -*- coding: utf-8 -*-
import os
from math import pi
import pyvista as pv
import numpy as np


def pumpkin(u, v, n, a, h):
    return np.array([
        a * np.sin(v) * ((n+1) * np.cos(u) - np.cos((n+1) * u)) / n,
        a * np.sin(v) * ((n+1) * np.sin(u) - np.sin((n+1) * u)) / n,
        h * np.cos(v)
    ])

U, V = np.meshgrid(np.linspace(0, 2*pi, 250), np.linspace(0, pi, 250))


nframes = 60
a_ = np.linspace(1/3, 1/4, nframes+1)[:nframes]

for i, a in enumerate(a_):
    X, Y, Z = pumpkin(U, V, n = 8, a = a, h = 1/4)
    grid = pv.StructuredGrid(X, Y, Z)
    mesh = grid.extract_geometry()
    mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)
    pngname = "zz/zzpic%03d.png" % i
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position((1, 1, 1))
    #pltr.camera.zoom(1)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=15, cmap="hot_r", 
        log_scale=False, show_scalar_bar=False, flip_scalars=False 
    )
    pltr.show(screenshot=pngname)

os.system(
    "convert -delay 1x10 -duplicate 1,-2-1 zz/zzpic*.png Pumpkin.gif"    
)
