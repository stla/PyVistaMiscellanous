# -*- coding: utf-8 -*-
import os
from math import pi, cos, sin
import pyvista as pv
import numpy as np


def satellite(t, R, alpha=3*pi/4, k=4):
    return R * np.array([
        cos(alpha) * cos(t) * cos(k*t) - sin(t) * sin(k*t),
        cos(alpha) * sin(t) * cos(k*t) + cos(t) * sin(k*t),
        sin(alpha) * cos(k*t)
    ])

def flat_torus(u, v, n):
    alpha = 0.5 + 0.5 * np.sin(n*2*v)
    return np.array([
        np.cos(alpha) * np.cos(u+v),
        np.cos(alpha) * np.sin(u+v),
        np.sin(alpha) * np.cos(u-v),
        np.sin(alpha) * np.sin(u-v)
    ])

def stereom(p):
    return np.arccos(p[3]) / np.sqrt(1-p[3]*p[3]) * p[0:3]

U, V = np.meshgrid(np.linspace(0, 2*pi, 250), np.linspace(0, pi, 250))
X, Y, Z = stereom(flat_torus(U, V, n=3))
grid = pv.StructuredGrid(X, Y, Z)
mesh = grid.extract_geometry()
mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)

nframes = 180
t_ = np.linspace(0, 2*pi, nframes+1)[:nframes]

for i, t in enumerate(t_):
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position(satellite(t, 12))
    #pltr.camera.zoom(1)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=15, cmap="hot_r", 
        log_scale=False, show_scalar_bar=False, flip_scalars=False 
    )
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=10 -o BianchiPinkallTorus.gif"    
)
