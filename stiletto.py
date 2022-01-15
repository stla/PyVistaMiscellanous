# -*- coding: utf-8 -*-
from math import pi
import pyvista as pv
import numpy as np

def f(u, v):
    return np.array([
        (2 + np.cos(u)) * np.cos(v)**3 * np.sin(v),
        (2 + np.cos(u+2*pi/3)) * np.cos(v+2*pi/3)**2 * np.sin(v+2*pi/3)**2,
        -(2 + np.cos(u-2*pi/3)) * np.cos(v+2*pi/3)**2 * np.sin(v+2*pi/3)**2
    ])

U, V = np.meshgrid(np.linspace(0, 2*pi, 300), np.linspace(0, 2*pi, 300))
X, Y, Z = f(U, V)
grid = pv.StructuredGrid(X, Y, Z)
mesh = grid.extract_geometry()
mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)

pltr = pv.Plotter(window_size = [512, 512])#, off_screen=True)
pltr.background_color = "#363940"
pltr.add_mesh(
    mesh, smooth_shading=True, cmap="viridis", specular=15, 
    show_scalar_bar=False    
)
pltr.set_focus([0.3, 0.5, 0])
pltr.set_position([3, 3, 3])
pltr.camera.zoom(1.4)
pltr.show()#screenshot="stiletto.png")