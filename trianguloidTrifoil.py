# -*- coding: utf-8 -*-
from math import pi
import pyvista as pv
import numpy as np

def f(u, v):
    return np.array([
        2*np.sin(3*u) / (2 + np.cos(v)),
        2*(np.sin(u) + 2*np.sin(2*u)) / (2 + np.cos(v + 2*pi/3)),
        (np.cos(u) - 2*np.cos(2*u)) * (2 + np.cos(v)) * (2 + np.cos(v + 2*pi/3))/4
    ])

U, V = np.meshgrid(np.linspace(-pi, pi, 200), np.linspace(-pi, pi, 200))
X, Y, Z = f(U, V)
grid = pv.StructuredGrid(X, Y, Z)
mesh = grid.extract_geometry()
mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)

pltr = pv.Plotter(window_size = [512, 512])#, off_screen=True)
pltr.background_color = "#363940"
pltr.add_mesh(
    mesh, smooth_shading=True, cmap="turbo", specular=15, 
    show_scalar_bar=False    
)
pltr.set_focus(mesh.center)
pltr.set_position([40, 14, 14])
pltr.camera.zoom(1.7)
pltr.show()#screenshot="trianguloidTrifoil.png")