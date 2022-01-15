# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv

def f1(u, v):
    R = np.cos(v)*np.cos(v) * np.maximum(abs(np.sin(4*u)), 0.9-0.2*abs(np.cos(8*u)))
    return [
        R*np.cos(u)*np.cos(v),
        R*np.sin(u)*np.cos(v),
        R*np.sin(v)*0.5
    ]

angle_u = np.linspace(0, 2*np.pi, 100) 
angle_v = np.linspace(0, 2*np.pi, 100)
u, v = np.meshgrid(angle_u, angle_v)
x, y, z = f1(u, v) 
grid = pv.StructuredGrid(x, y, z)
mesh = grid.extract_geometry().triangulate()

np.savetxt("flower1_points.csv", np.asarray(mesh.points), fmt="%.16g", delimiter=",")
np.savetxt("flower1_faces.csv", np.asarray(mesh.faces), fmt="%d", delimiter=",")
np.savetxt("flower1_normals.csv", np.asarray(mesh.point_normals), fmt="%.16g", delimiter=",")

