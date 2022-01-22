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

np.savetxt("spider_points.csv", np.asarray(spider.points), fmt="%.16g", delimiter=",")
np.savetxt("spider_faces.csv", np.asarray(spider.faces), fmt="%d", delimiter=",")
np.savetxt("spider_normals.csv", np.asarray(spider.point_normals), fmt="%.16g", delimiter=",")

from pyvista import examples
armadillo = examples.download_armadillo()
np.savetxt("armadillo_points.csv", np.asarray(armadillo.points), fmt="%.16g", delimiter=",")
np.savetxt("armadillo_faces.csv", np.asarray(armadillo.faces), fmt="%d", delimiter=",")
np.savetxt("armadillo_normals.csv", np.asarray(armadillo.point_normals), fmt="%.16g", delimiter=",")

shark = examples.download_shark()
np.savetxt("shark_points.csv", np.asarray(shark.points), fmt="%.16g", delimiter=",")
np.savetxt("shark_faces.csv", np.asarray(shark.faces), fmt="%d", delimiter=",")
np.savetxt("shark_normals.csv", np.asarray(shark.point_normals), fmt="%.16g", delimiter=",")
