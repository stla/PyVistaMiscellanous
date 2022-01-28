# -*- coding: utf-8 -*-
import os
from math import pi
import numpy as np
import pyvista as pv

A = 0.44
n = 3
def Gamma(t):
	alpha = pi/2 - (pi/2-A)*np.cos(n*t)
	beta = t + A*np.sin(2*n*t)
	return np.array([
      np.sin(alpha) * np.cos(beta),
      np.sin(alpha) * np.sin(beta),
      np.cos(alpha)
	])

def HopfInverse(p, phi):
	return np.array([
      (1+p[2])*np.cos(phi),
      p[0]*np.sin(phi) - p[1]*np.cos(phi), 
      p[0]*np.cos(phi) + p[1]*np.sin(phi),
      (1+p[2])*np.sin(phi)
	]) / np.sqrt(2*(1+p[2]))

def Stereo(q):
	return 2*q[0:3] / (1-q[3])

def F(t, phi):
	return Stereo(HopfInverse(Gamma(t), phi))

def HTmesh(nu=400, nv=200):
    angle_u = np.linspace(-pi, pi, nu) 
    angle_v = np.linspace(0, pi, nv)
    u, v = np.meshgrid(angle_u, angle_v)
    x, y, z = F(u, v)
    grid = pv.StructuredGrid(x, y, z)
    return grid.extract_geometry().clean(tolerance=1e-6)

mesh = HTmesh()
dists = np.linalg.norm(mesh.points, axis=1)
dists = 2*pi * (dists - dists.min()) / (dists.max() - dists.min())


t_ = np.linspace(0, 2*pi, 60, endpoint=False)
for i, t in enumerate(t_):
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus([0, 0, 0])
    pltr.set_position((75, 0, 0))
    pltr.camera.zoom(1)
    mesh["dist"] = np.sin(dists - t)
    pltr.add_mesh(
        mesh, smooth_shading=True, cmap="flag", show_scalar_bar=False,
        specular=10
    )
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=11 -o HopfTorusMovingFlag.gif"    
)