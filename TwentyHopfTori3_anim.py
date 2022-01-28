# -*- coding: utf-8 -*-
import os
from math import pi, sqrt
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
mesh.scale((0.02, 0.02, 0.02), inplace=True)
dists = np.linalg.norm(mesh.points, axis=1)
dists = 2*pi * (dists - dists.min()) / (dists.max() - dists.min())


# twenty vertices ####
phi = (1+sqrt(5))/2
a = 1/sqrt(3)
b = a/phi 
c = a*phi
vertices = np.array([
    [ a,  a,  a], 
    [ a,  a, -a],
    [ a, -a,  a],
    [-a, -a,  a],
    [-a,  a, -a],
    [-a,  a,  a],
    [ 0,  b, -c], 
    [ 0, -b, -c], 
    [ 0, -b,  c],
    [ c,  0, -b],
    [-c,  0, -b],
    [-c,  0,  b],
    [ b,  c,  0],
    [ b, -c,  0],
    [-b, -c,  0],
    [-b,  c,  0],
    [ 0,  b,  c],
    [ a, -a, -a],
    [ c,  0,  b],
    [-a, -a, -a]
  ])

def Reorient_Trans(Axis1, Axis2):
  vX1 = Axis1 #/ np.linalg.norm(Axis1)
  vX2 = Axis2 #/ np.linalg.norm(Axis2)
  Y = np.cross(vX1, vX2)
  vY = Y / np.linalg.norm(Y)
  Z1 = np.cross(vX1, vY)
  vZ1 = Z1 / np.linalg.norm(Z1)
  Z2 = np.cross(vX2, vY)
  vZ2 = Z2 / np.linalg.norm(Z2)
  M1 = np.transpose(np.array([vX1, vY, vZ1]))
  M2 = np.array([vX2, vY, vZ2])
  M = np.matmul(M1, M2)
  return np.transpose(
      np.column_stack(
          (np.vstack((M, np.zeros((1,3)))), np.transpose(np.array([0,0,0,1])))
      )
  )
  

cmaps = [
    "flag", 
    "prism", 
    "ocean", 
    "gist_earth", 
    "terrain",
    "gist_stern", 
    "gnuplot", 
    "gnuplot2", 
    "CMRmap",
    "cubehelix", 
    "brg", 
    "gist_rainbow", 
    "rainbow", 
    "jet",
    "turbo", 
    "nipy_spectral", 
    "gist_ncar",
    "twilight",
    "magma",
    "viridis"]

nframes = 120
t_ = np.linspace(0, 2*pi, nframes, endpoint=False)
angles = np.linspace(0, 360, nframes, endpoint=False)

for i, t in enumerate(t_):
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size=[512,512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus([0, 0, 0])
    pltr.set_position([3, 3, 3])
    pltr.camera.zoom(1)
    mesh["dist"] = np.sin(dists - t)
    for j, v in enumerate(vertices):
        M = Reorient_Trans(np.array([1,0,0]), v)
        m = mesh.copy()
        m.transform(M, inplace=True)
        m.translate((v[0], v[1], v[2]), inplace=True)
        m.rotate_vector(v, angle=angles[i], inplace=True)
        pltr.add_mesh(
            m, smooth_shading=True, cmap=cmaps[j], specular=15, 
            show_scalar_bar=False
        )
        cone = pv.Cone(radius=0, direction=v, angle=1.5, height=2)
        pltr.add_mesh(cone, smooth_shading=True, color="chocolate")    
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=10 -o twentyHopfTori3.gif"    
)
