# -*- coding: utf-8 -*-
import os
from math import sqrt, pi, sin, cos, acos
import pyvista as pv
import numpy as np
from pycyclides.cyclide import linkedCyclides
import seaborn as sb

lcs = linkedCyclides(5, 0.9)
colors = sb.color_palette(palette="turbo", n_colors=5)
for cyclide in lcs:
    cyclide.scale((0.016,0.016,0.016))
    
np.random.seed(666)
ncones = 100
lengths = 1 + 0.2*np.random.rand(ncones)
gauss = np.random.normal(size=(ncones,3))
positions = gauss / np.linalg.norm(gauss, axis=1).reshape((ncones, 1)) * lengths.reshape((ncones, 1))

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
          (
              np.vstack((M, np.zeros((1,3)))), 
              np.transpose(np.array([0,0,0,1]))
          )
      )
  )

nframes = 180
angles = np.linspace(0, 360, nframes+1)[:nframes]

  
for i in range(nframes):
    file = "zscreenshot%03d.png" % i
    plotter = pv.Plotter(window_size=[512,512], off_screen=True)
    plotter.set_background("#363940")
    for v, l in zip(positions, lengths):
        M = Reorient_Trans(np.array([0,1,0]), v/l)
        for color, cyclide in zip(colors, lcs):
            m = cyclide.copy()
            m.transform(M)
            m.rotate_vector(v, angle=angles[i])
            m.translate((v[0], v[1], v[2]))
            plotter.add_mesh(m, smooth_shading=True, color=color, specular=5)
            cone = pv.Cone(center=v/2, radius=0, direction=-v/l, angle=1, height=l)
            plotter.add_mesh(cone, smooth_shading=True, color="chocolate")
            plotter.set_focus([0,0,0])
            plotter.set_position([-3,1,0])
            plotter.camera.view_angle = 50
    plotter.show(screenshot=file)
    