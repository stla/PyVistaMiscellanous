# -*- coding: utf-8 -*-
from math import sqrt, cos, sin, tan
import os
from planegeometry.geometry import Mobius, Circle, unimodular_matrices
import numpy as np
import pyvista as pv

# Möbius transformations
T = Mobius(np.array([[0,-1], [1,0]]))
U = Mobius(np.array([[1,1], [0,1]]))
R = U.compose(T)
# R**t, generalized power
def Rt(t):
    return R.gpower(t)

# starting circles
I = Circle((0, 1.5), 0.5)
TI = T.transform_circle(I)

# modified Cayley transformation
Phi = Mobius(np.array([[1j, 1], [1, 1j]]))

# function to draw a circle
def draw_sphere(pltr, C):
    x, y = C.center
    sphere = pv.Sphere(C.radius, (x, y, 0))
    pltr.add_mesh(sphere, color="magenta", specular=15, smooth_shading=True)
    
def draw_pair(pltr, M, u, compose=False):
    if compose:
        M = M.compose(T)
    A = M.compose(Rt(u)).compose(Phi)
    C = A.transform_circle(I)
    draw_sphere(pltr, C)
    C = A.transform_circle(TI)
    draw_sphere(pltr, C)
    if not compose:
        draw_pair(pltr, M, u, compose=True)

n = 8
transfos = unimodular_matrices(n)

u_ = np.linspace(0, 3, 181)[:180]

for i, u in enumerate(u_):
    pltr = pv.Plotter(window_size=[512,512], off_screen=True)
    pltr.set_background("#363940")
    for transfo in transfos:
        M = Mobius(transfo)
        draw_pair(pltr, M, u)
        M = M.inverse()
        draw_pair(pltr, M, u)
        np.fill_diagonal(transfo, -np.diag(transfo))    
        M = Mobius(transfo)
        draw_pair(pltr, M, u)
        M = M.inverse()
        draw_pair(pltr, M, u)
        d = np.diag(transfo)
        if d[0] != d[1]:
            np.fill_diagonal(transfo, (d[1], d[0]))
            M = Mobius(transfo)
            draw_pair(pltr, M, u)
            M = M.inverse()
            draw_pair(pltr, M, u)
    pngname = "zpic_%03d.png" % i
    pltr.set_focus([0, 0, 0])
    pltr.set_position([-2, -2, 7])
    pltr.camera.zoom(1.6)
    pltr.show(screenshot=pngname)
        

os.system(
    "magick convert -dispose previous -loop 0 -delay 7 zpic_*.png ModularTessellation.gif"    
) 