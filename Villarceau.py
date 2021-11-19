# -*- coding: utf-8 -*-
from math import sqrt, pi, cos, sin
import numpy as np
import pyvista as pv
import seaborn as sb


# http://arkadiusz-jadczyk.eu/blog/tag/tennis-racket/
def f(x, y, z, t):
    r2 = x*x + y*y + z*z
    d = 1 + r2 + (1-r2)*cos(t) + 2*z*sin(t)
    return np.array([
        2*(x*cos(t)+y*sin(t)),
        2*(y*cos(t)-x*sin(t)),
        2*z*cos(t)-(1-r2)*sin(t)
    ])/d


colors = sb.color_palette(palette="turbo", n_colors=500)
t_ = np.linspace(0, 2*pi, 250)
thetas = np.linspace(0, 100*pi, 500)
pltr = pv.Plotter(window_size=[512,512], off_screen=True)
pltr.set_focus([0, 0, 0])
pltr.set_position([0, -9, 12])
pltr.camera.zoom(1.5)
for i, theta in enumerate(thetas):
    x = 0.5*cos(theta)
    y = 0.5*sin(theta)
    pts = np.array([f(x,y,0,t) for t in t_])
    spline = pv.Spline(pts, 1000)
    pltr.add_mesh(spline.tube(radius=0.025), color=colors[i])
    png = "zpic_%03d.png" % i
    pltr.screenshot(png)
pltr.close()
