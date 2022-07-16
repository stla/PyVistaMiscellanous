# -*- coding: utf-8 -*-
import os
from math import cos, sin, pi
import numpy as np
import pyvista as pv

def rotate4d(alpha, beta, xi, vec):
    a = cos(xi)
    b = sin(alpha) * cos(beta) * sin(xi)
    c = sin(alpha) * sin(beta) * sin(xi)
    d = cos(alpha) * sin(xi)
    x = vec[0,:,:,:]; y = vec[1,:,:,:]; z = vec[2,:,:,:]; w = vec[3,:,:,:]
    return np.array([
      a*x - b*y - c*z - d*w,
      a*y + b*x + c*w - d*z,
      a*z - b*w + c*x + d*y,
      a*w + b*z - c*y + d*x
    ])    


def f(X, Y, Z, w0, xi, c=pi/4, d=pi/4, t=pi/4):
    W = w0 * np.ones(X.shape)
    rxyzw = rotate4d(pi/4, pi/4, xi, np.array([X, Y, Z, W]))
    x = rxyzw[0,:,:,:]
    y = rxyzw[1,:,:,:]
    z = rxyzw[2,:,:,:]
    w = rxyzw[3,:,:,:]
    return (
        (np.sqrt(
            (np.sqrt(
                (np.sqrt(w**2 
                          + ((z*cos(d) - y*sin(d))*sin(t))**2) 
                 - 10)**2) - 5)**2 + 
            (np.sqrt(x**2) - 5)**2) 
            - 2.5)**2 + (np.sqrt(
                (np.sqrt((z*sin(d) 
                          + y*cos(d))**2 
                         + ((z*cos(d) - y 
                             * sin(d))*cos(t))**2) - 5)**2) - 2.5)**2
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-20):20:300j, (-20):20:300j, (-20):20:300j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)


# b varies from 0 to 2pi
xi_ = np.linspace(0, 2*pi, 120, endpoint=False)

for i, xi in enumerate(xi_):
    values = f(X, Y, Z, 0, xi)
    grid.point_data["values"] = values.ravel(order="F")
    # compute the isosurface f(x, y, z) = 0
    isosurf = grid.contour(isosurfaces=[1])
    # make a polydata mesh
    mesh = isosurf.extract_geometry()
    # scalars = distances
    mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)
    #
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size=[512, 512], off_screen=True)
    pltr.add_background_image("SpaceBackground.png")
    pltr.set_focus((0,0,0))
    pltr.set_position((17, 17, 67))
    pltr.camera.zoom(0.75)
    pltr.add_mesh(
        mesh,
        smooth_shading=True,
        specular=15,
        cmap="nipy_spectral",
        log_scale=False,
        show_scalar_bar=False,
        flip_scalars=False,
    )
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=12 -o ICN5D_topo_changes_isoclinic2.gif"
)
