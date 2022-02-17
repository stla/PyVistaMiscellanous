# -*- coding: utf-8 -*-
import os
from math import pi, atan2, asin, sqrt, cos, sin
import pyvista as pv
import numpy as np
import quaternion


def quaternion2hab(q):
    c = 180 / pi
    t = q.x*q.y + q.z*q.w
    if t > 0.499: # north pole
        heading = 2 * atan2(q.x, q.w) * c
        attitude = 90
        bank = 0
    elif t < - 0.499: # south pole
        heading = - 2 * atan2(q.x, q.w) * c
        attitude = - 90
        bank = 0
    else:
        heading = atan2(2*(q.y*q.w - q.x*q.z) , 1 - 2*(q.y*q.y + q.z*q.z)) * c
        attitude = asin(2*t) * c
        bank = atan2(2*(q.x*q.w - q.y*q.z), 1 - 2*(q.x*q.x + q.z*q.z)) * c
    return (heading, attitude, bank)
# then the rotation is Ry(h) @ Rz(a) @ Rx(b)


def get_quaternion(u ,v): # u and v must be normalized
    "Get a unit quaternion whose corresponding rotation sends u to v"
    d = np.vdot(u, v)
    c = sqrt(1+d)
    r = 1 / sqrt(2) / c
    W = np.cross(u, v)
    arr = np.concatenate((np.array([c/sqrt(2)]), r*W))
    return quaternion.from_float_array(arr)


def satellite(t, R, alpha, k):
    return R * np.array([
        cos(alpha) * cos(t) * cos(k*t) - sin(t) * sin(k*t),
        cos(alpha) * sin(t) * cos(k*t) + cos(t) * sin(k*t),
        sin(alpha) * cos(k*t)
    ])


def satellite_motion(nframes, R, alpha=3*pi/4, k=4):
    quats = [None]*nframes
    t_ = np.linspace(0, 2*pi, nframes+1)[:nframes]
    satellite0 = satellite(0, R, alpha, k)
    A = satellite0.copy()
    q0 = quaternion.one
    quats[0] = q0
    for i in range(nframes-1):
        B = satellite(t_[i+1], R, alpha, k)
        q1 = get_quaternion(A/R, B/R) * q0
        quats[i+1] = q1
        A = B
        q0 = q1
    return (satellite0, quats)


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


pos0, quats = satellite_motion(180, 9)

for i, q in enumerate(quats):
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position(pos0)
    h, a, b = quaternion2hab(q)
    pltr.camera.roll = b
    pltr.camera.azimuth = a
    pltr.camera.elevation = h
    pltr.camera.zoom(1.9)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=5, cmap="plasma", log_scale=False,
        show_scalar_bar=False, flip_scalars=False 
    )
    pltr.show(screenshot=pngname)

os.system(
    "magick convert -dispose previous -delay 1x9 zzpic*.png stiletto.gif"    
)