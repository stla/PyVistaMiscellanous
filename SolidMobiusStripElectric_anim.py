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



def f(x, y, z, a, b):
    return ((
        (x * x + y * y + 1) * (a * x * x + b * y * y)
        + z * z * (b * x * x + a * y * y)
        - 2 * (a - b) * x * y * z
        - a * b * (x * x + y * y)
    ) ** 2 
    - 4 * (x * x + y * y) * (a * x * x + b * y * y - x * y * z * (a - b)) ** 2)


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-2):2:250j, (-2):2:250j, (-1):1:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)

tex = pv.read_texture("ElectricityTexture.jpg")

b_ = np.concatenate(
    (np.linspace(0.05, 0.25, 90), np.linspace(0.25, 0.05, 90))
)
pos0, quats = satellite_motion(180, 9)

for i, q in enumerate(quats):
    # compute and assign the values
    b = b_[i]
    a = 0.5 - b
    values = f(X, Y, Z, a, b)
    grid.point_data["values"] = values.ravel(order="F")
    # compute the isosurface f(x, y, z) = 0
    isosurf = grid.contour(isosurfaces=[0])
    mesh = isosurf.extract_geometry()
    mesh.texture_map_to_sphere(inplace=True)
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.add_background_image("SpaceBackground.png")
    pltr.set_focus(mesh.center)
    pltr.set_position(pos0)
    head, attitude, bank = quaternion2hab(q)
    pltr.camera.roll = bank
    pltr.camera.azimuth = attitude
    pltr.camera.elevation = head
    pltr.camera.zoom(1.1)
    pltr.add_mesh(
        mesh, texture=tex, specular=25 
    )
    pltr.show(screenshot=pngname)
    

os.system(
    "magick convert -dispose previous -delay 1x8 zzpic*.png SolidMobiusStripElectric.gif"    
)
