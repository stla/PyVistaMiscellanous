# -*- coding: utf-8 -*-
# https://mathmod.deviantart.com/art/Pseudo-Hopf-Tori-565531249
import os
from math import pi, atan2, asin, sqrt, cos, sin
import numpy as np
import pyvista as pv
import quaternion


def quaternion2hab(q):
    "Quaternion to heading, attitude, bank"
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


cu = 0.0000000000001
cv = 0.0000000000001
N = 3

def Fx(u, v):
    return -np.cos(u+v) / (sqrt(2)+np.cos(v-u))

def DFx(u, v):
    DFxu = (Fx(u,v)-Fx(u+cu,v))/cu
    DFxv = (Fx(u,v)-Fx(u,v+cv))/cv
    return (DFxu, DFxv)

def Fy(u, v):
    return np.sin(v-u) / (sqrt(2)+np.cos(v-u))

def DFy(u, v):
    DFyu = (Fy(u,v)-Fy(u+cu,v))/cu
    DFyv = (Fy(u,v)-Fy(u,v+cv))/cv
    return (DFyu, DFyv)

def Fz(u, v):
    return np.sin(u+v) / (sqrt(2)+np.cos(v-u))

def DFz(u, v):
    DFzu = (Fz(u,v)-Fz(u+cu,v))/cu
    DFzv = (Fz(u,v)-Fz(u,v+cv))/cv
    return (DFzu, DFzv)

def n1(u, v):
    dfyu, dfyv = DFy(u, v)
    dfzu, dfzv = DFz(u, v)
    return dfyu*dfzv - dfzu*dfyv

def n2(u, v):
    dfxu, dfxv = DFx(u, v)
    dfzu, dfzv = DFz(u, v)
    return dfzu*dfxv - dfxu*dfzv

def n3(u, v):
    dfxu, dfxv = DFx(u, v)
    dfyu, dfyv = DFy(u, v)
    return dfxu*dfyv - dfyu*dfxv

def f(u, v):
    r = np.sqrt(n1(u,v)**2+n2(u,v)**2+n3(u,v)**2)
    t = (np.abs(np.sin(15*u)*np.cos(15*v)))**7 + 0.4*np.sin(2*N*u)
    tr = t / r
    return np.array([
        Fx(u, v) + tr*n1(u, v), 
        Fy(u, v) + tr*n2(u, v), 
        Fz(u, v) + tr*n3(u, v)
    ])

x = np.linspace(0, 2*pi, 500) 
U, V = np.meshgrid(x, x)
X, Y, Z = f(U, V)
grid = pv.StructuredGrid(X, Y, Z)
mesh = grid.extract_geometry()#.clean(tolerance=1e-6)
mesh["dist"] = np.linalg.norm(mesh.points, axis=1)


pos0, quats = satellite_motion(120, 12)

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
    pltr.camera.zoom(1)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=15, cmap="plasma", 
        log_scale=False, show_scalar_bar=False, flip_scalars=False 
    )
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=9 -o pseudoHopfTorus.gif"    
)
