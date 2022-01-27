# -*- coding: utf-8 -*-
# https://mathmod.deviantart.com/art/Pseudo-Hopf-Tori-565531249
from math import pi, sqrt
import numpy as np
import pyvista as pv

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

pltr = pv.Plotter(window_size=[512,512])
pltr.background_color = "#363940"
pltr.add_mesh(
    mesh, smooth_shading=True, cmap="plasma", specular=15, show_scalar_bar=False
)
pltr.show()