# -*- coding: utf-8 -*-
import os
import numpy as np
import pyvista as pv
import cmath
from math import sqrt, pi

def stereo(phi, theta):
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)/2 + 0.5
    return x/2/(1/2-z) + 1j*y/2/(1/2-z)
    #return np.sin(phi)/(1 - np.cos(phi)) * (np.cos(theta) + 1j * np.sin(theta))

p = 3
r = 2 * sqrt(2*p - 1) / (p-1)
def Phip(z):
    v = np.array([z**(2*p-1)-z, -1j*(z**(2*p-1)+z), (p-1)/p*(z**(2*p)+1)])
    v = v * 1j / (z**(2*p) + r*z**p - 1)
    return np.apply_along_axis(lambda x: x.real, 0, v)

def f(phi, theta):
    return Phip(stereo(phi, theta))

angle_u = np.linspace(0.1, pi-0.01, 200) 
angle_v = np.linspace(0, 2*pi, 200)
ph, th = np.meshgrid(angle_u, angle_v)
x, y, z = f(ph, th) 
grid = pv.StructuredGrid(x, y, z)
#mesh = grid.extract_geometry()#.clean(tolerance=1e-6)
