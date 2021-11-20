# -*- coding: utf-8 -*-
import os
from math import sqrt, pi, sin, cos, acos
import pyvista as pv
import numpy as np
from numdifftools import Gradient, Hessian
import bisect

# cyclidoidal helix curve
def ccld(t, a, c, d, w):
    b = sqrt(a*a-c*c)
    den = a - c * cos(t) * cos(t/w)
    return np.array([
        d * (c - a * cos(t/w) * cos(t)) + b * b * cos(t/w),
        b * sin(t/w) * (a - d * cos(t)),
        b * sin(t) * (c * cos(t/w) - d)
    ]) / den

def fx(t, a, c, d, w):
    b = sqrt(a*a-c*c)
    return (
        (d * (c - a * cos(t/w) * cos(t)) + b * b * cos(t/w)) 
        / (a - c * cos(t) * cos(t/w))
    )

def fy(t, a, c, d, w):
    b = sqrt(a*a-c*c)
    return (
        (b * sin(t/w) * (a - d * cos(t))) 
        / (a - c * cos(t) * cos(t/w))
    )

def fz(t, a, c, d, w):
    b = sqrt(a*a-c*c)
    return (
        (b * sin(t) * (c * cos(t/w) - d)) 
        / (a - c * cos(t) * cos(t/w))
    )

# derivative (tangent)
def dccld(t, a, c, d, w):
    lambdax = lambda x: fx(x, a, c, d, w)
    lambday = lambda x: fy(x, a, c, d, w)
    lambdaz = lambda x: fz(x, a, c, d, w)
    return np.array([
        Gradient(lambdax)(t),
        Gradient(lambday)(t),
        Gradient(lambdaz)(t)
    ])

# second derivative (normal)
def ddccld(t, a, c, d, w):
    lambdax = lambda x: fx(x, a, c, d, w)
    lambday = lambda x: fy(x, a, c, d, w)
    lambdaz = lambda x: fz(x, a, c, d, w)
    v = np.array([
        Hessian(lambdax)(t),
        Hessian(lambday)(t),
        Hessian(lambdaz)(t)
    ]).flatten()
    return v / np.linalg.norm(v)

# binormal
def bnrml(t, a, c, d, w):
    v = np.cross(dccld(t, a, c, d, w), ddccld(t, a, c, d, w))
    return v / np.linalg.norm(v)

###################
def scos(x, alpha):
    cosx = cos(x)
    return cosx**alpha if cosx >= 0 else -(-cosx)**alpha

def ssin(x, alpha):
    sinx = sin(x)
    return sinx**alpha if sinx >= 0 else -(-sinx)**alpha

def CyclidoidalTubularHelixMesh(a, c, d, w, r, nu, nv, alpha, twists):
    vs = np.empty((nu*nv, 3), dtype=float)
    #colors <- matrix(NA_character_, nrow = 3L, ncol = nu*nv)
    u_ = np.linspace(0, w*2*pi, nu+1)[:nu]
    v_ = np.linspace(0, 2*pi, nv+1)[:nv]
    distances = []
    breakpoints = pi*(1/16 + np.linspace(0, 2, 17)) % (2*pi)
    for i in range(nu):
        u = u_[i]
        h = ccld(u, a, c, d, w)
        dd = ddccld(u, a, c, d, w)
        d1 = dccld(u, a, c, d, w)
        bn = np.cross(d1, dd)
        bn = bn / np.linalg.norm(bn)
        for j in range(nv):
            v = v_[j]
            k = bisect.bisect_left(breakpoints, v) % 2
            distances.append(k)
            vs[i*nv+j, :] = (
                h
                + r*(scos(v,alpha)
                     * (cos(twists*u)*dd
                        + sin(twists*u)*bn)
                     + ssin(v,alpha)
                     * (-sin(twists*u)*dd
                        + cos(twists*u)*bn))
            ) 
      # colors[,(i-1)*nv+j] <- 
      #   ifelse(findInterval(v, pi*(1/16+seq(0,2,len=17))) %% 2 == 0, 
      #          "#440154FF", "#FDE725FF") 
    tris1 = np.empty((nu*nv, 3), dtype=int)
    tris2 = np.empty((nu*nv, 3), dtype=int)
    for i in range(nu):
        ip1 = (i+1) % nu
        for j in range(nv):
            jp1 = (j+1) % nv
            tris1[i*nv+j, :] = np.array([i*nv+j, i*nv+jp1, ip1*nv+j])   
            tris2[i*nv+j, :] = np.array([i*nv+jp1, ip1*nv+jp1, ip1*nv+j])
    threes = np.full((2*nu*nv, 1), 3)
    indices = np.hstack((threes, np.vstack((tris1, tris2)))).flatten()
    mesh = pv.PolyData(vs, indices)
    mesh.point_data["distance"] = np.asarray(distances)
    return mesh

mesh = CyclidoidalTubularHelixMesh(a=1.94, c=0.34, d=0.56, w=15, r=0.3, nu=15*40, nv=40, alpha=1, twists=2)

mesh.plot(
    smooth_shading=True, show_scalar_bar=False, show_axes=False,
    window_size=[512,512], screenshot="CyclidoidalTubularHelix.png"
)