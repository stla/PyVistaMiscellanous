# -*- coding: utf-8 -*-
from math import cos, sin, pi, sqrt
import numpy as np
import pyvista as pv
import bisect

def scos(x, alpha):
    cosx = cos(x)
    return cosx**alpha if cosx >= 0 else -(-cosx)**alpha

def ssin(x, alpha):
    sinx = sin(x)
    return sinx**alpha if sinx >= 0 else -(-sinx)**alpha

def torusMesh(alpha, ntwists, R, r, S=64, s=32, arc=2*pi):
    vertices = np.empty((0, 3))
    faces = np.empty((0), dtype=int)
    distances = []
    breakpoints = pi*(1/16 + np.linspace(0, 2, 17)) % (2*pi)
    for j in range(s + 1):
        v = j / s * 2 * pi
        rcos3_v = r * scos(v, alpha)
        rsin3_v = r * ssin(v, alpha)
        for i in range(S + 1):
            u = i / S * arc
            cos_u = cos(u)
            sin_u = sin(u)
            cos_2u = cos(ntwists*u)
            sin_2u = sin(ntwists*u)
            cx = R * cos_u
            cy = R * sin_u
            w = rcos3_v*cos_2u + rsin3_v*sin_2u
            vertex = np.array(
                [
                  cx + cos_u * w,
                  cy + sin_u * w,
                  rsin3_v*cos_2u - rcos3_v*sin_2u
                ]
            )
            vertices = np.vstack((vertices, vertex))
            if i < S and j < s:
                k = bisect.bisect_left(breakpoints, v) % 2
                distances.append(k)
    for j in range(s):
        for i in range(S):
            a = (S + 1) * (j + 1) + i
            b = (S + 1) * j + i
            c = (S + 1) * j + i + 1
            d = (S + 1) * (j + 1) + i + 1
            faces = np.concatenate((
              faces, np.array([3, a, b, d, 3, b, c, d])
            ), axis=0)
    mesh = pv.PolyData(vertices, faces).extract_geometry().clean(tolerance=1e-6)
    mesh.point_data["distance"] = np.asarray(distances)
    return mesh


def cyclideMesh(mu, a, c, ntwists=2, alpha=3, S=128, s=64, arc=2*pi):
    b = sqrt(a*a - c*c)
    bb = b*sqrt(mu*mu - c*c)
    omega = (a*mu + bb)/c
    Omega = np.array([omega, 0.0, 0.0])
    inversion0 = lambda M: Omega + 1/np.vdot(M, M) * M
    inversion = lambda M: inversion0(M - Omega)
    d = (a-c)*(mu-c)+bb
    r = c*c*(mu-c) / ((a+c)*(mu-c)+bb) / d
    R = c*c*(a-c) / ((a-c)*(mu+c)+bb) / d
    b2 = b*b
    omegaT = omega - (b2*(omega-c)) / ((a-c)*(mu+omega)-b2) / ((a+c)*(omega-c)+b2)
    tmesh = torusMesh(alpha, ntwists, R, r, S, s, arc)
    tmesh.points[:, 0] = tmesh.points[:, 0] + omegaT
    tmesh.points = np.apply_along_axis(inversion, 1, tmesh.points)
    return tmesh


mesh = torusMesh(3, 1, 4, 2)
mesh.plot(
  smooth_shading=True, cmap=["#440154", "#FDE725"], specular=10,
  show_scalar_bar=False ,window_size=[512,512], 
  screenshot="AstroidalTwistedTorus.png"
)

mesh = cyclideMesh(0.56, 0.94, 0.34, ntwists=1, alpha=3)
mesh.plot(
  smooth_shading=True, cmap=["#440154", "#FDE725"], specular=10,
  show_scalar_bar=False ,window_size=[512,512], 
  screenshot="AstroidalTwistedCyclide.png"
)
