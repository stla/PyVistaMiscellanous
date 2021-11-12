# -*- coding: utf-8 -*-
import os
from math import sqrt, sin, cos, pi, acos
import pyvista as pv
import numpy as np




# vertices --------------------------------------------------------------------
a = sqrt(3) / 2
vertices = [
    [a, 0.5, a, 0.5],
    [a, 0.5, 0.0, 1.0],
    [a, 0.5, -a, 0.5],
    [a, 0.5, -a, -0.5],
    [a, 0.5, 0.0, -1.0],
    [a, 0.5, a, -0.5],
    [0.0, 1.0, a, 0.5],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, -a, 0.5],
    [0.0, 1.0, -a, -0.5],
    [0.0, 1.0, 0.0, -1.0],
    [0.0, 1.0, a, -0.5],
    [-a, 0.5, a, 0.5],
    [-a, 0.5, 0.0, 1.0],
    [-a, 0.5, -a, 0.5],
    [-a, 0.5, -a, -0.5],
    [-a, 0.5, 0.0, -1.0],
    [-a, 0.5, a, -0.5],
    [-a, -0.5, a, 0.5],
    [-a, -0.5, 0.0, 1.0],
    [-a, -0.5, -a, 0.5],
    [-a, -0.5, -a, -0.5],
    [-a, -0.5, 0.0, -1.0],
    [-a, -0.5, a, -0.5],
    [0.0, -1.0, a, 0.5],
    [0.0, -1.0, 0.0, 1.0],
    [0.0, -1.0, -a, 0.5],
    [0.0, -1.0, -a, -0.5],
    [0.0, -1.0, 0.0, -1.0],
    [0.0, -1.0, a, -0.5],
    [a, -0.5, a, 0.5],
    [a, -0.5, 0.0, 1.0],
    [a, -0.5, -a, 0.5],
    [a, -0.5, -a, -0.5],
    [a, -0.5, 0.0, -1.0],
    [a, -0.5, a, -0.5]
  ]
facetVertices = [0, 5, 6, 30, 11, 35, 12, 17, 18, 23, 24, 29]
otherVertices = [
  1, 2, 3, 4, 7, 8, 
  9, 10, 13, 14, 15, 16, 
  19, 20, 21, 22, 25, 26, 
  27, 28, 31, 32, 33, 34]

# edges -------------------------------------------------------------------
facetEdges = [
    [0, 5],
    [0, 6],
    [0, 30],
    [5, 11],
    [5, 35],
    [6, 11],
    [6, 12],
    [11, 17],
    [12, 17],
    [12, 18],
    [17, 23],
    [18, 23],
    [18, 24],
    [23, 29],
    [24, 29],
    [24, 30],
    [29, 35],
    [30, 35]
  ]
  
otherEdges = [
    [0, 1],
    [1, 2],
    [1, 7],
    [1, 31],
    [2, 3],
    [2, 8],
    [2, 32],
    [3, 4],
    [3, 9],
    [3, 33],
    [4, 5],
    [4, 10],
    [4, 34],
    [6, 7],
    [7, 8],
    [7, 13],
    [8, 9],
    [8, 14],
    [9, 10],
    [9, 15],
    [10, 11],
    [10, 16],
    [12, 13],
    [13, 14],
    [13, 19],
    [14, 15],
    [14, 20],
    [15, 16],
    [15, 21],
    [16, 17],
    [16, 22],
    [18, 19],
    [19, 20],
    [19, 25],
    [20, 21],
    [20, 26],
    [21, 22],
    [21, 27],
    [22, 23],
    [22, 28],
    [24, 25],
    [25, 26],
    [25, 31],
    [26, 27],
    [26, 32],
    [27, 28],
    [27, 33],
    [28, 29],
    [28, 34],
    [30, 31],
    [31, 32],
    [32, 33],
    [33, 34],
    [34, 35]
  ]

def sphericalSegment(P, Q, n):
    out = [None]*(n+1)
    P_Q = Q - P
    for i in range(n+1):
      pt = P + (i/n) * P_Q
      out[i] = sqrt(2) / np.linalg.norm(pt) * pt
    return out

# rotation in 4D space (right-isoclinic) ######################################
def rotate4d(alpha, beta, xi, vec):
    a = cos(xi)
    b = sin(alpha) * cos(beta) * sin(xi)
    c = sin(alpha) * sin(beta) * sin(xi)
    d = cos(alpha) * sin(xi)
    p = vec[0]
    q = vec[1]
    r = vec[2]
    s = vec[3]
    return np.array(
        [
            a * p - b * q - c * r - d * s,
            a * q + b * p + c * s - d * r,
            a * r - b * s + c * p + d * q,
            a * s + b * r - c * q + d * p
        ]
    )

def StereographicProjection(q):
    return acos(q[3]/sqrt(2))/sqrt(2 - q[3]*q[3]) * q[0:3]

def ProjectedFacetVertices(theta, phi, xi):
    n = len(facetVertices)
    out = [None]*n
    for i in range(n):
        out[i] = StereographicProjection(
          rotate4d(theta, phi, xi, vertices[facetVertices[i]])
        )
    return out

def ProjectedOtherVertices(theta, phi, xi):
    n = len(otherVertices)
    out = [None]*n
    for i in range(n):
        out[i] = StereographicProjection(
          rotate4d(theta, phi, xi, vertices[otherVertices[i]])
        )
    return out

def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

def Edge(verts, v1, v2, theta, phi, xi):
    P = np.array(verts[v1])
    Q = np.array(verts[v2])
    PQ = sphericalSegment(P, Q, 100)
    npoints = len(PQ)
    pq = np.empty((npoints, 3), dtype=float)
    dists = np.empty(npoints, dtype=float)
    for k in range(npoints):
        O = StereographicProjection(rotate4d(theta,phi,xi,PQ[k]))
        dists[k] = np.linalg.norm(O)
        pq[k, :] = O
    radii = dists / 20
    r = radii.min()
    rfactor = radii.max() / r
    polyline = polyline_from_points(pq)
    polyline["R"] = radii
    tube = polyline.tube(
        radius=r, scalars="R", radius_factor=rfactor, n_sides=200
    )
    return tube
  
theta = pi/2
phi = 0

def fplot(xi, off_screen, pngname=None):
    vsFacet = ProjectedFacetVertices(theta, phi, xi)
    vsOther = ProjectedOtherVertices(theta, phi, xi)
    pltr = pv.Plotter(window_size=[512,512], off_screen=off_screen)
    pltr.set_background("white")
    for otherEdge in otherEdges:
        edge = Edge(vertices, otherEdge[0], otherEdge[1], theta, phi, xi)
        pltr.add_mesh(edge, color = "silver", pbr=True, metallic=20)
    for facetEdge in facetEdges:
        edge = Edge(vertices, facetEdge[0], facetEdge[1], theta, phi, xi)
        pltr.add_mesh(edge, color = "red", smooth_shading=True, specular=20)
    for v in vsOther:
        sphere = pv.Sphere(np.linalg.norm(v)/10, center=v)
        pltr.add_mesh(sphere, color = "silver", pbr=True, metallic=20)
    for v in vsFacet:
        sphere = pv.Sphere(np.linalg.norm(v)/10, center=v)
        pltr.add_mesh(sphere, color = "red", smooth_shading=True, specular=20)
    pltr.set_focus([0, 0, 0])
    pltr.set_position([-10, 5, 10])
    if pngname is None:
        pltr.show()
    else:
        pltr.show(screenshot=pngname)

xi_ = np.linspace(0, 2*pi, 181)[:180]
for i, xi in enumerate(xi_):
    fplot(xi, True, "zpic_%03d.png" % i)
os.system(
    "magick convert -dispose previous -loop 0 -delay 8 zpic_*.png HexagonalDuoprism.gif"
) 
