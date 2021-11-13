# -*- coding: utf-8 -*-
import os
from math import sqrt, pi, sin, cos, acos
import pyvista as pv
import numpy as np

# Castellated rhombicosidodecahedral prism - tetrahedra only 

# vertices --------------------------------------------------------------------
phi = (1+sqrt(5))/2
phi2 = phi * phi
phi3 = phi * phi2
vertices = [
    [0,-phi,-phi3,0],
    [0,phi,-phi3,0],
    [0,-phi,phi3,0],
    [0,phi,phi3,0],
    [-phi,-phi3,0,0],
    [phi,-phi3,0,0],
    [-phi,phi3,0,0],
    [phi,phi3,0,0],
    [-phi3,0,-phi,0],
    [phi3,0,-phi,0],
    [-phi3,0,phi,0],
    [phi3,0,phi,0],
    [0,-phi3,-phi2,-1],
    [0,phi3,-phi2,-1],
    [0,-phi3,phi2,-1],
    [0,phi3,phi2,-1],
    [0,-phi3,-phi2,1],
    [0,phi3,-phi2,1],
    [0,-phi3,phi2,1],
    [0,phi3,phi2,1],
    [-phi2,0,-phi3,-1],
    [phi2,0,-phi3,-1],
    [-phi2,0,phi3,-1],
    [phi2,0,phi3,-1],
    [-phi2,0,-phi3,1],
    [phi2,0,-phi3,1],
    [-phi2,0,phi3,1],
    [phi2,0,phi3,1],
    [-phi3,-phi2,0,-1],
    [phi3,-phi2,0,-1],
    [-phi3,phi2,0,-1],
    [phi3,phi2,0,-1],
    [-phi3,-phi2,0,1],
    [phi3,-phi2,0,1],
    [-phi3,phi2,0,1],
    [phi3,phi2,0,1],
    [0,-phi2,-phi-2,-phi],
    [0,phi2,-phi-2,-phi],
    [0,-phi2,phi+2,-phi],
    [0,phi2,phi+2,-phi],
    [0,-phi2,-phi-2,phi],
    [0,phi2,-phi-2,phi],
    [0,-phi2,phi+2,phi],
    [0,phi2,phi+2,phi],
    [-phi2,-phi-2,0,-phi],
    [phi2,-phi-2,0,-phi],
    [-phi2,phi+2,0,-phi],
    [phi2,phi+2,0,-phi],
    [-phi2,-phi-2,0,phi],
    [phi2,-phi-2,0,phi],
    [-phi2,phi+2,0,phi],
    [phi2,phi+2,0,phi],
    [-phi-2,0,-phi2,-phi],
    [phi+2,0,-phi2,-phi],
    [-phi-2,0,phi2,-phi],
    [phi+2,0,phi2,-phi],
    [-phi-2,0,-phi2,phi],
    [phi+2,0,-phi2,phi],
    [-phi-2,0,phi2,phi],
    [phi+2,0,phi2,phi],
    [-phi2,-phi2,-phi2,0],
    [phi2,-phi2,-phi2,0],
    [-phi2,phi2,-phi2,0],
    [phi2,phi2,-phi2,0],
    [-phi2,-phi2,phi2,0],
    [phi2,-phi2,phi2,0],
    [-phi2,phi2,phi2,0],
    [phi2,phi2,phi2,0],
    [-1,-1,-phi3,-phi],
    [1,-1,-phi3,-phi],
    [-1,1,-phi3,-phi],
    [1,1,-phi3,-phi],
    [-1,-1,phi3,-phi],
    [1,-1,phi3,-phi],
    [-1,1,phi3,-phi],
    [1,1,phi3,-phi],
    [-1,-1,-phi3,phi],
    [1,-1,-phi3,phi],
    [-1,1,-phi3,phi],
    [1,1,-phi3,phi],
    [-1,-1,phi3,phi],
    [1,-1,phi3,phi],
    [-1,1,phi3,phi],
    [1,1,phi3,phi],
    [-1,-phi3,-1,-phi],
    [1,-phi3,-1,-phi],
    [-1,phi3,-1,-phi],
    [1,phi3,-1,-phi],
    [-1,-phi3,1,-phi],
    [1,-phi3,1,-phi],
    [-1,phi3,1,-phi],
    [1,phi3,1,-phi],
    [-1,-phi3,-1,phi],
    [1,-phi3,-1,phi],
    [-1,phi3,-1,phi],
    [1,phi3,-1,phi],
    [-1,-phi3,1,phi],
    [1,-phi3,1,phi],
    [-1,phi3,1,phi],
    [1,phi3,1,phi],
    [-phi3,-1,-1,-phi],
    [phi3,-1,-1,-phi],
    [-phi3,1,-1,-phi],
    [phi3,1,-1,-phi],
    [-phi3,-1,1,-phi],
    [phi3,-1,1,-phi],
    [-phi3,1,1,-phi],
    [phi3,1,1,-phi],
    [-phi3,-1,-1,phi],
    [phi3,-1,-1,phi],
    [-phi3,1,-1,phi],
    [phi3,1,-1,phi],
    [-phi3,-1,1,phi],
    [phi3,-1,1,phi],
    [-phi3,1,1,phi],
    [phi3,1,1,phi],
    [-phi,-2*phi,-phi2,-phi],
    [phi,-2*phi,-phi2,-phi],
    [-phi,2*phi,-phi2,-phi],
    [phi,2*phi,-phi2,-phi],
    [-phi,-2*phi,phi2,-phi],
    [phi,-2*phi,phi2,-phi],
    [-phi,2*phi,phi2,-phi],
    [phi,2*phi,phi2,-phi],
    [-phi,-2*phi,-phi2,phi],
    [phi,-2*phi,-phi2,phi],
    [-phi,2*phi,-phi2,phi],
    [phi,2*phi,-phi2,phi],
    [-phi,-2*phi,phi2,phi],
    [phi,-2*phi,phi2,phi],
    [-phi,2*phi,phi2,phi],
    [phi,2*phi,phi2,phi],
    [-phi2,-phi,-2*phi,-phi],
    [phi2,-phi,-2*phi,-phi],
    [-phi2,phi,-2*phi,-phi],
    [phi2,phi,-2*phi,-phi],
    [-phi2,-phi,2*phi,-phi],
    [phi2,-phi,2*phi,-phi],
    [-phi2,phi,2*phi,-phi],
    [phi2,phi,2*phi,-phi],
    [-phi2,-phi,-2*phi,phi],
    [phi2,-phi,-2*phi,phi],
    [-phi2,phi,-2*phi,phi],
    [phi2,phi,-2*phi,phi],
    [-phi2,-phi,2*phi,phi],
    [phi2,-phi,2*phi,phi],
    [-phi2,phi,2*phi,phi],
    [phi2,phi,2*phi,phi],
    [-2*phi,-phi2,-phi,-phi],
    [2*phi,-phi2,-phi,-phi],
    [-2*phi,phi2,-phi,-phi],
    [2*phi,phi2,-phi,-phi],
    [-2*phi,-phi2,phi,-phi],
    [2*phi,-phi2,phi,-phi],
    [-2*phi,phi2,phi,-phi],
    [2*phi,phi2,phi,-phi],
    [-2*phi,-phi2,-phi,phi],
    [2*phi,-phi2,-phi,phi],
    [-2*phi,phi2,-phi,phi],
    [2*phi,phi2,-phi,phi],
    [-2*phi,-phi2,phi,phi],
    [2*phi,-phi2,phi,phi],
    [-2*phi,phi2,phi,phi],
    [2*phi,phi2,phi,phi]
  ]

# rotated and projected vertices ----------------------------------------------
def rotate4d(theta,phi,xi,vec):
    a = cos(xi)
    b = sin(theta)*cos(phi)*sin(xi)
    c = sin(theta)*sin(phi)*sin(xi)
    d = cos(theta)*sin(xi)
    p = vec[0]
    q = vec[1]
    r = vec[2]
    s = vec[3]
    return np.array([ a*p - b*q - c*r - d*s
    , a*q + b*p + c*s - d*r
    , a*r - b*s + c*p + d*q
    , a*s + b*r - c*q + d*p ])


def StereographicProjection(q):
    h = 25.8
    return acos(q[3]/sqrt(h))/pi*q[0:3]/sqrt(h-q[3]*q[3]) 


def Vertices3(theta,phi,xi):
    nvertices = len(vertices)
    out = [None]*nvertices
    for i in range(nvertices):
        out[i] = 3 * StereographicProjection(
                            rotate4d(theta,phi,xi,vertices[i])
                        )
    return out


# macro draw Tetrahedron ------------------------------------------------------
def edge(pltr, i, j):
    line = pv.Line(rvs[i], rvs[j])
    tube = line.tube(radius = 0.03)
    pltr.add_mesh(tube, color = "gold", pbr=True, metallic=20)

def drawTetrahedron(pltr, i, j, k, l):
    points = np.array([rvs[i], rvs[j], rvs[k], rvs[l]])
    faces = [3, 0, 1, 2,
             3, 0, 1, 3,
             3, 0, 2, 3,
             3, 1, 2, 3]
    mesh = pv.PolyData(points, faces)
    pltr.add_mesh(mesh, color = "red", opacity = 0.3)
    edge(pltr, i, j)
    edge(pltr, i,k)
    edge(pltr, i,l)
    edge(pltr, j,k)
    edge(pltr, j,l)
    edge(pltr, k,l)
    for index in [i, j, k, l]:
        sphere = pv.Sphere(0.045, center = rvs[index])
        pltr.add_mesh(sphere, smooth_shading=True, color = "gold")

# draw ------------------------------------------------------------------------
xi_ = np.linspace(0, 2*pi, 181)[:180]
for i, xi in enumerate(xi_):
    rvs = Vertices3(0, pi/2, xi)
    pltr = pv.Plotter(window_size=[512,512], off_screen=True)
    pltr.add_background_image("SpaceBackground.png")
    drawTetrahedron(pltr, 112, 114, 10, 58)
    drawTetrahedron(pltr, 142, 158, 62, 126)
    drawTetrahedron(pltr, 146, 130, 66, 162)
    drawTetrahedron(pltr, 81, 80, 42, 2)
    drawTetrahedron(pltr, 111, 109, 57, 9)
    drawTetrahedron(pltr, 76, 77, 0, 40)
    drawTetrahedron(pltr, 79, 41, 78, 1)
    drawTetrahedron(pltr, 48, 92, 96, 4)
    drawTetrahedron(pltr, 49, 97, 93, 5)
    drawTetrahedron(pltr, 140, 124, 156, 60)
    drawTetrahedron(pltr, 98, 94, 50, 6)
    drawTetrahedron(pltr, 56, 108, 110, 8)
    drawTetrahedron(pltr, 160, 128, 144, 64)
    drawTetrahedron(pltr, 85, 45, 89, 5)
    drawTetrahedron(pltr, 129, 161, 145, 65)
    drawTetrahedron(pltr, 115, 59, 113, 11)
    drawTetrahedron(pltr, 159, 143, 63, 127)
    drawTetrahedron(pltr, 43, 83, 82, 3)
    drawTetrahedron(pltr, 53, 101, 103, 9)
    drawTetrahedron(pltr, 61, 157, 141, 125)
    drawTetrahedron(pltr, 102, 52, 100, 8)
    drawTetrahedron(pltr, 149, 61, 133, 117)
    drawTetrahedron(pltr, 136, 120, 152, 64)
    drawTetrahedron(pltr, 7, 51, 95, 99)
    drawTetrahedron(pltr, 116, 148, 132, 60)
    drawTetrahedron(pltr, 44, 88, 84, 4)
    drawTetrahedron(pltr, 69, 68, 36, 0)
    drawTetrahedron(pltr, 135, 151, 119, 63)
    drawTetrahedron(pltr, 154, 138, 122, 66)
    drawTetrahedron(pltr, 106, 54, 104, 10)
    drawTetrahedron(pltr, 107, 105, 55, 11)
    drawTetrahedron(pltr, 67, 163, 147, 131)
    drawTetrahedron(pltr, 38, 72, 73, 2)
    drawTetrahedron(pltr, 134, 150, 118, 62)
    drawTetrahedron(pltr, 46, 86, 90, 6)
    drawTetrahedron(pltr, 75, 39, 74, 3)
    drawTetrahedron(pltr, 153, 121, 137, 65)
    drawTetrahedron(pltr, 37, 71, 70, 1)
    drawTetrahedron(pltr, 123, 155, 67, 139)
    drawTetrahedron(pltr, 47, 91, 7, 87)
    pltr.set_focus([0, 0, 0])
    pltr.set_position([-10, 0, 0])
    pltr.show(screenshot="zpic_CRP%03d.png" % i)

os.system(
    "magick convert -dispose previous -loop 0 -delay 8 zpic_CRP*.png CRP.gif"
) 