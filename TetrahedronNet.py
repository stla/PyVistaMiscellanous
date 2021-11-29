# -*- coding: utf-8 -*-
import os
from math import sqrt, cos, sin, acos, pi
import numpy as np
import pyvista as pv

# Affine rotation ####
#' Matrix of the affine rotation around an axis 
#' @param theta angle of rotation in radians
#' @param P1,P2 the two points defining the axis of rotation 
def AffineRotationMatrix(theta, P1, P2):
    T = np.vstack(
        (
            np.hstack((np.eye(3), -P1.reshape(3,1))),
            np.array([0, 0, 0, 1])
        )
    )
    invT = np.vstack(
        (
            np.hstack((np.eye(3), P1.reshape(3,1))),
            np.array([0, 0, 0, 1])
        )
    )
    a, b, c = (P2 - P1) / np.linalg.norm(P2 - P1)
    d = sqrt(b*b + c*c)
    if d > 0:
        Rx = np.array([
            [1, 0, 0, 0],
            [0, c/d, -b/d, 0],
            [0, b/d, c/d, 0],
            [0, 0, 0, 1]
        ])
        invRx = np.array([
            [1, 0, 0, 0],
            [0, c/d, b/d, 0],
            [0, -b/d, c/d, 0],
            [0, 0, 0, 1]
        ])
    else:
        Rx = invRx = np.eye(4)
    Ry = np.array([
        [d, 0, -a, 0],
        [0, 1, 0, 0],
        [a, 0, d, 0],
        [0, 0, 0, 1]
    ])
    invRy = np.array([
        [d, 0, a, 0],
        [0, 1, 0, 0],
        [-a, 0, d, 0],
        [0, 0, 0, 1]
    ])
    Rz = np.array([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta), cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    #return T @ Rx @ Ry @ Rz @ invRy @ invRx @ invT
    return invT @ invRx @ invRy @ Rz @ Ry @ Rx @ T

# dihedral angles ####
# https://math.stackexchange.com/questions/49330/the-dihedral-angles-of-a-tetrahedron-in-terms-of-its-edge-lengths
def squaredTriangleArea(A, B, C):
    a = np.linalg.norm(B-C)
    b = np.linalg.norm(C-A)
    c = np.linalg.norm(A-B)
    s = (a + b + c) / 2
    return s*(s-a)*(s-b)*(s-c)

def squaredNorm(A):
    return np.vdot(A, A)

def dihedralAngles(O, A, B, C):
    a2 = squaredNorm(O-A)
    b2 = squaredNorm(O-B)
    c2 = squaredNorm(O-C)
    d2 = squaredNorm(B-C)
    e2 = squaredNorm(C-A)
    f2 = squaredNorm(A-B)
    H2 = (4*a2*d2 - ((b2+e2)-(c2+f2))**2) / 16
    J2 = (4*b2*e2 - ((c2+f2)-(a2+d2))**2) / 16
    K2 = (4*c2*f2 - ((a2+d2)-(b2+e2))**2) / 16
    W2 = squaredTriangleArea(A, B, C)
    X2 = squaredTriangleArea(O, B, C)
    Y2 = squaredTriangleArea(O, A, C)
    Z2 = squaredTriangleArea(O, A, B)
    cosBC = (W2+X2-H2) / 2 / sqrt(W2*X2)
    cosCA = (W2+Y2-J2) / 2 / sqrt(W2*Y2)
    cosAB = (W2+Z2-K2) / 2 / sqrt(W2*Z2)
    return {
        "AB": acos(cosAB), 
        "BC": acos(cosBC), 
        "CA": acos(cosCA)
    }


O = np.array([0.0,0.0,0.0])
A = np.array([2.0,0.0,0.0])
B = np.array([1.0,2.0,0.0])
C = np.array([1.0,1.0,1.0])

ABC = pv.PolyData(np.array([A, B, C]), [3, 0, 1, 2])
OAB = pv.PolyData(np.array([O, A, B]), [3, 0, 1, 2])
OBC = pv.PolyData(np.array([O, B, C]), [3, 0, 1, 2])
OCA = pv.PolyData(np.array([O, C, A]), [3, 0, 1, 2])

angles = dihedralAngles(O, A, B, C)

n = 100
alpha_ = np.linspace(0, angles["AB"]-pi, n)
beta_ = np.linspace(0, angles["BC"]-pi, n)
gamma_ = np.linspace(0, angles["CA"]-pi, n)

for k in range(n):
    transfo1 = AffineRotationMatrix(alpha_[k], A, B)
    transfo2 = AffineRotationMatrix(beta_[k], B, C)
    transfo3 = AffineRotationMatrix(gamma_[k], C, A)
    OABcopy = OAB.copy()
    OBCcopy = OBC.copy()
    OCAcopy = OCA.copy()
    pltr = pv.Plotter(window_size=[512,512], off_screen=True)
    pltr.set_background("#363940")
    pltr.set_focus((1, 3/4, 1/4))
    pltr.set_position((-7, -3, 2))
    pltr.add_mesh(ABC, color="white")
    OABcopy.transform(transfo1)
    pltr.add_mesh(OABcopy, color="red")
    OBCcopy.transform(transfo2)
    pltr.add_mesh(OBCcopy, color="green")
    OCAcopy.transform(transfo3)
    pltr.add_mesh(OCAcopy, color="blue")
    png = "zpic_%03d" % k
    pltr.show(screenshot=png)

os.system(
    "magick convert -dispose previous -loop 0 -delay 8 -duplicate 1,-2-1 zpic_*.png TetrahedronNet.gif"    
)
