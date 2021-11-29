# -*- coding: utf-8 -*-
import numpy as np
from cmath import sqrt
import pyvista as pv

ta = complex(1.958591030,-0.011278560)
tb = complex(2, 0)
tab = (ta*tb + sqrt((ta**2 * tb**2) - 4 * (ta**2 + tb**2))) / 2
z0 = (tb * (tab-2)) / (tb * tab - 2*ta + 2*tab*1j)
b = np.array([[(tb-2j)/2, tb/2], [tb/2, (tb+2j)/2]])
B = np.linalg.inv(b)
a = np.matmul(np.array([[tab, (tab-2)/z0], [(tab+2)*z0, tab]]), B)
A = np.linalg.inv(a)
def Fix(A):
    return (A[0,0]-A[1,1]-sqrt(4*A[0,1]*A[1,0] + (A[0,0]-A[1,1])**2)) / 2 / A[1,0]
def ToMatrix(z, r):
    return 1j/r * np.array([[z, r**2-abs(z)**2], [1, -z.conjugate()]])
def MotherCircle(M1, M2, M3):
    z1 = Fix(M1)
    x1 = z1.real
    y1 = z1.imag
    z2 = Fix(M2)
    x2 = z2.real
    y2 = z2.imag
    z3 = Fix(M3)
    x3 = z3.real
    y3 = z3.imag
    z0 = complex(x3**2 * (y1-y2) + (x1**2 + (y1-y2)*(y1-y3)) * (y2-y3) + x2**2 * (y3-y1), 
                 -x2**2 * x3 + x1**2 * (x3-x2) + x3 * (y1-y2) * (y1+y2) + x1 * (x2**2 - x3**2 + y2**2 - y3**2) + x2 * (x3**2 - y1**2 + y3**2)) / (2*(x3*(y1-y2)+x1*(y2-y3)+x2*(y3-y1)))
    print("z0", z0)
    print("z1", z1)
    return ToMatrix(z0, abs(z1-z0))
C1 = MotherCircle(b, a @ b @ A, a @ b @ A @ B)
C2 = MotherCircle(b @ np.linalg.matrix_power(a, 15), a @ b @ np.linalg.matrix_power(a, 14), a @ b @ A @ B)
def Reflect(C, M):
    return M @ C @ np.linalg.inv(np.conjugate(M))
def zcen(A):
    q = A[0,0] / A[1,0]
    return (q.real, q.imag, 0)
def rad(A):
    return (1j / A[1,0]).real

def spiral(pltr, C0, M, n, colorize, reverse):
    C=C0.copy()
    i=0
    while i<n-1:
        Z=zcen(C)
        print("Z", Z)
        r=rad(C)
        print("rad", r)
        if r < 1000:
            if colorize: #local hue=mod((18+(reverse?1:-1)*i)/15,1);
                sphere = pv.Sphere(r, Z)
                pltr.add_mesh(sphere, color="red")
            else:
                sphere = pv.Sphere(r, Z)
                pltr.add_mesh(sphere, opacity=0.1)
        C = Reflect(C, M)
        i = i+1

pltr = pv.Plotter()
#spiral(pltr,C1,a,83,True,False)
#spiral(pltr,C1,A,83,True, True)
spiral(pltr, C2,a,91,False, False)
#spiral(pltr, C2,A,76,False, True)
pltr.show()
