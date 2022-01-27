# -*- coding: utf-8 -*-
import os
from math import pi, atan2, asin, sqrt, cos, sin
from matplotlib import cm
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


A = 0.44
n = 3
def Gamma(t):
	alpha = pi/2 - (pi/2-A)*np.cos(n*t)
	beta = t + A*np.sin(2*n*t)
	return np.array([
      np.sin(alpha) * np.cos(beta),
      np.sin(alpha) * np.sin(beta),
      np.cos(alpha)
	])

def HopfInverse(p, phi):
	return np.array([
      (1+p[2])*np.cos(phi),
      p[0]*np.sin(phi) - p[1]*np.cos(phi), 
      p[0]*np.cos(phi) + p[1]*np.sin(phi),
      (1+p[2])*np.sin(phi)
	]) / np.sqrt(2*(1+p[2]))

def Stereo(q):
	return 2*q[0:3] / (1-q[3])

def F(t, phi):
	return Stereo(HopfInverse(Gamma(t), phi))

def HTmesh(nu=400, nv=200):
    angle_u = np.linspace(-np.pi, np.pi, nu) 
    angle_v = np.linspace(0, np.pi, nv)
    u, v = np.meshgrid(angle_u, angle_v)
    x, y, z = F(u, v)
    grid = pv.StructuredGrid(x, y, z)
    return grid.extract_geometry().clean(tolerance=1e-6)

mesh = HTmesh()
mesh.scale((0.02,0.02,0.02))

# twenty vertices ####
phi = (1+sqrt(5))/2
a = 1/sqrt(3)
b = a/phi 
c = a*phi
vertices = np.array([
    [ a,  a,  a], 
    [ a,  a, -a],
    [ a, -a,  a],
    [-a, -a,  a],
    [-a,  a, -a],
    [-a,  a,  a],
    [ 0,  b, -c], 
    [ 0, -b, -c], 
    [ 0, -b,  c],
    [ c,  0, -b],
    [-c,  0, -b],
    [-c,  0,  b],
    [ b,  c,  0],
    [ b, -c,  0],
    [-b, -c,  0],
    [-b,  c,  0],
    [ 0,  b,  c],
    [ a, -a, -a],
    [ c,  0,  b],
    [-a, -a, -a]
  ])

def Reorient_Trans(Axis1, Axis2):
  vX1 = Axis1 #/ np.linalg.norm(Axis1)
  vX2 = Axis2 #/ np.linalg.norm(Axis2)
  Y = np.cross(vX1, vX2)
  vY = Y / np.linalg.norm(Y)
  Z1 = np.cross(vX1, vY)
  vZ1 = Z1 / np.linalg.norm(Z1)
  Z2 = np.cross(vX2, vY)
  vZ2 = Z2 / np.linalg.norm(Z2)
  M1 = np.transpose(np.array([vX1, vY, vZ1]))
  M2 = np.array([vX2, vY, vZ2])
  M = np.matmul(M1, M2)
  return np.transpose(
      np.column_stack(
          (np.vstack((M, np.zeros((1,3)))), np.transpose(np.array([0,0,0,1])))
      )
  )
  
nframes = 180
angles = np.linspace(0, 1440, nframes+1)[:nframes]
pos0, quats = satellite_motion(nframes, 5)

colors = cm.viridis(np.linspace(0, 1, len(vertices)))

for i, q in enumerate(quats):
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size=[512,512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus([0, 0, 0])
    pltr.set_position(pos0)
    h, a, b = quaternion2hab(q)
    pltr.camera.roll = b
    pltr.camera.azimuth = a
    pltr.camera.elevation = h
    pltr.camera.zoom(1)
    for j, v in enumerate(vertices):
        M = Reorient_Trans(np.array([1,0,0]), v)
        m = mesh.copy()
        m.transform(M)
        m.translate((v[0], v[1], v[2]), inplace=True)
        m.rotate_vector(v, angle=angles[i], inplace=True)
        pltr.add_mesh(m, smooth_shading=True, color=colors[j][:3], specular=15)
        cone = pv.Cone(radius=0, direction=v, angle=1.5, height=2)
        pltr.add_mesh(cone, smooth_shading=True, color="chocolate")    
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=10 -o twentyHopfTori.gif"    
)
