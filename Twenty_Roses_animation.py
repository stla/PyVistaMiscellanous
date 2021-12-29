import os
from math import pi, sqrt, asin, atan2, cos, sin
import numpy as np
import pyvista as pv
import quaternion


# stuff for camera motion #####################################################

def quaternion2hab(q):
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


nframes = 180
quats = [None]*nframes
t_ = np.linspace(0, 2*pi, nframes+1)[:nframes]
satellite0 = satellite(0, 5, 3*pi/4, 4)
A = satellite0.copy()
q0 = quaternion.one
quats[0] = q0
for i in range(nframes-1):
    B = satellite(t_[i+1], 5, 3*pi/4, 4)
    q1 = get_quaternion(A/5, B/5) * q0
    quats[i+1] = q1
    A = B
    q0 = q1


# stuff for the rose ##########################################################

def mod2pi(x):
    return np.mod(x, 2 * pi) 
    
def F(u, v):
    theta = 2*pi + 16*pi*v
    phi = pi/2 * np.exp(-theta / 8/pi)
    beta = 1 - 0.5 * (
        ((5 / 4) * (1 - (mod2pi(3.6 * theta)) / pi) ** 2 - 1 / 4) ** 2    
    )
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    y = 1.95653 * u ** 2 * (1.27689 * u - 1) ** 2 * sin_phi
    r = beta * (u * sin_phi + y * cos_phi)
    return np.array([
        r * np.cos(theta),
        r * np.sin(theta),
        beta * (u * cos_phi - y * sin_phi)
    ])
                
u_ = np.linspace(0, 1, 200) 
v_ = np.linspace(0, 1, 200)
u, v = np.meshgrid(u_, v_)
x, y, z = F(u,v) 
mesh = pv.StructuredGrid(x, y, z)
mesh.scale((0.43, 0.43, 0.43))

# twenty vertices on the unit ball ####
phi = (1 + sqrt(5))/2
a = 1/sqrt(3)
b = a/phi 
c = a*phi
vertices = [
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
  ]

def Reorient_Trans(Axis1, Axis2):
  vX1 = Axis1 / np.linalg.norm(Axis1)
  vX2 = Axis2 / np.linalg.norm(Axis2)
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
          (
              np.vstack((M, np.zeros((1,3)))), 
              np.transpose(np.array([0,0,0,1]))
          )
      )
  )
  

for j, q in enumerate(quats):
    pltr = pv.Plotter(window_size=[512,512], off_screen=True)
    pltr.set_background("#363940")
    pltr.set_focus((0,0,0))
    pltr.set_position(satellite0)
    he, at, ba = quaternion2hab(q)
    pltr.camera.roll = ba
    pltr.camera.azimuth = at
    pltr.camera.elevation = he
    pltr.camera.zoom(0.85)
    for i in range(20):
        v = np.array(vertices[i])
        M = Reorient_Trans(np.array([0,0,1]), v)
        m = mesh.copy()
        m.transform(M)
        m.translate((v[0], v[1], v[2]))
        pltr.add_mesh(m, smooth_shading=True, color="red")
        cone = pv.Cone(radius=0, direction=v, angle=1.5, height=2)
        pltr.add_mesh(cone, smooth_shading=True, color="forestgreen") 
    pngname = "zzpic%03d.png" % j
    pltr.show(screenshot=pngname)

os.system(
    "magick convert -dispose previous -loop 0 -delay 12 zzpic*.png TwentyRoses.gif"    
)
    