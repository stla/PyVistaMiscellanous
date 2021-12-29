from math import pi, sqrt
import numpy as np
import pyvista as pv

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
#mesh.plot(color = "red", smooth_shading = True)

mesh.scale((0.4,0.4,0.4))
# mesh.plot()

# vertices ####
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
  

plotter = pv.Plotter(window_size=[512,512])

for i in range(20):
    v = np.array(vertices[i])
    M = Reorient_Trans(np.array([0,0,1]), v)
    m = mesh.copy()
    m.transform(M)
    m.translate((v[0], v[1], v[2]))
    m.rotate_vector(v, angle=40)
    plotter.add_mesh(m, smooth_shading=True, color="red")
    cone = pv.Cone(radius=0, direction=v, angle=2, height=2)
    plotter.add_mesh(cone, smooth_shading=True, color="chocolate")

plotter.show()
    