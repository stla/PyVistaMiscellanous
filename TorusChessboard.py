from math import cos, sin, pi, sqrt
import numpy as np
import pyvista as pv
import bisect

def torusMesh(R, r, S=64, s=32):
  vertices = np.empty((0,3))
  faces    = np.empty((0), dtype=int)
  distances = []
  breakpoints_v = np.linspace(0, 36, 7)
  breakpoints_u = np.linspace(0, 64, 9)
  for j in range(s+1):
    v = j/s * 2*pi
    cos_v = np.cos(v)
    sin_v = np.sin(v)
    for i in range(S+1):
      u = i/S * 2*pi
      vertex = np.array(
        [
          (R + r*cos_v) * np.cos(u),
          (R + r*cos_v) * np.sin(u),
          r * sin_v          
        ]
      )
      vertices = np.vstack((vertices, vertex))
      if i < S and j < s :
        kj = (bisect.bisect_left(breakpoints_v, j) -1) % 2
        ki = (bisect.bisect_right(breakpoints_u, i) -1) % 2 
        bp_v = pi*(np.linspace(0, 17, 2)) % (2*pi)
        bp_u = pi*(np.linspace(0, 33, 2)) % (2*pi)
        k = (ki + kj) % 2
        k = 0 if j <= 16 else 1
        if i <= 32:
            k = 1-k
        distances.append(k)
  for j in range(s):
    for i in range(S):
      a = (S + 1) * (j+1) + i  
      b = (S + 1) * j + i 
      c = (S + 1) * j + i + 1
      d = (S + 1) * (j+1) + i + 1
      faces = np.concatenate(
        (faces, np.array([4, a, b, c, d])), axis = 0
      )
  mesh = pv.PolyData(vertices, faces).extract_geometry().clean(tolerance=1e-6)
  mesh.point_data["distance"] = np.asarray(distances)
  return mesh
  
surf = torusMesh(1, 0.25, 64, 32)
#print(surf)
surf.plot(smooth_shading=True, cmap=["#440154", "#FDE725"])

