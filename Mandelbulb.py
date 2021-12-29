# -*- coding: utf-8 -*-
from math import sin, cos, sqrt, atan2
import pyvista as pv
import numpy as np

def mandelbulb0(x0, y0, z0):
    x = x0
    y = y0
    z = z0
    r2 = theta = phi = r8 = None
    for i in range(24):
        r2 = x*x + y*y + z*z
        if r2 > 4:
            break
        theta = 8 * atan2(sqrt(x*x + y*y), z)
        phi = 8 * atan2(y, x)
        r8 = r2*r2*r2*r2
        x = r8*cos(phi)*sin(theta) + x0
        y = r8*sin(phi)*sin(theta) + y0
        z = r8*cos(theta) + z0
    return sqrt(r2)


# generate data grid for computing the values
n = 200
X, Y, Z = np.mgrid[-1.2:1.2:(n*1j), -1.2:1.2:(n*1j), -1.2:1.2:(n*1j)]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
#grid.points = np.transpose(sph2cart(grid.points))


values = np.empty((n,n,n), dtype=float)
for i in range(n):
    for j in range(n):
        for k in range(n):
            x = X[i, j, k]
            y = Y[i, j, k]
            z = Z[i, j, k]
            values[i,j,k] = mandelbulb0(x, y, z)

grid.point_data["values"] = values.ravel(order="F") 
# compute one isosurface
mesh = grid.contour(isosurfaces=[2])
lengths = np.linalg.norm(mesh.points, axis=1)
toremove = lengths > 2
mesh2, idx = mesh.remove_points(toremove)
mesh2.plot(smooth_shading=True, color="hotpink")


# NumericVector mandelbulb(double m, double M, unsigned n) {
#   NumericVector out(n*n*n);
#   double h = (M-m)/(n-1);
#   NumericVector xyz(3);
#   double x,y,z;
#   unsigned l = 0;
#   for(unsigned i=0; i<n; i++){
#     xyz[0] = m + i*h;
#     for(unsigned j=0; j<n; j++){
#       xyz[1] = m + j*h;
#       for(unsigned k=0; k<n; k++){
#         xyz[2] = m + k*h;
#         if(x*x+y*y+z*z > 4){
#           out[l] = R_PosInf;
#         }else{
#           out[l] = mandelbulb0(xyz);
#         }
#         l += 1;
#       }
#     }
#   }
#   return out;
# }