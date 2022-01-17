# -*- coding: utf-8 -*-
# this script requires the Weierstrass functions P and Zeta, available
#  here: https://gist.github.com/stla/d771e0a8c351d16d186c79bc838b6c48
import pyvista as pv
import numpy as np
exec(open("../WeierstrassP02.py").read())

e1 = p_weierstrass_from_w1_w2(1/2, 1j/2)(1/2).real
c = 4*e1**2
zeta = np.frompyfunc(zeta_weierstrass(c, 0), 1, 1)
weier = np.frompyfunc(p_weierstrass_from_g2_g3(c, 0), 1, 1)
def fx(u, v):
    z = u + 1j*v
    zout = pi*(u + pi/4/e1) - zeta(z) + pi/2/e1*(zeta(z-1/2) - zeta(z-1j/2))
    return zout.real
fxufunc = np.frompyfunc(fx, 2, 1)
logufunc = np.frompyfunc(log, 1, 1)
def f(u, v):
    zx = fxufunc(u, v)
    zy = fxufunc(v, u)
    p = weier(u + 1j*v)
    return (
        zx/2,
        zy/2,
        sqrt(pi/2) * logufunc(abs((p-e1)/(p+e1)))/2
    )

tofloat = np.vectorize(float, otypes=["float"])
U, V = np.meshgrid(np.linspace(0.05, 0.95, 100), np.linspace(0.05, 0.95, 100))
x, y, z = f(U, V)
grid = pv.StructuredGrid(tofloat(x), tofloat(y), tofloat(z))
mesh = grid.extract_geometry()
mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)

#points = mesh.points
surf = pv.wrap(mesh.points).reconstruct_surface()

pltr = pv.Plotter(window_size = [512, 512])
pltr.add_mesh(
    surf, smooth_shading=True, cmap="viridis", specular=15, 
    show_scalar_bar=False    
)
pltr.set_focus([0, 0, 0])
pltr.set_position([18, 0, 18])
pltr.show()