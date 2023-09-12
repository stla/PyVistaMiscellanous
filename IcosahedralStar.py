# -*- coding: utf-8 -*-
import os
from math import pi, cos, sin
import pyvista as pv
import numpy as np


def satellite(t, R, alpha=3 * pi / 4, k=4):
    return R * np.array(
        [
            cos(alpha) * cos(t) * cos(k * t) - sin(t) * sin(k * t),
            cos(alpha) * sin(t) * cos(k * t) + cos(t) * sin(k * t),
            sin(alpha) * cos(k * t),
        ]
    )


def icosahedral_star(x, y, z, a):
    u = x**2 + y**2 + z**2
    v = (
        -z
        * (2 * x + z)
        * (
            x**4
            - x**2 * z**2
            + z**4
            + 2 * (x**3 * z - x * z**3)
            + 5 * (y**4 - y**2 * z**2)
            + 10 * (x * y**2 * z - x**2 * y**2)
        )
    )
    return (1 - u) ** 3 + a * u**3 + a * v


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-0.75):0.75:250j, (-0.75):0.75:250j, (-0.75):0.75:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
a = -1000
values = icosahedral_star(X, Y, Z, a)
grid.point_data["values"] = values.ravel(order="F")

isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()
dists = np.linalg.norm(mesh.points, axis=1)
dists = (dists - dists.min()) / (dists.max() - dists.min())

pltr = pv.Plotter(window_size=[512, 512], off_screen=False)
pltr.background_color = "#363940"
pltr.set_focus(mesh.center)
pltr.set_position((7, 4, 2))
pltr.camera.zoom(2.5)
pltr.add_mesh(
    mesh,
    scalars = dists,
    smooth_shading=True,
    specular=0.2,
    cmap="turbo",
    log_scale=False,
    show_scalar_bar=False,
    flip_scalars=False,
)
pltr.show()

# animation
nframes = 180
t_ = np.linspace(0, 2 * pi, nframes + 1)[:nframes]
for i, t in enumerate(t_):
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size=[512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position(satellite(t, 8))
    pltr.camera.zoom(2.4)
    pltr.add_mesh(
        mesh,
        smooth_shading=True,
        specular=0.2,
        cmap="turbo",
        log_scale=False,
        show_scalar_bar=False,
        flip_scalars=False,
    )
    pltr.show(screenshot=pngname)

os.system("gifski --frames=zzpic*.png --fps=10 -o IcosahedralStar.gif")
