import os
from math import pi, cos, sin
import numpy as np
import pyvista as pv


def satellite(t, R, alpha, k):
    return R * np.array([
        cos(alpha) * cos(t) * cos(k*t) - sin(t) * sin(k*t),
        cos(alpha) * sin(t) * cos(k*t) + cos(t) * sin(k*t),
        sin(alpha) * cos(k*t)
    ])


def f(x, y, z):
    return (
        (x*x + y*y + z*z)**2 - 8*x*x - 6*y*y + z*z + 4
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-4):4:250j, (-4):4:250j, (-4):4:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)


values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")
# compute one isosurface
isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()
mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)

t_ = np.linspace(0, 2*pi, 180, endpoint=False)

for i, t in enumerate(t_):
    pltr = pv.Plotter(window_size=[512, 512], off_screen=True)
    pltr.add_background_image("SpaceBackground.png")
    pltr.set_focus((0,0,0))
    pltr.set_position(satellite(t, 12, 3*pi/4, 4))
    pltr.camera.zoom(1)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=10, cmap="plasma", log_scale=False,
        show_scalar_bar=False, flip_scalars=False 
    )
    pngname = "zzpic%03d.png" % i
    pltr.show(screenshot=pngname)


os.system(
    "gifski --frames=zzpic*.png --fps=9 -o BlumCyclide.gif"    
)
