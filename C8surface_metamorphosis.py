# -*- coding: utf-8 -*-
import os
import numpy as np
import pyvista as pv


def f(x, y, z):
  return (
      64*x**8 - 128*x**6 + 80*x**4 - 16*x**2 + 2 + 64*y**8 - 128*y**6 + 80*y**4 
      - 16*y**2 + 64*z**8 - 128*z**6 + 80*z**4 - 16*z**2
  )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-1.1):1.1:300j, (-1.1):1.1:300j, (-1.1):1.1:300j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")

v_ = np.linspace(-0.5, 0.5, 70)
for i, v in enumerate(v_):
    # compute the isosurface f(x, y, z) = v
    isosurf = grid.contour(isosurfaces = [v])
    mesh = isosurf.extract_geometry()
    mesh["dist"] = np.linalg.norm(mesh.points, axis=1)
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position((7, 4, 2))
    pltr.camera.zoom(1.3)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=15, cmap="twilight", 
        log_scale=False, show_scalar_bar=False, flip_scalars=False 
    )
    pltr.show(screenshot=pngname)
    
os.system(
  "magick convert -dispose previous -delay 1x9 -duplicate 1,-2-1 zzpic*.png C8surface_metamorphosis.gif"  
)
    