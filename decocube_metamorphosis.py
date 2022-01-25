# -*- coding: utf-8 -*-
import os
import numpy as np
import pyvista as pv


def f(x, y, z, a = 0.95):
  x2 = x*x
  y2 = y*y
  z2 = z*z
  a2 = a*a
  x2y2a2 = x2+y2-a2
  y2z2a2 = y2+z2-a2
  z2x2a2 = z2+x2-a2
  return (
      (x2y2a2*x2y2a2 + (z2-1)*(z2-1)) 
      * (y2z2a2*y2z2a2 + (x2-1)*(x2-1)) 
      * (z2x2a2*z2x2a2 + (y2-1)*(y2-1))
  )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-1.3):1.3:300j, (-1.3):1.3:300j, (-1.3):1.3:300j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")

v_ = np.linspace(0.005, 0.02, 60)
for i, v in enumerate(v_):
    # compute the isosurface f(x, y, z) = v
    isosurf = grid.contour(isosurfaces = [v])
    mesh = isosurf.extract_geometry()
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position((7, 4, 2))
    pltr.camera.zoom(1.3)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=15, cmap="viridis", 
        log_scale=False, show_scalar_bar=False, flip_scalars=False 
    )
    pltr.show(screenshot=pngname)
    
os.system(
  "magick convert -dispose previous -delay 1x8 -duplicate 1,-2-1 zzpic*.png decocube_metamorphosis.gif"  
)
    