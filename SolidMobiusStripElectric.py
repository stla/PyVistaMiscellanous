from math import sqrt
import numpy as np
import pyvista as pv


def f(x, y, z, a=0.4, b=0.1):
    return ((
        (x * x + y * y + 1) * (a * x * x + b * y * y)
        + z * z * (b * x * x + a * y * y)
        - 2 * (a - b) * x * y * z
        - a * b * (x * x + y * y)
    ) ** 2 
    - 4 * (x * x + y * y) * (a * x * x + b * y * y - x * y * z * (a - b)) ** 2)


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-2):2:250j, (-2):2:250j, (-1):1:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")
# compute the isosurface f(x, y, z) = 0
isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()


tex = pv.read_texture("ElectricityTexture.jpg")

pltr = pv.Plotter(window_size=[512, 512])
pltr.add_background_image("SpaceBackground.png")
mesh.texture_map_to_sphere(inplace=True)
pltr.add_mesh(mesh, texture=tex, specular=25)
pltr.show()
