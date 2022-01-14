from math import sqrt
import numpy as np
import pyvista as pv

def f(x, y, z):
    return (
        64
        * (x - 1)
        * (
            x ** 4
            - 4 * x ** 3
            - 10 * x ** 2 * y ** 2
            - 4 * x ** 2
            + 16 * x
            - 20 * x * y ** 2
            + 5 * y ** 4
            + 16
            - 20 * y ** 2
        )
        - 5
        * sqrt(5 - sqrt(5))
        * (2 * z - sqrt(5 - sqrt(5)))
        * (4 * (x ** 2 + y ** 2 - z ** 2) + (1 + 3 * sqrt(5))) ** 2
    )

# generate data grid for computing the values
X, Y, Z = np.mgrid[(-5):5:250j, (-5):5:250j, (-4):4:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")
# compute the isosurface f(x, y, z) = 0
isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()

# surface clipped to the box:
mesh.plot(smooth_shading=True, color="yellowgreen", specular=15)

# surface clipped to the ball of radius 4.8, with the help of `remove_points`:
lengths = np.linalg.norm(mesh.points, axis=1)
toremove = lengths >= 4.8
masked_mesh, idx = mesh.remove_points(toremove)
masked_mesh.plot(smooth_shading=True, color="orange", specular=15)

# surface clipped to the ball of radius 4.8, with the help of `clip_scalar`:
mesh["dist"] = lengths
clipped_mesh = mesh.clip_scalar("dist", value=4.8)
clipped_mesh.plot(
    smooth_shading=True, cmap="inferno", window_size=[512, 512],
    show_scalar_bar=False, specular=15, show_axes=False, zoom=1.2,
    background="#363940", screenshot="Togliatti_python.png"
)