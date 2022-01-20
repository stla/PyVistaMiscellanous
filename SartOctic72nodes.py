import numpy as np
import pyvista as pv


def f(x, y, z):
    return (
        3584 * z ** 4
        + 256 * z ** 8
        + 1792 * z ** 4 * x ** 4
        + 10752 * z ** 2 * x ** 4
        + 1792 * x ** 4
        + 256 * x ** 8
        + 256
        + 1792 * z ** 4 * y ** 4
        + 10752 * z ** 2 * y ** 4
        + 1792 * y ** 4
        + 256 * y ** 8
        + 10752 * z ** 4 * x ** 2 * y ** 2
        - 21504 * z ** 2 * x ** 2 * y ** 2
        + 10752 * x ** 2 * y ** 2
        + 3584 * x ** 4 * y ** 4
        + 192
        * (
            -1
            - 12 * x ** 4 * y ** 2 * z ** 2
            - 24 * x ** 2 * y ** 2 * z ** 2
            - 12 * x ** 2 * y ** 2
            - 12 * x ** 2 * z ** 2
            - 12 * y ** 2 * z ** 2
            - 12 * x ** 4 * y ** 2
            - 12 * x ** 4 * z ** 2
            - 12 * x ** 2 * y ** 4
            - 12 * x ** 2 * z ** 4
            - 12 * y ** 4 * z ** 2
            - 12 * y ** 2 * z ** 4
            - 4 * x ** 6 * y ** 2
            - 4 * x ** 6 * z ** 2
            - 6 * x ** 4 * y ** 4
            - 6 * x ** 4 * z ** 4
            - 4 * x ** 2 * y ** 6
            - 4 * x ** 2 * z ** 6
            - 4 * y ** 6 * z ** 2
            - 6 * y ** 4 * z ** 4
            - 4 * y ** 2 * z ** 6
            - 12 * x ** 2 * y ** 4 * z ** 2
            - 12 * x ** 2 * y ** 2 * z ** 4
            - 4 * x ** 2
            - 4 * y ** 2
            - 4 * z ** 2
            - 6 * x ** 4
            - 6 * y ** 4
            - 6 * z ** 4
            - 4 * x ** 6
            - 4 * y ** 6
            - 4 * z ** 6
            - x ** 8
            - y ** 8
            - z ** 8
        )
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-5):5:250j, (-5):5:250j, (-5):5:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")
# compute the isosurface f(x, y, z) = 0
isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()


# surface clipped to the ball of radius 5, with the help of `clip_scalar`:
mesh["dist"] = np.linalg.norm(mesh.points, axis=1)
clipped_mesh = mesh.clip_scalar("dist", value=5)
clipped_mesh.plot(
    smooth_shading=True,
    cmap="nipy_spectral",
    window_size=[512, 512],
    show_scalar_bar=False,
    specular=15,
    show_axes=False,
    zoom=1.3,
    background="#363940",
    screenshot="SartOctic72nodes.png",
)
