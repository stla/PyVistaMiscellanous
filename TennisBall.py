import pyvista as pv
import numpy as np


# Enneper surface: f=0
def f(x, y, z):
    return (
        64 * z**9
        - 128 * z**7
        + 64 * z**5
        - 702 * x**2 * y**2 * z**3
        - 18 * x**2 * y**2 * z
        + 144 * (y**2 * z**6 - x**2 * z**6)
        + 162 * (y**4 * z**2 - x**4 * z**2)
        + 27 * (y**6 - x**6)
        + 9 * (x**4 * z + y**4 * z)
        + 48 * (x**2 * z**3 + y**2 * z**3)
        - 432 * (x**2 * z**5 + y**2 * z**5)
        + 81 * (x**4 * y**2 - x**2 * y**4)
        + 240 * (y**2 * z**4 - x**2 * z**4)
        - 135 * (x**4 * z**3 + y**4 * z**3)
    )


sphere_mesh = pv.Sphere(theta_resolution=200, phi_resolution=200)
vertices = sphere_mesh.points

dists = f(vertices[:, 0], vertices[:, 1], vertices[:, 2])

mesh1 = sphere_mesh.copy()
mesh2 = sphere_mesh.copy()

mesh1["dist"] = dists
mesh2["dist"] = -dists
clipped1 = mesh1.clip_scalar("dist", value=0)
clipped2 = mesh2.clip_scalar("dist", value=0)

pv.set_plot_theme("document")
pltr = pv.Plotter(window_size=[512, 512], off_screen=False)
pltr.add_mesh(clipped1, smooth_shading=True, color="yellow")
pltr.add_mesh(clipped2, smooth_shading=True, color="purple")
pltr.show()


boundary = clipped1.extract_feature_edges(
    boundary_edges=True,
    non_manifold_edges=False,
    feature_edges=False,
    manifold_edges=False,
)

pltr = pv.Plotter(window_size=[512, 512], off_screen=False)
pltr.add_mesh(clipped1, smooth_shading=True, color="yellow")
pltr.add_mesh(clipped2, smooth_shading=True, color="purple")
pltr.add_mesh(boundary, color="black", line_width=3, render_lines_as_tubes=True)
pltr.show()


def boundary_lines(boundary):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    points = boundary.points
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


line = boundary_lines(boundary)


spline = pv.Spline(boundary.points, 10000)
spline["scalars"] = np.arange(spline.n_points)
tube = spline.tube(radius=0.005)

pltr = pv.Plotter(window_size=[512, 512], off_screen=False)
pltr.add_mesh(clipped1, smooth_shading=True, color="yellow")
pltr.add_mesh(clipped2, smooth_shading=True, color="purple")
pltr.add_mesh(tube, color="black")
pltr.show()
