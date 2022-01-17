from math import sqrt
import numpy as np
import pyvista as pv

def f(x, y, z, a=0.6, b=0.2):
    return (
        (-2*(a+b)*(x**2+y**2)**2+(a-b)*((x**3-3*x*y**2)*(x**2+y**2+1-z**2)
        -2 * (3 * x**2 * y - y**3) * z))**2
        -(x**2 + y**2) * ((a + b) * (x**2 + y**2) * (x**2 + y**2 + 1 + z**2)
        -2*(a-b)*(x**3-3*x*y**2-z*(3*x**2*y-y**3))-2*a*b*(x**2+y**2))**2
    )

# generate data grid for computing the values
X, Y, Z = np.mgrid[(-2):2:250j, (-2):2:250j, (-1):1:250j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")
# compute the isosurface f(x, y, z) = 0
isosurf = grid.contour(isosurfaces=[0], preference="cell")
mesh = isosurf.extract_geometry()

# surface clipped to the box:
#mesh.plot(smooth_shading=True, color="yellowgreen", specular=15)
mesh["dist"] = np.linalg.norm(mesh.points, axis=1)
mesh.plot(cmap="turbo")

###############################
# compute and assign the values
values = f(X, Y, Z, 0.5, -0.5)
grid.point_data["values"] = values.ravel(order="F")
# compute the isosurface f(x, y, z) = 0
isosurf = grid.contour(isosurfaces=[0])
mesh2 = isosurf.extract_geometry()

# surface clipped to the box:
#mesh.plot(smooth_shading=True, color="yellowgreen", specular=15)
mesh2["dist"] = np.linalg.norm(mesh2.points, axis=1)# mesh["dist"]

mesh2.plot(
    smooth_shading=True, cmap="inferno", window_size=[512, 512],
    show_scalar_bar=False, specular=15, show_axes=False, zoom=1.2,
    background="#363940"#, screenshot="Togliatti_python.png"
)