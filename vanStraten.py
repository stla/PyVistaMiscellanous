import numpy as np
import pyvista as pv


def f(x, y, z):
    return (
      0.99*(64*(0.5*z)**7-112*(0.5*z)**5+56*(0.5*z)**3-7*(0.5*z)-1) 
        + (0.7818314825-0.3765101982*y-0.7818314825*x) 
        * (0.7818314824-0.8460107361*y-0.1930964297*x)  
        * (0.7818314825-0.6784479340*y+0.5410441731*x)  
        * (0.7818314825+0.8677674789*x) 
        * (0.7818314824+0.6784479339*y+0.541044172*x) 
        * (0.7818314824+0.8460107358*y-0.193096429*x) 
        * (0.7818314821+0.3765101990*y-0.781831483*x)
    )


# generate data grid for computing the values
X, Y, Z = np.mgrid[(-6):6:150j, (-6):6:150j, (-6):6:150j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = f(X, Y, Z)
grid.point_data["values"] = values.ravel(order="F")
# compute the isosurface f(x, y, z) = 0
isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()


# surface clipped to the ball of radius 6, with the help of `clip_scalar`:
mesh["dist"] = np.linalg.norm(mesh.points, axis=1)
clipped_mesh = mesh.clip_scalar("dist", value=6)
clipped_mesh.plot(
    smooth_shading=True,
    cmap="nipy_spectral",
    window_size=[512, 512],
    show_scalar_bar=False,
    specular=15,
    show_axes=False,
    zoom=1.3,
    background="#363940"
#    screenshot="SartOctic72nodes.png",
)
