from math import sqrt, pi
import numpy as np
import pyvista as pv

def f(ρ, θ, ϕ):
    x = ρ * np.cos(θ) * np.sin(ϕ)
    y = ρ * np.sin(θ) * np.sin(ϕ)
    z = ρ * np.cos(ϕ)
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
Rho, Theta, Phi = np.mgrid[0:4.8:200j, 0:pi:200j, 0:(2*pi):200j]
# create a structured grid
grid = pv.StructuredGrid(Rho, Theta, Phi)

values = f(Rho, Theta, Phi)
grid.point_data["values"] = values.ravel(order="F")
# compute the isosurface f(ρ, θ, ϕ) = 0
isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()

# spherical to Cartesian:
def sph2cart(sph):
    ρ = sph[:, 0]
    θ = sph[:, 1]
    ϕ = sph[:, 2]
    return np.array([
        ρ * np.cos(θ) * np.sin(ϕ),
        ρ * np.sin(θ) * np.sin(ϕ),
        ρ * np.cos(ϕ)
    ])

mesh.points = np.transpose(sph2cart(mesh.points))
mesh.plot(smooth_shading=True, color="hotpink")
