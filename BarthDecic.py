from math import sqrt, pi
import numpy as np
import pyvista as pv

phi = (1 + sqrt(5)) / 2
def f(ρ, θ, ϕ):
    x = ρ * np.cos(θ) * np.sin(ϕ)
    y = ρ * np.sin(θ) * np.sin(ϕ)
    z = ρ * np.cos(ϕ)
    return (
        (18+30*phi) * x**4 * y**4
        + (-36 - 30*phi + 44*phi**2 - 10*phi**3) * x**2
        + (-18 - 24*phi + 10*phi**2) * x**6
        + (3 + 5*phi) * x**8
        + (36 + 60*phi) * x**2 * y**2 * z**4
        + (12 + 20*phi) * x**2 * y**6
        + phi
        + (-16 + 8*phi**4 - 8*phi**8 + 16*phi**12) * x**2 * y**4 * z**4
        + (8 * phi**8) * y**2 * z**8
        + (-18 - 24*phi + 10*phi**2) * z**6
        + (-8*phi**4 - 16*phi**8) * x**6 * z**4
        + (16*phi**4 + 8*phi**8) * x**6 * y**4
        + (-8*phi**4) * y**8 * z**2
        + (-18 - 24*phi + 10*phi**2) * y**6
        + (12 + 20*phi) * x**2 * z**6
        + (36 + 60*phi) * x**4 * y**2 * z**2
        + (36 + 60*phi) * x**2 * y**4 * z**2
        + (8 + 16*phi**4 -16*phi**8 - 8*phi**12) * x**2 * y**2 * z**6
        + (-54 - 72*phi + 30*phi**2) * y**4 * z**2
        + (-8*phi**4) * x**8 * y**2
        + (16*phi**4 + 8*phi**8) * y**6 * z**4
        + (12 + 20*phi) * y**2 * z**6
        + (3 + 5*phi) * z**8
        + (-8*phi**4) * x**2 * z**8
        + (39 + 41*phi - 37*phi**2 + 5*phi**3) * z**4
        + (-54 - 72*phi + 30*phi**2) * x**2 * y**4
        + (8 + 16*phi**4 -16*phi**8 - 8*phi**12) * x**6 * y**2 * z**2
        + (-54 - 72*phi + 30*phi**2) * x**2 * z**4
        + (12 + 20*phi) * x**6 * z**2
        + (-16 + 8*phi**4 - 8*phi**8 + 16*phi**12) * x**4 * y**2 * z**4
        + (16*phi**4 + 8*phi**8) * x**4 * z**6
        + (39 + 41*phi - 37*phi**2 + 5*phi**3) * y**4
        + (-36 - 30*phi + 44*phi**2 - 10*phi**3) * z**2
        + (8*phi**8) * x**2 * y**8
        + (12 + 20*phi) * y**6 * z**2
        + (8*phi**8) * x**8 * z**2
        + (-36 - 30*phi + 44*phi**2 - 10*phi**3) * y**2
        + (12 + 20*phi) * x**6 * y**2
        + (-8*phi**4 - 16*phi**8) * y**4 * z**6
        + (-16 + 8*phi**4 - 8*phi**8 + 16*phi**12) * x**4 * y**4 * z**2
        + (78 + 82*phi - 74*phi**2 + 10*phi**3) * x**2 * z**2
        + (18+30*phi) * x**4 * z**4
        + (-8*phi**4 - 16*phi**8) * x**4 * y**6
        + (-54 - 72*phi + 30*phi**2) * x**4 * y**2
        + (-54 - 72*phi + 30*phi**2) * x**4 * z**2
        + (-54 - 72*phi + 30*phi**2) * y**2 * z**4
        + (78 + 82*phi - 74*phi**2 + 10*phi**3) * x**2 * y**2
        + (-108 - 144*phi + 60*phi**2) * x**2 * y**2 * z**2
        + (18+30*phi) * y**4 * z**4
        + (3 + 5*phi) * y**8
        + (78 + 82*phi - 74*phi**2 + 10*phi**3) * y**2 * z**2
        + (8 + 16*phi**4 -16*phi**8 - 8*phi**12) * x**2 * y**6 * z**2
        + (39 + 41*phi - 37*phi**2 + 5*phi**3) * x**4
    )


def sph2cart(sph):
    ρ = sph[:, 0]
    θ = sph[:, 1]
    ϕ = sph[:, 2]
    return np.array([
        ρ * np.cos(θ) * np.sin(ϕ),
        ρ * np.sin(θ) * np.sin(ϕ),
        ρ * np.cos(ϕ)
    ])


# generate data grid for computing the values
b = sqrt((5+sqrt(5))/2)
Rho, Theta, Phi = np.mgrid[0:b:400j, 0:(pi):400j, 0:(2*pi):400j]
# create a structured grid
grid = pv.StructuredGrid(Rho, Theta, Phi)
#grid.points = np.transpose(sph2cart(grid.points))


values = f(Rho, Theta, Phi)
grid.point_data["values"] = values.ravel(order="F") 
# compute one isosurface
isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()
mesh.points = np.transpose(sph2cart(mesh.points))
