from math import sqrt, pi
import numpy as np
import pyvista as pv

T = 1
G = -1


def f(ρ, θ, ϕ):
    x = ρ * np.cos(θ) * np.sin(ϕ)
    y = ρ * np.sin(θ) * np.sin(ϕ)
    z = ρ * np.cos(ϕ)
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    sin_z = np.sin(z)
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    cos_z = np.cos(z)
    return (
        (
            np.cos(
                x
                - (-sin_x * sin_y + cos_x * cos_z)
                * T
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            * np.sin(
                y
                - (-sin_y * sin_z + cos_y * cos_x)
                * T
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            + np.cos(
                y
                - (-sin_y * sin_z + cos_y * cos_x)
                * T
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            * np.sin(
                z
                - (-sin_z * sin_x + cos_z * cos_y)
                * T
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            + np.cos(
                z
                - (-sin_z * sin_x + cos_z * cos_y)
                * T
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            * np.sin(
                x
                - (-sin_x * sin_y + cos_x * cos_z)
                * T
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
        )
    ) * (
        (
            np.cos(
                x
                - (-sin_x * sin_y + cos_x * cos_z)
                * G
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            * np.sin(
                y
                - (-sin_y * sin_z + cos_y * cos_x)
                * G
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            + np.cos(
                y
                - (-sin_y * sin_z + cos_y * cos_x)
                * G
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            * np.sin(
                z
                - (-sin_z * sin_x + cos_z * cos_y)
                * G
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            + np.cos(
                z
                - (-sin_z * sin_x + cos_z * cos_y)
                * G
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
            * np.sin(
                x
                - (-sin_x * sin_y + cos_x * cos_z)
                * G
                / np.sqrt(
                    (-sin_x * sin_y + cos_x * cos_z) ** 2
                    + (-sin_y * sin_z + cos_y * cos_x) ** 2
                    + (-sin_z * sin_x + cos_z * cos_y) ** 2
                )
            )
        )
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
Rho, Theta, Phi = np.mgrid[0:8:200j, 0:(pi):200j, 0 : (2 * pi) : 200j]
# create a structured grid
grid = pv.StructuredGrid(Rho, Theta, Phi)
# grid.points = np.transpose(sph2cart(grid.points))


values = f(Rho, Theta, Phi)
grid.point_data["values"] = values.ravel(order="F")
# compute one isosurface
isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()
mesh.points = np.transpose(sph2cart(mesh.points))
mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)

mesh.plot(
    smooth_shading=True, specular=5, cmap="turbo", log_scale=False,
    show_scalar_bar=True, flip_scalars=False    
)