import os
from math import pi, atan2, asin, sqrt, cos, sin
import numpy as np
import pyvista as pv
import quaternion


def quaternion2hab(q):
    c = 180 / pi
    t = q.x*q.y + q.z*q.w
    if t > 0.499: # north pole
        heading = 2 * atan2(q.x, q.w) * c
        attitude = 90
        bank = 0
    elif t < - 0.499: # south pole
        heading = - 2 * atan2(q.x, q.w) * c
        attitude = - 90
        bank = 0
    else:
        heading = atan2(2*(q.y*q.w - q.x*q.z) , 1 - 2*(q.y*q.y + q.z*q.z)) * c
        attitude = asin(2*t) * c
        bank = atan2(2*(q.x*q.w - q.y*q.z), 1 - 2*(q.x*q.x + q.z*q.z)) * c
    return (heading, attitude, bank)

# then the rotation is Ry(h) @ Rz(a) @ Rx(b)


def get_quaternion(u ,v): # u and v must be normalized
    "Get a unit quaternion whose corresponding rotation sends u to v"
    d = np.vdot(u, v)
    c = sqrt(1+d)
    r = 1 / sqrt(2) / c
    W = np.cross(u, v)
    arr = np.concatenate((np.array([c/sqrt(2)]), r*W))
    return quaternion.from_float_array(arr)


def satellite(t, R, alpha, k):
    return R * np.array([
        cos(alpha) * cos(t) * cos(k*t) - sin(t) * sin(k*t),
        cos(alpha) * sin(t) * cos(k*t) + cos(t) * sin(k*t),
        sin(alpha) * cos(k*t)
    ])


nframes = 180
quats = [None]*nframes
t_ = np.linspace(0, 2*pi, nframes+1)[:nframes]
satellite0 = satellite(0, 35, 3*pi/4, 4)
A = satellite0.copy()
q0 = quaternion.one
quats[0] = q0
for i in range(nframes-1):
    B = satellite(t_[i+1], 35, 3*pi/4, 4)
    q1 = get_quaternion(A/35, B/35) * q0
    quats[i+1] = q1
    A = B
    q0 = q1


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
Rho, Theta, Phi = np.mgrid[0:8:300j, 0:(pi):200j, 0 : (2 * pi) : 250j]
# create a structured grid
grid = pv.StructuredGrid(Rho, Theta, Phi)


values = f(Rho, Theta, Phi)
grid.point_data["values"] = values.ravel(order="F")
# compute one isosurface
isosurf = grid.contour(isosurfaces=[0])
mesh = isosurf.extract_geometry()
mesh.points = np.transpose(sph2cart(mesh.points))
mesh.point_data["distance"] = np.linalg.norm(mesh.points, axis=1)


for i, q in enumerate(quats):
    pltr = pv.Plotter(window_size=[512, 512], off_screen=True)
    pltr.add_background_image("SpaceBackground.png")
    pltr.set_focus((0,0,0))
    pltr.set_position(satellite0)
    h, a, b = quaternion2hab(q)
    pltr.camera.roll = b
    pltr.camera.azimuth = a
    pltr.camera.elevation = h
    pltr.camera.zoom(1)
    pltr.add_mesh(
        mesh, smooth_shading=True, specular=5, cmap="plasma", log_scale=False,
        show_scalar_bar=False, flip_scalars=False 
    )
    pngname = "zzpic%03d.png" % i
    pltr.show(screenshot=pngname)


os.system(
    "magick convert -dispose previous -loop 0 -delay 8 zzpic*.png Lattice.gif"    
)
