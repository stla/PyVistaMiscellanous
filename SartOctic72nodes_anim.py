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


def satellite_motion(nframes, R, alpha=3*pi/4, k=4):
    quats = [None]*nframes
    t_ = np.linspace(0, 2*pi, nframes+1)[:nframes]
    satellite0 = satellite(0, R, alpha, k)
    A = satellite0.copy()
    q0 = quaternion.one
    quats[0] = q0
    for i in range(nframes-1):
        B = satellite(t_[i+1], R, alpha, k)
        q1 = get_quaternion(A/R, B/R) * q0
        quats[i+1] = q1
        A = B
        q0 = q1
    return (satellite0, quats)


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


pos0, quats = satellite_motion(180, 20)

for i, q in enumerate(quats):
    pngname = "zzpic%03d.png" % i
    pltr = pv.Plotter(window_size = [512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(clipped_mesh.center)
    pltr.set_position(pos0)
    h, a, b = quaternion2hab(q)
    pltr.camera.roll = b
    pltr.camera.azimuth = a
    pltr.camera.elevation = h
    pltr.camera.zoom(1)
    pltr.add_mesh(
        clipped_mesh, smooth_shading=True, specular=15, cmap="nipy_spectral", 
        log_scale=False, show_scalar_bar=False, flip_scalars=False 
    )
    pltr.show(screenshot=pngname)

os.system(
    "gifski --frames=zzpic*.png --fps=9 -o SartOctic72nodes.gif"    
)

