from math import sqrt, cos, sin, pi
import numpy as np
import pyvista as pv

# Affine rotation ####
#' Matrix of the affine rotation around an axis 
#' @param theta angle of rotation in radians
#' @param P1,P2 the two points defining the axis of rotation 
def AffineRotationMatrix(theta, P1, P2):
    T = np.vstack(
        (
            np.hstack((np.eye(3), -P1.reshape(3,1))),
            np.array([0, 0, 0, 1])
        )
    )
    invT = np.vstack(
        (
            np.hstack((np.eye(3), P1.reshape(3,1))),
            np.array([0, 0, 0, 1])
        )
    )
    a, b, c = (P2 - P1) / np.linalg.norm(P2 - P1)
    d = sqrt(b*b + c*c)
    if d > 0:
        Rx = np.array([
            [1, 0, 0, 0],
            [0, c/d, -b/d, 0],
            [0, b/d, c/d, 0],
            [0, 0, 0, 1]
        ])
        invRx = np.array([
            [1, 0, 0, 0],
            [0, c/d, b/d, 0],
            [0, -b/d, c/d, 0],
            [0, 0, 0, 1]
        ])
    else:
        Rx = invRx = np.eye(4)
    Ry = np.array([
        [d, 0, -a, 0],
        [0, 1, 0, 0],
        [a, 0, d, 0],
        [0, 0, 0, 1]
    ])
    invRy = np.array([
        [d, 0, a, 0],
        [0, 1, 0, 0],
        [-a, 0, d, 0],
        [0, 0, 0, 1]
    ])
    Rz = np.array([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta), cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return invT @ invRx @ invRy @ Rz @ Ry @ Rx @ T

O = np.array([0.0, 0.0, 0.0])
A = np.array([0.0, 10.0, 0.0])
Rot = AffineRotationMatrix(3*pi/4, O, A)


def f(x, y, z, a, b):
    return ((
        (x * x + y * y + 1) * (a * x * x + b * y * y)
        + z * z * (b * x * x + a * y * y)
        - 2 * (a - b) * x * y * z
        - a * b * (x * x + y * y)
    ) ** 2 
    - 4 * (x * x + y * y) * (a * x * x + b * y * y - x * y * z * (a - b)) ** 2)

def inversion(omega, M):
    Omega0 = np.array([omega, 0.0, 0.0])
    OmegaM = M - Omega0;
    k = np.dot(OmegaM, OmegaM)
    return Omega0 + OmegaM / k

def params(alpha, gamma, mu):
    beta = sqrt(alpha*alpha - gamma*gamma)
    theta = beta * sqrt(mu * mu - gamma*gamma)
    omega = (alpha * mu + theta) / gamma
    ratio = (
        (mu - gamma) * ((alpha - gamma) * (mu + gamma) + theta) 
        / ((alpha + gamma) * (mu - gamma) + theta) / (alpha - gamma)
    )
    R = (
        1/ratio * gamma * gamma / ((alpha - gamma) * (mu - gamma) + theta) 
        * (mu - gamma) / ((alpha + gamma) * (mu - gamma) + theta)
    )
    omegaT = (
        omega - (beta * beta * (omega - gamma)) 
        / ((alpha - gamma) * (mu + omega) - beta * beta) 
        / ((alpha + gamma) * (omega - gamma) + beta * beta)
    )
    return (omega, omegaT, ratio, R)

alpha = 0.97
gamma = 0.32
mu = 0.56
omega, omegaT, ratio, R = params(alpha, gamma, mu)
OmegaT = np.array([omegaT, 0.0, 0.0])
a = ratio*ratio
b = 0.06

# generate data grid for computing the values
X, Y, Z = np.mgrid[(-1.3):1.3:350j, (-1.6):1.6:350j, (-0.6):0.6:350j]
# create a structured grid
grid = pv.StructuredGrid(X, Y, Z)
# compute and assign the values
values = f(X, Y, Z, a, b)
grid.point_data["values"] = values.ravel(order="F")
# compute the isosurface f(x, y, z) = 0
isosurf = grid.contour(isosurfaces=[0])
# convert to a PolyData mesh
mesh = isosurf.extract_geometry()
# rotate mesh
mesh.transform(Rot)
# transform 
points = R * mesh.points
points = np.apply_along_axis(lambda M: inversion(omega, M + OmegaT), 1, points)
newmesh = pv.PolyData(points, mesh.faces)
newmesh["dist"] = np.linalg.norm(mesh.points, axis=1)

pltr = pv.Plotter(window_size=[512, 512])
pltr.set_focus(newmesh.center)
pltr.set_position(newmesh.center - np.array([0.0, 0.0, 7.0]))
pltr.add_background_image("SpaceBackground.png")
pltr.add_mesh(
    newmesh, smooth_shading=True, cmap="turbo", specular=25,
    show_scalar_bar=False
)
pltr.show()
