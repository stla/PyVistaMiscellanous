# -*- coding: utf-8 -*-
import os
from math import atan, cos, pi, sin, sqrt
import numpy as np
import pyvista as pv
from torus_three_points.main import torusThreePoints # https://github.com/stla/PyTorusThreePoints
import seaborn as sb
import quaternion

##### THIS PART IS FOR THE CAMERA MOVEMENT #####

# the seam line of a tennis ball
def TennisSeam(t, A=0.44, n=2):
	alpha = pi/2 - (pi/2-A)*cos(n*t)
	beta = t + A*sin(2*n*t)
	return np.array([
      cos(alpha),
      sin(alpha) * sin(beta),
      sin(alpha) * cos(beta)
	])


def transform_matrix_from_rotation_matrix(rotation_matrix):
    return np.vstack(
        (np.hstack(
            (rotation_matrix, np.array([0, -3, 0]).reshape((3,1)))), 
            np.array([0, 0, 0, 1])
        )
    )


def get_quaternion(u, v): # u and v must be normalized
    "Get a unit quaternion whose corresponding rotation sends u to v"
    c = sqrt(1 + np.vdot(u, v))
    r = 1 / sqrt(2) / c
    W = np.cross(u, v)
    arr = np.concatenate((np.array([c/sqrt(2)]), r*W))
    return quaternion.from_float_array(arr)
    



###############################################################################
def HopfInverse(p, xi):
	cos_xi = np.cos(xi)
	sin_xi = np.sin(xi)
	return np.array([
        (1+p[2]) * cos_xi,
        p[0] * sin_xi - p[1] * cos_xi, 
        p[0] * cos_xi + p[1] * sin_xi,
        (1+p[2]) * sin_xi
	]) / np.sqrt(2*(1+p[2]))

def Stereo(q):
	return 2*q[0:3] / (1-q[3])

def StereoCircHinv(θ, ϕ, xi):
    p = np.array([
        np.cos(θ) * np.cos(ϕ),
        np.sin(θ) * np.cos(ϕ),
        np.sin(ϕ)
    ])
    return Stereo(HopfInverse(p, xi))

def tripletOnCircle(θ):
     ϕ = atan(0.5 * cos(θ))
     return [
         StereoCircHinv(θ, ϕ, 0),
         StereoCircHinv(θ, ϕ, 2),
         StereoCircHinv(θ, ϕ, 4)
     ]

######################################################################

def HopfTorusGreatCircle(ncircles, nframes, gifname = None, convert="magick convert", delay=8):
    matrices = []
    t_ = np.linspace(pi/2, 5*pi/2, nframes+1)
    A = TennisSeam(t_[0])
    for i in range(nframes):
        B = TennisSeam(t_[i+1])
        q1 = get_quaternion(A, B)
        rmatrix = quaternion.as_rotation_matrix(q1)
        A = B
        matrices.append(transform_matrix_from_rotation_matrix(rmatrix))

    if ncircles % 2 == 1:
        ncircles += 1
    myTriplets = [tripletOnCircle(θ) for θ in np.linspace(0, 2*pi, ncircles)[:ncircles - 1]]

    colors = sb.color_palette(palette="rocket", n_colors=ncircles//2)
    colors_copy = colors[:ncircles//2-1].copy()
    colors_copy.reverse()
    colors = colors + colors_copy

    anim = False
    if gifname is not None:
        anim = True
        gif_sansext, file_extension = os.path.splitext(os.path.basename(gifname))
        screenshotfmt = gif_sansext + "_%03d.png"
        screenshotglob = gif_sansext + "_*.png"

    for i, matrix in enumerate(matrices):
        pltr = pv.Plotter(window_size=[512, 512], off_screen=anim)
        pltr.set_background("seashell")
        for light in pltr.renderer.lights:
            light.transform_matrix = matrix
            light.position = matrix[0:3, 0:3].dot(np.asarray(light.position))
        for j, triplet in enumerate(myTriplets):
            p1, p2, p3 = triplet
            torusThreePoints(
                pltr, p1, p2, p3, 0.2, pbr = True, ambient=10,
                color=sb.saturate(colors[j]), specular=20, diffuse=10, 
                metallic=100
            )
        pltr.set_position(22 * TennisSeam(t_[i]))
        pltr.camera.view_angle = 40.0
        pltr.camera.model_transform_matrix = matrix
        pltr.set_focus([0, 0, 0])
        if anim:
            pngname = screenshotfmt % i
            pltr.show(screenshot=pngname)
        else:
            pltr.show()
    if anim:
        os.system(
            convert + (" -dispose previous -loop 0 -delay %d " % delay) + screenshotglob + " " + gifname    
        ) 



HopfTorusGreatCircle(ncircles=32, nframes=1)