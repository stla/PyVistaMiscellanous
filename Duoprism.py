# -*- coding: utf-8 -*-
import os
from math import sqrt, sin, cos, pi
import pyvista as pv
import numpy as np

# stereographic projection
def stereog(v):
    return v[0:3] / (sqrt(2) - v[3])

# tubular segment
def segment(P, Q, r):
    line = pv.Line(P, Q)
    return line.tube(radius=r)


# rotation in 4D space (right-isoclinic) ######################################
def rotate4d(alpha, beta, xi, vec):
    a = cos(xi)
    b = sin(alpha) * cos(beta) * sin(xi)
    c = sin(alpha) * sin(beta) * sin(xi)
    d = cos(alpha) * sin(xi)
    p = vec[0]
    q = vec[1]
    r = vec[2]
    s = vec[3]
    return np.array(
        [
            a * p - b * q - c * r - d * s,
            a * q + b * p + c * s - d * r,
            a * r - b * s + c * p + d * q,
            a * s + b * r - c * q + d * p
        ]
    )

def Duoprism(A, B, segradius, xi_, vertices_color="#EEC900", edges_color="gold", gifname=None, convert="magick convert", delay=8):
    """
    :param A: number of sides of the first polygon
    :param B: number of sides of the second polygon
    :param segradius: radius of the tubular segments (the edges)
    :param xi_: vector of angles of rotations, e.g. `np.linspace(0, pi, 61)[:60]`
    :param vertices color: color for the vertices, plotted as balls
    :param edges color: color for the edges
    :param gifname: name of output gif; set to `None` if you don't want an animation
    :param convert: the ImageMagick main command (`magick convert` on Windows)
    :param delay`: delay of frames in the animation 
    """
    anim = False
    if gifname is not None:
        anim = True
        gif_sansext, file_extension = os.path.splitext(os.path.basename(gifname))
        screenshotfmt = "zpic_" + gif_sansext + "_%03d.png"
        screenshotglob = "zpic_" + gif_sansext + "_*.png"

    # construction of the vertices
    vertices = np.empty((A, B, 4))
    for i in range(A):
        v1 = [cos(i/A*2*pi), sin(i/A*2*pi)]
        for j in range(B):
            v2 = [cos(j/B*2*pi), sin(j/B*2*pi)]
            vertices[i, j, :] = np.array(v1 + v2)
            
    # construction of the edges
    edges = np.empty((2, 2, 2*A*B), dtype=int)

    def dominates(c1, c2):
        return c2[0]>c1[0] or (c2[0]==c1[0] and c2[1]>c1[1])
        
    counter = 0
    for i in range(A):
        for j in range(B):
            c1 = np.array([i, j])
            candidate = np.array([i, (j-1) % B])
            if dominates(c1, candidate):
                edges[:, :, counter] = np.column_stack((c1, candidate)) 
                counter += 1
            candidate = np.array([i, (j+1) % B])
            if dominates(c1, candidate):
                edges[:, :, counter] = np.column_stack((c1, candidate))
                counter += 1
            candidate = np.array([(i-1) % A, j])
            if dominates(c1, candidate):
                edges[:, :, counter] = np.column_stack((c1, candidate))
                counter += 1
            candidate = np.array([(i+1) % A, j])
            if dominates(c1, candidate):
                edges[:, :, counter] = np.column_stack((c1, candidate))
                counter += 1

    ###############################################################################
    alpha = pi/2
    beta = 0
    #xi_ = np.linspace(0, pi, 61)[:60]
    if np.isscalar(xi_):
        xi_ = [xi_]

    for idx, xi in enumerate(xi_):
        # rotated vertices
        Vertices = np.apply_along_axis(
            lambda v: rotate4d(alpha, beta, xi, v), 2, vertices
        )
        # projected vertices
        vs = np.apply_along_axis(stereog, 2, Vertices)
        ####~~~~ plot ~~~~####
        pltr = pv.Plotter(window_size=[512, 512], off_screen=anim)
#        pltr.set_background("#363940")
        pltr.add_background_image("SpaceBackground.png")
        ## plot the edges
        for k in range(2*A*B):
            v1 = edges[:, 0, k]
            v2 = edges[:, 1, k]
            P = vs[v1[0], v1[1], :]
            Q = vs[v2[0], v2[1], :]
            edge = segment(P, Q, segradius)
            pltr.add_mesh(
              edge, color = edges_color, specular=10, pbr=True, metallic=20
            )
        ## plot the vertices
        for i in range(A):
            for j in range(B):
                v = vs[i, j, :]
                ball = pv.Sphere(2*segradius, center = v)
                pltr.add_mesh(
                    ball, color=vertices_color, specular=10, pbr=True, metallic=20
                )
        pltr.set_focus([0, 0, 0])
        pltr.set_position([9, 0, 9])
        if anim:
            pltr.show(screenshot=screenshotfmt % idx)
        else:
            pltr.show()
    if anim:
        os.system(
            convert + (" -dispose previous -loop 0 -delay %d " % delay) + screenshotglob + " " + gifname    
        ) 

Duoprism(3, 30, 0.1, xi_ = np.linspace(0, 2*pi/3, 61)[:60], gifname = "Duoprism_3-30.gif")
