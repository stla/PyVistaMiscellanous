# -*- coding: utf-8 -*-
import os
from math import sqrt, pi
import pyvista as pv
import numpy as np

def vnormalize(v):
    return v / np.linalg.norm(v, axis=0)

def helix(t, R, r, w):
    return np.array([
        (R + r*np.cos(t))*np.cos(t/w),
        (R + r*np.cos(t))*np.sin(t/w),
        r*np.sin(t)
    ])
 
def dhelix(t, R, r, w):
    return np.array([
        -r*np.sin(t)*np.cos(t/w) - (R+r*np.cos(t))/w*np.sin(t/w),
        -r*np.sin(t)*np.sin(t/w) + (R+r*np.cos(t))/w*np.cos(t/w),
        r*np.cos(t)
    ])
 
def ddhelix(t, R, r, w):
    return vnormalize(np.array([
        -r*np.cos(t)*np.cos(t/w) + r*np.sin(t)/w*np.sin(t/w) + r*np.sin(t)/w*np.sin(t/w) - (R+r*np.cos(t))/w/w*np.cos(t/w),
        -r*np.cos(t)*np.sin(t/w) - r*np.sin(t)/w*np.cos(t/w) - r*np.sin(t)/w*np.cos(t/w) - (R+r*np.cos(t))/w/w*np.sin(t/w),
        -r*np.sin(t)
    ]))

def bnrml(t, R, r, w):
    return vnormalize(
        np.cross(dhelix(t, R, r, w), ddhelix(t, R, r, w), axisa=0, axisb=0, axisc=0)
    )

def f(u, v, R, r, w, a):
    v = v.reshape((1,) + v.shape)
    return helix(u, R, r, w) + a * (np.cos(v)*ddhelix(u, R, r, w) + np.sin(v)*bnrml(u, R, r, w)) 

def ToroidalHelix(
    nu = 1000, nv = 100, w = 10, R = 4, r = 1, a = 0.65, nframes=180, gifname=None, convert="magick convert", delay=8,
    color="darkred", texture = "ElectricityTexture.jpg", bg_image = "SpaceBackground.png"
):
    angle_u = np.linspace(0, w*2*pi, nu) 
    angle_v = np.linspace(0, 2*pi, nv)
    u, v = np.meshgrid(angle_u, angle_v)
    x, y, z = f(u, v, R, r, w, a)
    grid = pv.StructuredGrid(x, y, z)
    mesh = grid.extract_geometry()#.clean(tolerance=1e-6)
    if texture is not None:
        tex = pv.read_texture(texture)
        mesh.texture_map_to_sphere(inplace=True)
    angle_ = np.linspace(0, 360, nframes+1)[:nframes]
    anim = False
    if gifname is not None:
        anim = True
        gif_sansext, file_extension = os.path.splitext(os.path.basename(gifname))
        screenshotfmt = gif_sansext + "_%03d.png"
        screenshotglob = gif_sansext + "_*.png"
    for i, angle in enumerate(angle_):
        pltr = pv.Plotter(window_size=[512,512], off_screen=anim)
        if bg_image is not None:
            pltr.add_background_image(bg_image)
        if texture is not None:
            pltr.add_mesh(mesh, texture=tex)
        else:
            pltr.add_mesh(mesh, smooth_shading=True, specular=15, color=color)
        pltr.camera_position = "xz"
        pltr.camera.roll = angle
        pltr.camera.azimuth = angle
        pltr.camera.view_angle = 30.0 
        pltr.camera.reset_clipping_range()
        if anim:
            pngname = screenshotfmt % i
            pltr.show(screenshot=pngname)
        else:
            pltr.show()
    if anim:
        os.system(
            convert + (" -dispose previous -loop 0 -delay %d " % delay) + screenshotglob + " " + gifname    
        ) 



ToroidalHelix(
    nu = 1000, nv = 100, w = 10, R = 4, r = 1, a = 0.65, nframes=1, gifname=None, 
    color="darkred", texture = None, bg_image = "SpaceBackground.png"
)