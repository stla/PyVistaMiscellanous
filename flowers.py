# -*- coding: utf-8 -*-
import os
import numpy as np
import pyvista as pv

def f1(u, v):
    R = np.cos(v)*np.cos(v) * np.maximum(abs(np.sin(4*u)), 0.9-0.2*abs(np.cos(8*u)))
    return [
        R*np.cos(u)*np.cos(v),
        R*np.sin(u)*np.cos(v),
        R*np.sin(v)*0.5
    ]

def f2(u,v):
    R = np.cos(v)*np.cos(v) * abs(np.sin(4*u))
    return [
        R*np.cos(u)*np.cos(v),
        R*np.sin(u)*np.cos(v),
        R*np.sin(v)
    ]

def f3(u,v):
    R = np.cos(v)*np.cos(v) * 0.9-0.2*abs(np.cos(8*u))
    return [
        R*np.cos(u)*np.cos(v),
        R*np.sin(u)*np.cos(v),
        R*np.sin(v)
    ]


def Flower(
    which, nu = 200, nv = 200, nframes=180, gifname=None, 
    convert="magick convert", delay=8, color="hotpink", bg_color="#363940"
):
    angle_u = np.linspace(0, 2*np.pi, nu) 
    angle_v = np.linspace(0, 2*np.pi, nv)
    u, v = np.meshgrid(angle_u, angle_v)
    x, y, z = f1(u, v) if which == 1 else (f2(u,v) if which == 2 else f3(u,v))
    grid = pv.StructuredGrid(x, y, z)
    mesh = grid.extract_geometry()#.clean(tolerance=1e-6)
    angle1_ = np.linspace(0, 360, nframes+1)[:nframes]
    angle2_ = 360 * np.sin(np.linspace(0, 2*np.pi, nframes+1)[:nframes])
    #angle2_ = 360 * np.linspace(0, 1, nframes+1)[:nframes]**2
    anim = False
    if gifname is not None:
        anim = True
        gif_sansext, file_extension = os.path.splitext(os.path.basename(gifname))
        screenshotfmt = gif_sansext + "_%03d.png"
        screenshotglob = gif_sansext + "_*.png"
    for i, (angle1, angle2) in enumerate(zip(angle1_, angle2_)):
        pltr = pv.Plotter(window_size=[512,512], off_screen=anim)
        pltr.set_background(bg_color)
        pltr.add_mesh(mesh, pbr=True, specular=15, color=color, metallic=20)
        pltr.camera_position = "xz"
        pltr.camera.roll = angle1
        pltr.camera.azimuth = angle2
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



Flower(
    1, nframes=180, gifname="Flower1.gif", color="fuchsia"
)