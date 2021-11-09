import os
from math import cos, sin, sqrt, pi
import numpy as np
import pyvista as pv
from planegeometry.geometry import Triangle

def Gamma(t, A, nlobes):
	alpha = np.pi/2 - (np.pi/2-A)*np.cos(nlobes*t)
	beta = t + A*np.sin(2*nlobes*t)
	return np.array([
      np.sin(alpha) * np.cos(beta),
      np.sin(alpha) * np.sin(beta),
      np.cos(alpha)
	])

def HopfInverse(p, phi):
	return np.array([
      (1+p[2])*np.cos(phi),
      p[0]*np.sin(phi) - p[1]*np.cos(phi), 
      p[0]*np.cos(phi) + p[1]*np.sin(phi),
      (1+p[2])*np.sin(phi)
	]) / np.sqrt(2*(1+p[2]))

def Stereo(q):
	return 2*q[0:3] / (1-q[3])

def F(t, phi, A, nlobes):
	return Stereo(HopfInverse(Gamma(t, A, nlobes), phi))

def HopfTorusMesh(nu=400, nv=200, A=0.44, nlobes=3):
    angle_u = np.linspace(-np.pi, np.pi, nu) 
    angle_v = np.linspace(0, np.pi, nv)
    u, v = np.meshgrid(angle_u, angle_v)
    z, x, y = F(u, v, A, nlobes)
    grid = pv.StructuredGrid(x, y, z)
    mesh = grid.extract_geometry().clean(tolerance=1e-6)
    return mesh



# -----------------------------------------------------------------------------
def Hexlet(Center, Radius, HTmesh, nframes, gifname, convert="magick convert", delay=8, bgcolor="#363940", tori_color="orangered"):
    s = Radius # side length of the hexagon 
    Coef = 2/3 
    a = Coef*(Radius+s/2)/sin(pi/2-2*pi/6)
    I = np.array([a, 0]) # inversion pole
    ## ------------------------------------------------------------------ //// 
    O1 = np.array([2*a, 0, 0]) 
    # interior sphere
    def inversion(M, RRadius):
        II = np.array([Coef*(RRadius+RRadius/2)/sin(pi/2-2*pi/6), 0])
        S = Coef*(RRadius+RRadius/2) * np.array([cos(2*pi/6), sin(2*pi/6)]) 
        k = np.vdot(S-II, S-II) # negated inversion constant
        M = np.asarray(M, dtype=float)
        IM = M-II
        return II - k/np.vdot(IM,IM)*IM 
    SmallRadius = Coef*(Radius-s/2)
    p1 = inversion((SmallRadius,0), Radius)
    p2 = inversion((0,SmallRadius), Radius)
    p3 = inversion((-SmallRadius,0), Radius)
    tr = Triangle(p1, p2, p3)
    cs = tr.circumcircle()
    
    shift = pi/90
    frames = np.linspace(1, 180, nframes) 
    anim = False
    if not gifname is None:
        gif_sansext, file_extension = os.path.splitext(os.path.basename(gifname))
        anim = True
        screenshotfmt = gif_sansext + "_%03d.png"
        screenshotglob = gif_sansext + "_*.png"
    for frame_number in frames:
        pltr = pv.Plotter(window_size=[512,512], off_screen=anim)
        pltr.set_background(bgcolor)
        i = 1
        while i<= 6:
            beta = i*pi/3 - frame_number*shift; # frame from 1 to 180
            ccenter = Coef*Radius*np.array([cos(beta),sin(beta)])
            p1 = inversion((0,Coef*Radius/2)+ccenter,Radius)
            p2 = inversion((Coef*Radius/2,0)+ccenter,Radius)
            p3 = inversion((0,-Coef*Radius/2)+ccenter,Radius)
            tr = Triangle(p1, p2, p3)
            cs = tr.circumcircle()
            center = np.array([cs.center[0], cs.center[1], 0])
            r = cs.radius
            mesh = HTmesh.copy()
            mesh.scale(r*0.05)
            mesh.rotate_z(2*frame_number)
            mesh.translate(center-O1)
            pltr.add_mesh(mesh, color=tori_color, specular=20, smooth_shading=True)
            i += 1
        pltr.set_focus([0,0,0])
        pltr.set_position([2, 0, 8])
        if anim:
            pltr.show(screenshot=screenshotfmt % frame_number)
        else:
            pltr.show()
    if anim:
        os.system(
            convert + (" -dispose previous -loop 0 -delay %d " % delay) + screenshotglob + " " + gifname    
        ) 


Hexlet((0,0,0), 2, HopfTorusMesh(), nframes=1, gifname=None)#"HopfToriSteinerOrbit.gif")

