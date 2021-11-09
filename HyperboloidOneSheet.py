import os
from math import sqrt, pi
import pyvista as pv
import numpy as np

def index_of(l, x):
    try:
    	i = l.index(x)
    except:
    	i = -1
    finally:
    	return i

def hyperboloidMesh0(a, b, c, signature, nu, nv, vmin):
    if index_of(["++-", "+-+", "-++"], signature) == -1:
        raise ValueError("Invalid signature.")
    if vmin >= 1 or a <= 0 or b <= 0 or c <= 0:
        raise ValueError("xx")
    if signature == "+-+":
        points, faces = hyperboloidMesh0(a, c, b, "++-", nu, nv, vmin)
        points = points[:, [0,2,1]]
        faces = faces[:, [3,2,1,0]]
        return (points, faces)
    if signature == "-++":
        points, faces = hyperboloidMesh0(c, b, a, "++-", nu, nv, vmin)
        points = points[:, [2,1,0]]
        faces = faces[:, [3,2,1,0]]
        return (points, faces)
    c0 = c
    if b > a:
        exchange = True
        a0 = b
        b0 = a
    else:
        exchange = False
        a0 = a
        b0 = b
    Mu2 = a0*a0
    h2ab = Mu2 - b0*b0
    h2ac = c0*c0 + Mu2
    c2 = 1
    a2 = c2 + h2ac
    b2 = a2 - h2ab
    h2bc = b2 - c2
    #
    vertices = np.empty((nu*nv, 3), dtype=float)
    indices = np.empty(((nu-1)*(nv-1), 4), dtype=int)
    v_ = np.linspace(vmin, c2, nv)
    #
    if a0 != b0:
        u_ = np.linspace(b2, a2, nu)
        x = a0 / sqrt(h2ac*h2ab) * np.sqrt(a2-u_)
        y = b0 / sqrt(h2bc*h2ab) * np.sqrt(u_-b2)
        z = c0 / sqrt(h2bc*h2ac) * np.sqrt(u_-c2)
    else:
        u_ = np.linspace(0, 2*pi, nu+1)[:nu]
        z = np.repeat(c0/sqrt(h2ac), nu)
        mxy = a0 / sqrt(h2ac)
        x = mxy * np.cos(u_)
        y = mxy * np.sin(u_)
    for i in range(nu):
        for j in range(nv):
            vertices[i*nv+j, :] = np.array(
                [
                    x[i] * sqrt(a2-v_[j]),
                    y[i] * sqrt(b2-v_[j]),
                    z[i] * sqrt(c2-v_[j])
                ]
        )
    # quads
    for i in range(nu-1):
        ip1 = i+1
        for j in range(nv-1):
            jp1 = j+1
            quad = [i*nv+j, i*nv+jp1, ip1*nv+jp1, ip1*nv+j]
            if exchange:
                quad.reverse()
            indices[i*(nv-1)+j, :] = quad
    vs = vertices.copy()
    vs[:, 0] *= -1
    vertices = np.vstack((vertices, vs))
    ids = indices[:, [3,2,1,0]]
    indices = np.vstack((indices, ids + nu*nv))
    vs = vertices.copy()
    vs[:, 1] *= -1
    vertices = np.vstack((vertices, vs))
    ids = indices[:, [3,2,1,0]]
    indices = np.vstack((indices, ids + 2*nu*nv))
    vs = vertices.copy()
    vs[:, 2] *= -1
    vertices = np.vstack((vertices, vs))
    ids = indices[:, [3,2,1,0]]
    indices = np.vstack((indices, ids + 4*nu*nv))
    if exchange:
        vertices = vertices[:, [1,0,2]]
    return (vertices, indices)

def hyperboloidMesh(a, b, c, signature, nu, nv, vmin):
    points, faces = hyperboloidMesh0(a, b, c, signature, nu, nv, vmin)
    nfaces = faces.shape[0]
    fours = np.full((nfaces, 1), 4, dtype=int)
    faces = np.hstack((fours, faces))
    faces = faces.reshape((5*nfaces, ), order="C")
    return pv.PolyData(points, faces).clean()


def curvatureLines(a, b, c, signature, nhlines, nvlines, vmin, npoints = 100):
    nu = int(nhlines/2)
    nv = int(nvlines/2)
    if signature == "+-+":
        clines = curvatureLines(a, c, b, "++-", nhlines, nvlines, vmin, npoints)
        return [cline[:, [0, 2, 1]] for cline in clines]
    elif signature == "-++":
        clines = curvatureLines(c, b, a, "++-", nhlines, nvlines, vmin, npoints)
        return [cline[:, [2, 1, 0]] for cline in clines]
    c0 = c
    if b > a:
        exchange = True
        a0 = b
        b0 = a
    else:
        exchange = False
        a0 = a 
        b0 = b
    Mu2 = a0*a0
    h2ab = Mu2 - b0*b0
    h2ac = c0*c0 + Mu2
    c2 = 1
    a2 = c2 + h2ac
    b2 = a2 - h2ab
    h2bc = b2 - c2
    #  
    v_ = np.linspace(vmin, c2, nv)
    t_ = np.linspace(vmin, c2, npoints)
    #
    if a0 != b0:
        u_ = np.linspace(b2, a2, nu)
        s_ = np.linspace(b2, a2, npoints)
        mx = a0 / sqrt(h2ac*h2ab)
        my = b0 / sqrt(h2bc*h2ab)
        mz = c0 / sqrt(h2bc*h2ac)
        out = [None]*(2*nv + 4*nu - 5)
        for j in range(nv):
            x = mx * np.sqrt((a2-s_)*(a2-v_[j]))
            y = my * np.sqrt((s_-b2)*(b2-v_[j]))
            z = mz * np.sqrt((s_-c2)*(c2-v_[j]))
            idx1 = list(range(npoints-1))
            idx1.reverse()
            M = np.vstack(
                (
                    np.column_stack((x, y, z)), 
                    np.column_stack((-x, y, z))[idx1, :]
                )
            )
            idx = list(range(2*npoints-2))
            idx.reverse()
            M = np.vstack(
                (
                    M, 
                    np.column_stack((M[:,0], -M[:, 1], M[:, 2]))[idx, :]
                )
            )
            out[j] = M
            if j < nv-1:
              out[nv+j] = np.hstack((M[:, [0, 1]], -M[:, [2]]))
        for i in range(nu):
            x = mx * np.sqrt((a2-u_[i])*(a2-t_))
            y = my * np.sqrt((u_[i]-b2)*(b2-t_))
            z = mz * np.sqrt((u_[i]-c2)*(c2-t_))
            M = np.vstack(
                (
                    np.column_stack((x, y, z)), 
                    np.column_stack((x, y, -z))[idx1, :]
                )
            )
            out[2*nv-1+i] = M
            if i < nu-1:
                out[2*nv-1+nu+i] = np.hstack((-M[:, [0]], M[:, [1, 2]]))
                if i > 0:
                    out[2*nv+3*nu+i-4] = np.hstack((-M[:, [0, 1]], M[:, [2]]))
            if i > 0:
                out[2*nv+2*nu+i-3] = np.column_stack((M[:, 0], -M[:, 1], M[:, 2]))
    else: # a0 = b0
        u_ = np.linspace(0, 2*pi, nu+1)[:nu]
        s_ = np.linspace(0, 2*pi, npoints)
        coss_ = np.cos(s_)
        sins_ = np.sin(s_)
        mxy = a0 / sqrt(h2ac)
        mz = c0 / sqrt(h2bc)
        out = [None]*(2*nv + 4*nu - 5)
        for j in range(nv):
            x = mxy * sqrt(a2-v_[j]) * coss_
            y = mxy * sqrt(b2-v_[j]) * sins_
            z = mz * sqrt(c2-v_[j]) 
            M = np.column_stack((x, y, z))
            out[j] = M
            if j < nv-1:
                out[nv+j] = np.hstack((M[:, [0, 1]], -M[:, [2]]))
        r = mxy * np.sqrt(a2-t_)
        z = mz * np.sqrt(c2-t_)
        for i in range(nu):
            x = r * cos(u_[i])
            y = r * sin(u_[i])
            M = np.vstack(
                (
                    np.column_stack((x, y, z)), 
                    np.column_stack((x, y, -z))[idx1, :]
                )
            )
            out[2*nv-1+i] = M
            if i < nu-1:
                out[2*nv-1+nu+i] = np.hstack((-M[:, [0]], M[:, [1, 2]]))
                if i > 0:
                    out[2*nv+3*nu+i-4] = np.hstack((-M[:, [0, 1]], M[:, [2]]))
            if i > 0:
                out[2*nv+2*nu+i-3] = np.column_stack((M[:, 0], -M[:, 1], M[:, 2]))
    if exchange:
        out = [M[:, [1,0,2]] for M in out]
    return out

def hyperboloidOneSheet(
    a, b, c, signature, nu, nv, vmin, nhlines, nvlines, inside_color="gold", outside_color= "green",
    gifname=None, nframes=180, convert="magick convert", delay=8
):
    mesh = hyperboloidMesh(a, b, c, signature, nu, nv, vmin)
    clines = curvatureLines(a, b, c, signature, nhlines, nvlines, vmin)
    cmesh = mesh.copy()
    angle_ = np.linspace(0, 360, nframes+1)[:nframes]
    anim = False
    if gifname is not None:
        anim = True
        gif_sansext, file_extension = os.path.splitext(os.path.basename(gifname))
        screenshotfmt = gif_sansext + "_%03d.png"
        screenshotglob = gif_sansext + "_*.png"
    for i, angle in enumerate(angle_):
        pltr = pv.Plotter(window_size=[512,512], off_screen=anim)
        pltr.set_background("#363940")
        pltr.add_mesh(
            mesh, smooth_shading=True, specular=20, culling="front", color=inside_color
        )
        pltr.add_mesh(
            cmesh, smooth_shading=True, specular=20, culling="back", color=outside_color
        )
        for cline in clines:
            spline = pv.Spline(cline, 1000)
            tube = spline.tube(radius=0.2)
            pltr.add_mesh(tube, smooth_shading=True, color="black", specular=20)
        pltr.camera_position = "yz"
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


hyperboloidOneSheet(
    4, 5, 3, "+-+", 100, 100, -150, 6, 8, inside_color="gold", outside_color= "green",
    gifname=None, nframes=1
)