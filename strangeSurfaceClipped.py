import numpy as np
import pyvista as pv
import os

def f(ρ, θ, ϕ, A, B):
    x = ρ * np.cos(θ) * np.sin(ϕ)
    y = ρ * np.sin(θ) * np.sin(ϕ)
    z = ρ * np.cos(ϕ)    
    return (
        z**4 * B**2
        + 4 * x * y**2 * A * B**2
        + x * z**2 * A * B**2
        - 2 * z**4 * A
        - 4 * x * y**2 * B**2
        - x * z**2 * B**2
        + 3 * z**2 * A * B**2
        - 2 * z**4
        - x * A * B**2
        - 2 * z**2 * A
        + x * B**2
        + A * B**2
        + 2 * z**2
        - B**2
    )

def sph2cart(sph):
    ρ = sph[:,0]
    θ = sph[:,1]
    ϕ = sph[:,2]
    return [
        ρ * np.cos(θ) * np.sin(ϕ),
        ρ * np.sin(θ) * np.sin(ϕ),
        ρ * np.cos(ϕ)
    ]

# generate data grid for computing the values
Rho, Theta, Phi = np.mgrid[0:np.sqrt(3):180j, 0:(np.pi):180j, 0:2*np.pi:180j]

# create a structured grid
grid = pv.StructuredGrid(Rho, Theta, Phi)

nframes = 30
t_ = np.linspace(2.5, 3.5, nframes)

for i, t in enumerate(t_):
    pngname = "zzpic%03d.png" % i
    A = np.cos(t*np.pi/4)
    B = np.sin(t*np.pi/4)
    values = f(Rho, Theta, Phi, A, B)
    grid.point_data['values'] = values.ravel(order='F')  # also the active scalars
    # compute one isosurface
    isosurf = grid.contour(isosurfaces=[0])
    mesh = isosurf.extract_geometry()
    mesh.points = np.transpose(sph2cart(mesh.points))
    # plot it interactively 
    pltr = pv.Plotter(window_size=[512, 512], off_screen=True)
    pltr.background_color = "#363940"
    pltr.set_focus(mesh.center)
    pltr.set_position((13, 12, 18))
    pltr.camera.zoom(2.5)
    pltr.add_mesh(
        mesh,
        smooth_shading=True,
        specular=0.3,
        color="#ff00ff"
    )
    pltr.show(screenshot=pngname)

os.system(
    "magick convert -delay 1x10 -duplicate 1,-2-1 zzpic*.png strangeSurface.gif"    
)
