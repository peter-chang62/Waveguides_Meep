"""Calculate the dispersion of a 3D LiNBO3 waveguide, and / or do a simulation of a 3D waveguide with absorbing 
boundaries. 

I've come to the realization that whereas there isn't anything wrong with this code per se, it is an impractically 
large simulation to run! """

import sys

sys.path.append('../')
import meep as mp
import meep.materials as mt
import numpy as np
import clipboard_and_style_sheet
import matplotlib.pyplot as plt
import h5py
import utilities as util
from mayavi import mlab

clipboard_and_style_sheet.style_sheet()

just_vslz_mayavi = False
calculate_dispersion = False


def plot_cross_section(Xlen, Ylen, eps, n=None):
    assert len(eps.shape) > 2, f'eps should be a 3D array but has shape {eps.shape}'
    data = eps[:, :, n].T
    x = np.linspace(0, Xlen, data.shape[1])
    y = np.linspace(0, Ylen, data.shape[0])
    plt.pcolormesh(x, y, data)
    square_figure()


def square_figure():
    plt.gca().set_aspect('equal', adjustable='box')


# %%  try a narrower bandwidth source
wl_min, wl_max = .4, 0.8

# %% Set up the geometry of the problem.
wl_wvgd = 3.5
n_cntr_wl = mt.LiNbO3.epsilon(1 / wl_wvgd)[2, 2]  # X11 and X22 are no, X33 is ne
wdth_wvgd = 0.5 * wl_wvgd / n_cntr_wl  # width of the waveguide is half a wavelength wide
hght_wvgd = 0.5  # height of the waveguide, I recall Tsung-Han said a few hundred nanometers
cntr_wvgd = mp.Vector3(0, 0.25, 0)  # set origin as the center of the waveguide

sy = 5  # size of the cell in the y direction
sx = 5  # size of the cell in the x direction
if calculate_dispersion:  # effectively 2D
    sz = 0
else:  # extend to 3D
    sz = 5

dpml = wl_max  # PML thickness

# %% Create the geometry using the above information
blk1 = mp.Block(
    size=mp.Vector3(wdth_wvgd, hght_wvgd, mp.inf),
    center=cntr_wvgd
)

hght_blk2 = sy / 2 - hght_wvgd / 2 + cntr_wvgd.y
offst_blk2 = hght_blk2 / 2 + hght_wvgd / 2 - cntr_wvgd.y
blk2 = mp.Block(
    size=mp.Vector3(mp.inf, hght_blk2, mp.inf),
    center=mp.Vector3(0, -offst_blk2, 0)
)

cell = mp.Vector3(sx, sy, sz)

ABSX = mp.Absorber(dpml, direction=mp.X)  # left and right
ABSZ = mp.Absorber(dpml, direction=mp.Z)  # front and back
ABSY = mp.Absorber(dpml, direction=mp.Y, side=mp.Low)  # bottom
PMLY = mp.PML(dpml, direction=mp.Y, side=mp.High)  # top

# %% set the dielectric constant of the geometric objects
if just_vslz_mayavi:
    blk1.material = mp.Medium(epsilon=10)
    blk2.material = mp.Medium(epsilon=5)
else:
    blk1.material = mt.LiNbO3
    blk2.material = mt.SiO2

# %% create the geometry and boundary_layers list for the simulation
geometry = [
    blk1,
    blk2
]

if calculate_dispersion:  # periodic in Z
    boundary_layers = [ABSX, PMLY, ABSY]
else:  # otherwise absorb the light at the end
    boundary_layers = [ABSX, PMLY, ABSY, ABSZ]

# %% create the sources list for the simulation

# create a Gaussian time pulse with bandwidth such that you cover all frequencies
# in the bandwidth of frequencies whose dispersion you want to calculate
bw = 1 / wl_max, 1 / wl_min
f_src = float((np.diff(bw) / 2) + bw[0])
df_src = float(np.diff(bw) * 1)
src = mp.GaussianSource(frequency=f_src,
                        fwidth=df_src
                        )

pt_src_offset = mp.Vector3(0, hght_wvgd * 0.25)  # offset in y
pt_src = cntr_wvgd + pt_src_offset

# waveguide oriented longitudinally in Z, width is in X, so polarization is in Y
source = mp.Source(
    src=src,
    component=mp.Ey,
    center=pt_src,
)

Sources = [source]

# %% create the simulation instance
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    sources=Sources,
                    boundary_layers=boundary_layers,
                    resolution=200,  # fields keep blowing up :(
                    )
sim.use_output_directory('sim_output')
if not just_vslz_mayavi:
    if calculate_dispersion:
        # periodicity is only in the z direction, setting this flag offers a performance boost
        sim.special_kz = True

# %% specify relevant symmetries and add them to the simulation
symx = mp.Symmetry(
    direction=mp.X,
    phase=1  # E-field is a true vector
)
symz = mp.Symmetry(
    direction=mp.Z,
    phase=1  # E-field is a true vector
)

sim.symmetries = [symx, symz]

# %%
if just_vslz_mayavi:
    sim.init_sim()
    eps = sim.get_epsilon()

    if sz > 0:
        plot_cross_section(sx, sy, eps, 0)

        plt.axvline(dpml, color='r')
        plt.axvline(sx - dpml, color='r')
        plt.axhline(dpml, color='r')
        plt.axhline(sy - dpml, color='r')
    else:
        sim.plot2D()

# %%
if (not just_vslz_mayavi) and calculate_dispersion:
    kmin, kmax = blk1.material.valid_freq_range
    kpts = mp.interpolate(5, [mp.Vector3(0, 0, kmin), mp.Vector3(0, 0, kmax)])
    kz = np.array([i.z for i in kpts])

    freq = sim.run_k_points(150, kpts)

    # %%
    plt.figure()
    plt.plot([kz.min(), kz.max()], [kz.min(), kz.max()], 'k', label='light line')
    for n in range(len(freq)):
        if len(freq[n]) > 0:
            [plt.plot(kz[n], i.real, marker='o', color='C0') for i in freq[n]]
    plt.legend(loc='best')

# %%
if (not just_vslz_mayavi) and (not calculate_dispersion):
    sim.run(
        mp.to_appended("ey", mp.at_every(0.6, mp.output_efield_y)),
        until_after_sources=300
    )

# %%
# with h5py.File('sim_output/3-scratch-ey.h5', 'r') as f:
#     data = np.array(f[util.get_key(f)])

# %%
# save = False
# fig, ax = plt.subplots(1, 1)
# for n in range(0, data.shape[3], 1):
#     ax.clear()
#     ax.imshow(data[::-1, ::-1, 12, n].T, cmap='jet',
#               # vmax=data.max(),
#               # vmin=data.min()
#               )
#     ax.set_title(n)
#     if save:
#         plt.savefig(f'../fig/{n}.png')
#     else:
#         plt.pause(.1)

# %%
# zero = np.zeros(data.shape[:-1])
# mlab.quiver3d(zero, data[:, :, :, 2], zero)
# mlab.show()
