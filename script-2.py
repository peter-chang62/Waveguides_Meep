"""For the 06-16-2022 simulations, I varied the etch width and the waveguide
height (I believe the etch depth was fixed at 300 nm). In a meeting, people
pointed out it was more realistic to fix the height and vary the etch depth,
and that is what is saved in 07-19-2022.

The simulations for 06-16-2022 occured in two runs, where I continued the
parameter sweep for the waveguide height. The simulations for fixed waveguide
height also occured in two runs. However, it looks like one because the
second time I extended the wavelength axis and just overwrote the previous
simulation data. """

import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import meep as mp
import meep.materials as mt
import materials as mtp
import waveguide_dispersion as wg
import os
import geometry
from pynlo.media.crystals.XTAL_PPLN import Gayer5PctSellmeier
import scipy.constants as sc
import itertools

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________

# Gayer paper Sellmeier equation for ne (taken from PyNLO
# 1 / omega is in um -> multiply by 1e3 to get to nm -> then square to go
# from ne to eps
eps_func_wvgd = lambda omega: Gayer5PctSellmeier(24.5).n((1 / omega) * 1e3) ** 2

# %%____________________________________________________________________________
sim = wg.ThinFilmWaveguide(etch_width=3,  # will be changed later
                           etch_depth=.3,  # will be changed later
                           film_thickness=1,  # I'll fix the height at 1 um now
                           substrate_medium=mtp.Al2O3,
                           waveguide_medium=mt.LiNbO3,
                           resolution=30,
                           num_bands=1,
                           cell_width=10,
                           cell_height=4)

# %%____________________________________________________________________________
# individual sampling (comment out if running the for loop block instead)
sim.etch_width, sim.etch_depth = (1.38, 0.65)

block_waveguide = sim.blk_wvgd  # save sim.blk_wvgd
# set the blk_wvgd to a trapezoid
sim.blk_wvgd = geometry.convert_block_to_trapezoid(sim.blk_wvgd)
# run simulation
res = sim.calc_dispersion(.4, 5, 100, eps_func_wvgd=eps_func_wvgd)
sim.blk_wvgd = block_waveguide  # reset trapezoid back to blk_wvgd

wl = 1 / res.freq
omega = res.freq * 2 * np.pi
conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9
beta = res.kx.flatten() * 2 * np.pi
beta1 = np.gradient(beta, omega, edge_order=2)
beta2 = np.gradient(beta1, omega, edge_order=2)

# plotting
plt.plot(wl, beta2, 'o-')
plt.axhline(0, color='r')
plt.axvline(1.55, color='r')
plt.xlabel("wavelength ($\mathrm{\mu m}$)")
plt.ylabel("$\mathrm{\\beta_2 \; (ps^2/km})$")

fig, ax = sim.plot_mode(0, 0)
ax.title.set_text(ax.title.get_text() + "\n" + "$\mathrm{\lambda = }$" +
                  '%.2f' % wl[0] + " $\mathrm{\mu m}$")

fig, ax = sim.plot_mode(0, 2)
ax.title.set_text(ax.title.get_text() + "\n" + "$\mathrm{\lambda = }$" +
                  '%.2f' % wl[2] + " $\mathrm{\mu m}$")

# saving
arr = np.c_[res.freq, beta, beta1, beta2]
path = r"/Users/peterchang/SynologyDrive/Research_Projects/Waveguide " \
       r"Simulations/sim_output/ "
np.save(path + f'07-19-2022/dispersion-curves/{sim.etch_width}_'
               f'{sim.etch_depth}.npy', arr)  # save to synology
np.save(path + f'07-19-2022/E-fields/{sim.etch_width}_{sim.etch_depth}.npy',
        sim.E[:, :, :, :, 1].__abs__() ** 2)
np.save(path + f'07-19-2022/eps/{sim.etch_width}_{sim.etch_depth}.npy',
        sim.ms.get_epsilon())

# %%____________________________________________________________________________
# # 300 nm to 3 um in 135 nm steps
# etch_width = wg.get_omega_axis(1 / 3, 1 / 0.3, 20)
# etch_depth = np.arange(0.1, 1.05, .05)
# etch_width = np.round(etch_width, 3)  # round the etch width
# etch_depth = np.round(etch_depth, 3)  # round the etch depth
# params = np.asarray(list(itertools.product(etch_width, etch_depth)))

# ported from script-7.py
etch_width = np.array(
    [1.245, 1.245, 1.245, 1.245, 1.245, 1.245, 1.38, 1.38, 1.38,
     1.38, 1.38, 1.38, 1.515, 1.515, 1.515, 1.515, 1.515, 1.515,
     1.515, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65, 1.65,
     1.785, 1.785, 1.785, 1.785, 1.785, 1.785, 1.785, 1.785, 1.92,
     1.92, 1.92, 1.92, 2.055, 2.055, 2.055, 2.19, 2.19])
etch_depth = np.array(
    [0.75, 0.8, 0.85, 0.9, 0.95, 1., 0.7, 0.75, 0.8, 0.85, 0.9,
     0.95, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.45, 0.5, 0.55,
     0.6, 0.65, 0.7, 0.75, 0.8, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
     0.7, 0.75, 0.35, 0.4, 0.45, 0.6, 0.35, 0.4, 0.45, 0.35, 0.4])
params = np.c_[etch_width, etch_depth]

path = r"/Users/peterchang/SynologyDrive/Research_Projects/Waveguide " \
       r"Simulations/sim_output/ "
center = len(params) // 2  # use to launch separate consoles
for w, d in params[:center]:
    sim.etch_width = w
    sim.etch_depth = d

    block_waveguide = sim.blk_wvgd  # save sim.blk_wvgd
    sim.blk_wvgd = geometry.convert_block_to_trapezoid(
        sim.blk_wvgd)  # set the blk_wvgd to a trapezoid
    res = sim.calc_dispersion(.4, 5, 100,
                              eps_func_wvgd=eps_func_wvgd)  # run simulation
    sim.blk_wvgd = block_waveguide  # reset trapezoid back to blk_wvgd

    # _________________________________ calculate beta2 ______________________
    omega = res.freq * 2 * np.pi
    beta = res.kx.flatten() * 2 * np.pi
    beta1 = np.gradient(beta, omega, edge_order=2)
    beta2 = np.gradient(beta1, omega, edge_order=2)

    # _______________________________________ save the data __________________
    arr = np.c_[res.freq, beta, beta1, beta2]

    np.save(path + f'07-19-2022/dispersion-curves/{w}_{d}.npy',
            arr)  # same but push to synology
    np.save(path + f'07-19-2022/E-fields/{w}_{d}.npy',
            sim.E[:, :, :, :, 1].__abs__() ** 2)
    np.save(path + f'07-19-2022/eps/{w}_{d}.npy', sim.ms.get_epsilon())
