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

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________________________________________________
# Gayer paper Sellmeier equation for ne (taken from PyNLO
# 1 / omega is in um -> multiply by 1e3 to get to nm -> then square to go from ne to eps
eps_func_wvgd = lambda omega: Gayer5PctSellmeier(24.5).n((1 / omega) * 1e3) ** 2

# %%____________________________________________________________________________________________________________________
sim = wg.ThinFilmWaveguide(etch_width=3,
                           etch_depth=.3,
                           # film_thickness=.7,
                           film_thickness=1,  # I'll fix the height at 1 um now
                           substrate_medium=mtp.Al2O3,
                           waveguide_medium=mt.LiNbO3,
                           resolution=30,
                           num_bands=1,
                           cell_width=10,
                           cell_height=4)

# %%____________________________________________________________________________________________________________________
# individual sampling (comment out if running the for loop block instead)
# sim.etch_width = 5.0
# sim.etch_depth = 0.7
#
# block_waveguide = sim.blk_wvgd  # save sim.blk_wvgd
# sim.blk_wvgd = geometry.convert_block_to_trapezoid(sim.blk_wvgd)  # set the blk_wvgd to a trapezoid
# res = sim.calc_dispersion(.8, 5, 50, eps_func_wvgd=eps_func_wvgd)  # run simulation
# sim.blk_wvgd = block_waveguide  # reset trapezoid back to blk_wvgd
#
# wl = 1 / res.freq
# omega = res.freq * 2 * np.pi
# conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9
# beta = res.kx.flatten() * 2 * np.pi
# beta1 = np.gradient(beta, omega, edge_order=2)
# beta2 = np.gradient(beta1, omega, edge_order=2) * conversion
#
# # plt.figure()
# plt.plot(wl, beta2, 'o-')
# plt.axhline(0, color='r')
# plt.axvline(1.55, color='r')
# plt.xlabel("wavelength ($\mathrm{\mu m}$)")
# plt.ylabel("$\mathrm{\\beta_2 \; (ps^2/km})$")
#
# fig, ax = sim.plot_mode(0, 0)
# ax.title.set_text(ax.title.get_text() + "\n" + "$\mathrm{\lambda = }$" +
#                   '%.2f' % wl[0] + " $\mathrm{\mu m}$")
#
# fig, ax = sim.plot_mode(0, 2)
# ax.title.set_text(ax.title.get_text() + "\n" + "$\mathrm{\lambda = }$" +
#                   '%.2f' % wl[2] + " $\mathrm{\mu m}$")

# %%____________________________________________________________________________________________________________________
etch_width = wg.get_omega_axis(1 / 3, 1 / 0.3, 20)  # 300 nm to 3 um in 135 nm steps
# height = wg.get_omega_axis(1 / 1, 1 / 0.7, 10)  # 700 nm to 1 um in 30 nm steps
# height = np.arange(1.05, 3.05, .05)  # continue parameter sweep: 1050 nm to 3000 nm in 50 nm steps
etch_depth = np.arange(0.1, 1.05, .05)

# round the parameters
etch_width = np.round(etch_width, 3)
etch_depth = np.round(etch_depth, 3)

for w in etch_width:
    # for h in height:
    for d in etch_depth:
        sim.etch_width = w
        # sim.height = h
        sim.etch_depth = d

        block_waveguide = sim.blk_wvgd  # save sim.blk_wvgd
        sim.blk_wvgd = geometry.convert_block_to_trapezoid(sim.blk_wvgd)  # set the blk_wvgd to a trapezoid
        res = sim.calc_dispersion(.4, 5, 100, eps_func_wvgd=eps_func_wvgd)  # run simulation
        sim.blk_wvgd = block_waveguide  # reset trapezoid back to blk_wvgd

        # _________________________________ calculate beta2 __________________________________________________________
        omega = res.freq * 2 * np.pi
        beta = res.kx.flatten() * 2 * np.pi
        beta1 = np.gradient(beta, omega, edge_order=2)
        beta2 = np.gradient(beta1, omega, edge_order=2)

        # _______________________________________ save the data ______________________________________________________
        arr = np.c_[res.freq, beta, beta1, beta2]
        # np.save(f'sim_output/06-16-2022/dispersion-curves/{w}_{h}.npy', arr)
        # np.save(f'sim_output/06-16-2022/E-fields/{w}_{h}.npy', sim.E[:, :, :, :, 1].__abs__() ** 2)
        # np.save(f'sim_output/06-16-2022/eps/{w}_{h}.npy', sim.ms.get_epsilon())

        np.save(f'sim_output/07-19-2022/dispersion-curves/{w}_{d}.npy', arr)
        np.save(f'sim_output/07-19-2022/E-fields/{w}_{d}.npy', sim.E[:, :, :, :, 1].__abs__() ** 2)
        np.save(f'sim_output/07-19-2022/eps/{w}_{d}.npy', sim.ms.get_epsilon())
