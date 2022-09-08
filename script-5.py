"""
First pass at simulating suspended waveguide
"""
import clipboard_and_style_sheet
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mt
import materials as mtp
import waveguide_dispersion as wg
import geometry
from pynlo.media.crystals.XTAL_PPLN import Gayer5PctSellmeier
import scipy.constants as sc

clipboard_and_style_sheet.style_sheet()


def get_beta(res, plot=True):
    wl = 1 / res.freq
    omega = res.freq * 2 * np.pi
    conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9
    beta = res.kx.flatten() * 2 * np.pi
    beta1 = np.gradient(beta, omega, edge_order=2)
    beta2 = np.gradient(beta1, omega, edge_order=2) * conversion

    if plot:
        # plt.figure()
        plt.plot(wl, beta2, 'o-')
        plt.axhline(0, color='r')
        plt.axvline(1.55, color='r')
        plt.xlabel("wavelength ($\mathrm{\mu m}$)")
        plt.ylabel("$\mathrm{\\beta_2 \; (ps^2/km})$")

    return wl, omega, beta, beta1, beta2


def plot_mode(res, sim, band_index, k_index):
    wl = 1 / res.freq
    fig, ax = sim.plot_mode(band_index, k_index)
    ax.title.set_text(ax.title.get_text() + "\n" + "$\mathrm{\lambda = }$" +
                      '%.2f' % wl[k_index] + " $\mathrm{\mu m}$")


# %%____________________________________________________________________________________________________________________
# Gayer paper Sellmeier equation for ne (taken from PyNLO
# 1 / omega is in um -> multiply by 1e3 to get to nm -> then square to go from ne to eps
eps_func_wvgd = lambda omega: Gayer5PctSellmeier(24.5).n((1 / omega) * 1e3) ** 2

# %%____________________________________________________________________________________________________________________
sim = wg.ThinFilmWaveguide(etch_width=3,
                           etch_depth=.3,
                           film_thickness=1,  # I'll fix the height at 1 um now
                           substrate_medium=mp.Medium(index=1),
                           waveguide_medium=mt.LiNbO3,
                           resolution=30,
                           num_bands=1,
                           cell_width=10,
                           cell_height=4)

# %%____________________________________________________________________________________________________________________
# # single waveguide parameter (comment out if running the for loop block instead)
# sim.etch_width = 2.5
# sim.etch_depth = 0.6
#
# block_waveguide = sim.blk_wvgd  # save sim.blk_wvgd
# sim.blk_wvgd = geometry.convert_block_to_trapezoid(sim.blk_wvgd)  # set the blk_wvgd to a trapezoid
# res = sim.calc_dispersion(.8, 5, 50, eps_func_wvgd=eps_func_wvgd)  # run simulation
# sim.blk_wvgd = block_waveguide  # reset trapezoid back to blk_wvgd

# wl, omega, beta, beta1, beta2 = get_beta(res)
# plot_mode(res, sim, 0, 0)

# %%____________________________________________________________________________________________________________________
# for loop sweep through parameters
etch_width = wg.get_omega_axis(1 / 5, 1 / 2.5, 5)
etch_depth = wg.get_omega_axis(1 / .6, 1 / .3, 6)
NPTS = len(etch_width) * len(etch_depth)
RES = []

h = 0
for w in etch_width:
    # for h in height:
    for d in etch_depth:
        sim.etch_width = w
        sim.etch_depth = d

        block_waveguide = sim.blk_wvgd  # save sim.blk_wvgd
        sim.blk_wvgd = geometry.convert_block_to_trapezoid(sim.blk_wvgd)  # set the blk_wvgd to a trapezoid
        res = sim.calc_dispersion(.8, 5, 50, eps_func_wvgd=eps_func_wvgd)  # run simulation
        sim.blk_wvgd = block_waveguide  # reset trapezoid back to blk_wvgd

        RES.append(res)

        h += 1
        print(f'______________________________________finished {h} of {NPTS}________________________________')

# %%____________________________________________________________________________________________________________________
params = []
for w in etch_width:
    for d in etch_depth:
        params.append([w, d])
params = np.array(params)

fig, ax = plt.subplots(2, 4, figsize=np.array([11.81, 6.46]))
ax = ax.flatten()
ax[-1].axis(False)
for n in range(len(etch_depth)):
    for dat in RES[n::len(etch_depth)]:
        wl, omega, b, b1, b2 = get_beta(dat, False)
        ax[n].plot(wl, b2, '-')
    ax[n].set_ylim(-1000, 5800)
    ax[n].set_title(f'{np.round(etch_depth[n], 2)}' + " $\mathrm{\mu m}$")
    ax[n].axvline(1.55, color='k', linestyle='--')
    ax[n].axhline(0, color='k', linestyle='--')
    print(params[n::len(etch_depth)])
[i.set_xlabel("wavelength ($\mathrm{\mu m}$)") for i in ax]
[i.set_ylabel("$\mathrm{\\beta_2 \; (ps^2/km})$") for i in ax]
fig.suptitle("fixed depth, sweeping width")

fig, ax = plt.subplots(2, 3, figsize=np.array([8.79, 6.46]))
ax = ax.flatten()
h = 0
for n in range(0, len(params), len(etch_depth)):
    for dat in RES[n: n + len(etch_depth)]:
        wl, omega, b, b1, b2 = get_beta(dat, False)
        ax[h].plot(wl, b2, '-')
    ax[h].set_ylim(-1000, 5800)
    ax[h].set_title(f'{np.round(etch_width[h], 2)}' + " $\mathrm{\mu m}$")
    ax[h].axvline(1.55, color='k', linestyle='--')
    ax[h].axhline(0, color='k', linestyle='--')
    print(params[n: n + len(etch_depth)])
    h += 1
[i.set_xlabel("wavelength ($\mathrm{\mu m}$)") for i in ax]
[i.set_ylabel("$\mathrm{\\beta_2 \; (ps^2/km})$") for i in ax]
fig.suptitle("fixed width, sweeping depth")
