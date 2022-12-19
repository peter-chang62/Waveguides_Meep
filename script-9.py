"""Combine MEEP simulations and PyNLO into one script (will only run on a linux computer). This is good for sort of
point sampling of the big simulation outputs. You can make small adjustment to parameters here """

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
import scipy.integrate as scint
from scipy.interpolate import interp1d
import pynlo_connor as pynlo
from pynlo_connor import utility as utils

clipboard_and_style_sheet.style_sheet()

resolution = 30


def get_bp_ind(wl_grid, wl_ll, wl_ul):
    return np.where(np.logical_and(wl_grid >= wl_ll, wl_grid <= wl_ul), 1, 0)


def e_p_in_window(wl_grid, dv, a_v, wl_ll, wl_ul):
    h = get_bp_ind(wl_grid, wl_ll, wl_ul)
    a_v_filt = a_v * h
    return scint.simps(abs(a_v_filt) ** 2, axis=1, dx=dv)


def instantiate_pulse(n_points, v_min, v_max, e_p=300e-3 * 1e-9, t_fwhm=50e-15):
    v0 = sc.c / 1560e-9  # sc.c / 1550 nm
    pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
    pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing
    return pulse


def mode_area(I):
    # integral(I * dA) ** 2  / integral(I ** 2 * dA) is the common definition that is used
    # reference: https://www.rp-photonics.com/effective_mode_area.html
    # this gives an overall dimension of dA in the numerator
    area = scint.simpson(scint.simpson(I)) ** 2 / scint.simpson(scint.simpson(I ** 2))
    area /= resolution ** 2
    return area


def load_waveguide(pulse, res, sim):
    pulse: pynlo.light.Pulse
    v_grid = pulse.v_grid
    v0 = pulse.v0

    # a_eff = mode_area(get_field(n)[21]) * 1e-12  # um^2 -> m^2 @ lamda = 1560 nm

    wl, b = 1 / res.freq, res.kx.flatten * 2 * np.pi
    k = b * 1e6 / (2 * np.pi)  # 1/m
    nu = sc.c / (wl * 1e-6)
    n = sc.c * k / nu
    n_wvgd = interp1d(nu, n, kind='cubic', bounds_error=True)

    ind = np.argmin(abs(wl - 1.55))
    field = sim.E[ind][0][:, :, 1].__abs__() ** 2  # mp.Ey = 1
    a_eff = mode_area(field) * 1e-12  # um^2 -> m^2 @ lamda = 1560 nm

    n_eff = n_wvgd(v_grid)
    beta = n_eff * 2 * np.pi * v_grid / sc.c  # n * w / sc.c

    # 2nd order nonlinearity
    d_eff = 27e-12  # 27 pm / V
    chi2_eff = 2 * d_eff
    g2 = utils.chi2.g2_shg(v0, v_grid, n_eff, a_eff, chi2_eff)

    # 3rd order nonlinearity
    chi3_eff = 5200e-24
    g3 = utils.chi3.g3_spm(n_eff, a_eff, chi3_eff)

    mode = pynlo.media.Mode(
        v_grid=v_grid,
        beta_v=beta,
        g2_v=g2,
        g2_inv=None,
        g3_v=g3,
        z=0.0
    )
    return mode


def simulate(pulse, mode, length=3e-3, npts=100):
    model = pynlo.model.SM_UPE(pulse, mode)
    local_error = 1e-6
    dz = model.estimate_step_size(n=20, local_error=local_error)

    z_grid = np.linspace(0, length, npts)
    pulse_out, z, a_t, a_v = model.simulate(z_grid, dz=dz, local_error=local_error, n_records=100, plot=None)
    return pulse_out, z, a_t, a_v


def aliasing_av(a_v):
    x = abs(a_v) ** 2
    x /= x.max()
    x_v_min = x[:, 0]
    x_v_max = x[:, -1]

    ind_v_min_aliasing = np.where(x_v_min > 1e-3)[0]
    ind_v_max_aliasing = np.where(x_v_max > 1e-3)[0]
    List = []
    side = []
    if len(ind_v_min_aliasing) > 0:
        List.append(ind_v_min_aliasing[0])
        side.append("long")
    if len(ind_v_max_aliasing) > 0:
        List.append(ind_v_max_aliasing[0])
        side.append("short")
    if len(List) > 0:
        ind = np.argmin(List)
        return List[ind], side[ind]
    else:
        return False


"""Copy from script-2.py"""
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
                           resolution=resolution,
                           num_bands=1,
                           cell_width=10,
                           cell_height=4)

# %%____________________________________________________________________________________________________________________
# individual sampling (comment out if running the for loop block instead)
sim.etch_width = 2.055
sim.etch_depth = 0.65

block_waveguide = sim.blk_wvgd  # save sim.blk_wvgd
sim.blk_wvgd = geometry.convert_block_to_trapezoid(sim.blk_wvgd)  # set the blk_wvgd to a trapezoid
res = sim.calc_dispersion(.8, 5, 50, eps_func_wvgd=eps_func_wvgd)  # run simulation
sim.blk_wvgd = block_waveguide  # reset trapezoid back to blk_wvgd

wl = 1 / res.freq
omega = res.freq * 2 * np.pi
conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9
beta = res.kx.flatten() * 2 * np.pi
beta1 = np.gradient(beta, omega, edge_order=2)
beta2 = np.gradient(beta1, omega, edge_order=2) * conversion

# plt.figure()
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

n_points = 2 ** 13
v_min = sc.c / ((5000 - 10) * 1e-9)  # sc.c / 5000 nm
v_max = sc.c / ((400 + 10) * 1e-9)  # sc.c / 400 nm
e_p = 300e-3 * 1e-9
t_fwhm = 50e-15
pulse = instantiate_pulse(n_points=n_points,
                          v_min=v_min,
                          v_max=v_max,
                          e_p=e_p,
                          t_fwhm=t_fwhm)
mode = load_waveguide(pulse, res, sim)
pulse_out, z, a_t, a_v = simulate(pulse, mode, length=10e-3)
wl = sc.c / pulse.v_grid
ind_alias = aliasing_av(a_v)


def video():
    fig, ax = plt.subplots(1, 2)
    for n, i in enumerate(abs(a_v) ** 2):
        [i.clear() for i in ax]
        ax[0].semilogy(wl * 1e6, i)
        ax[0].axvline(4.07, color='r')
        ax[1].plot(pulse.t_grid * 1e12, abs(a_t[n]) ** 2)
        # ax[1].set_xlim(-.5, .5)
        plt.pause(.1)
