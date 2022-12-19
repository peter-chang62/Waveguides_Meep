"""sim data analysis the arrays were saved as np.c_[res.freq, beta, beta1, beta2]

This script runs all the dispersion curves through PyNLO"""

import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.constants as sc
import scipy.integrate as scint
import os
from scipy.interpolate import interp1d
import pynlo_connor as pynlo
from pynlo_connor import utility as utils

try:
    import materials as mtp

    on_linux = True
except:
    Al2O3 = np.load('convenience/freq_epsilon_data.npy')
    Al2O3 = InterpolatedUnivariateSpline(Al2O3[:, 0], Al2O3[:, 1])
    on_linux = False

clipboard_and_style_sheet.style_sheet()


def width(s):
    return float(s.split('_')[0])


def depth(s):
    return float(s.split('_')[1].split('.npy')[0])


def mode_area(I):
    # integral(I * dA) ** 2  / integral(I ** 2 * dA) is the common definition that is used
    # reference: https://www.rp-photonics.com/effective_mode_area.html
    # this gives an overall dimension of dA in the numerator
    area = scint.simpson(scint.simpson(I)) ** 2 / scint.simpson(scint.simpson(I ** 2))
    area /= resolution ** 2
    return area


# %%____________________________________________________________________________________________________________________
resolution = 30
conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9  # already multiplied for some sim data (should be obvious)
path_wvgd = 'sim_output/07-19-2022/'
# path_wvgd = 'sim_output/09-08-2022/'
names_wvgd = [i.name for i in os.scandir(path_wvgd + 'dispersion-curves/')]

width_limit = 1.245
[names_wvgd.remove(i) for i in names_wvgd.copy() if width(i) < width_limit]

names_wvgd = sorted(names_wvgd, key=depth)
names_wvgd = sorted(names_wvgd, key=width)

# to match indexing of pynlo simulations
# names_wvgd = names_wvgd[38:]

# %%____________________________________________________________________________________________________________________
get_disp = lambda n: np.load(path_wvgd + 'dispersion-curves/' + names_wvgd[n])
get_eps = lambda n: np.load(path_wvgd + 'eps/' + names_wvgd[n])
get_field = lambda n: np.squeeze(np.load(path_wvgd + 'E-fields/' + names_wvgd[n]))


# %%____________________________________________________________________________________________________________________
def plot_eps(n, ax=None):
    if ax is not None:
        ax.imshow(get_eps(n)[::-1, ::-1].T, interpolation='spline36', cmap='binary')
    else:
        plt.imshow(get_eps(n)[::-1, ::-1].T, interpolation='spline36', cmap='binary')


def plot_field(n, k_index, alpha=0.9, ax=None):
    if ax is not None:
        ax.imshow(get_field(n)[k_index][::-1, ::-1].T, cmap='RdBu', alpha=alpha)
    else:
        plt.imshow(get_field(n)[k_index][::-1, ::-1].T, cmap='RdBu', alpha=alpha)


def is_guided(kx, freq):
    # __________________________________________________________________________________________________________________
    # bulk propagation is omega = c * k / n
    # to be guided we need omega < omega_bulk
    # __________________________________________________________________________________________________________________

    index_substrate = 1
    freq_substrate = kx / index_substrate
    print(freq, freq_substrate, freq - freq_substrate)
    return freq < freq_substrate


def plot_mode(n, k_index, new_figure=True, ax=None):
    assert not np.all([new_figure, ax is not None])
    if new_figure:
        plt.figure()
    plot_eps(n, ax)
    plot_field(n, k_index, 0.9, ax)
    data = get_disp(n)  # np.c_[res.freq, beta, beta1, beta2]
    freq = data[:, 0]
    kx = data[:, 1] / (2 * np.pi)  # kx = beta / (2 pi)
    guided = is_guided(kx[k_index], freq[k_index])
    wl = 1 / freq[k_index]
    if guided:
        s = "guided"
    else:
        s = "NOT guided"
    if ax is not None:
        ax.set_title(
            f'{np.round(width(names_wvgd[n]), 3)} x {np.round(depth(names_wvgd[n]), 3)}' + ' $\mathrm{\mu m}$' '\n' +
            "$\mathrm{\lambda = }$" + '%.2f' % wl + ' $\mathrm{\mu m}$' + '\n' +
            '$\mathrm{A_{eff}}$ = %.3f' % mode_area(get_field(n)[k_index]) +
            ' $\mathrm{\mu m^2}$')
    else:
        plt.title(
            f'{np.round(width(names_wvgd[n]), 3)} x {np.round(depth(names_wvgd[n]), 3)}' + ' $\mathrm{\mu m}$' '\n' +
            "$\mathrm{\lambda = }$" + '%.2f' % wl + ' $\mathrm{\mu m}$' + '\n' +
            '$\mathrm{A_{eff}}$ = %.3f' % mode_area(get_field(n)[k_index]) +
            ' $\mathrm{\mu m^2}$')


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


def load_waveguide(pulse, n):
    pulse: pynlo.light.Pulse
    v_grid = pulse.v_grid
    v0 = pulse.v0

    a_eff = mode_area(get_field(n)[21]) * 1e-12  # um^2 -> m^2 @ lamda = 1560 nm

    b_data = get_disp(n)
    wl, b = 1 / b_data[:, 0], b_data[:, 1]
    k = b * 1e6 / (2 * np.pi)  # 1/m
    nu = sc.c / (wl * 1e-6)
    n = sc.c * k / nu
    n_wvgd = interp1d(nu, n, kind='cubic', bounds_error=True)

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


def Omega(beta_spl, gamma, Pp, wl, wl_p=1550e-9):
    beta_spl: InterpolatedUnivariateSpline

    w = (sc.c / wl) * 2 * np.pi
    wp = (sc.c / wl_p) * 2 * np.pi

    dk = beta_spl(w) - beta_spl(wp) - beta_spl.derivative(n=1)(w - wp) - gamma * Pp

# n_points = 2 ** 13
# v_min = sc.c / ((5000 - 10) * 1e-9)  # sc.c / 5000 nm
# v_max = sc.c / ((400 + 10) * 1e-9)  # sc.c / 400 nm
# e_p = 300e-3 * 1e-9
# t_fwhm = 50e-15
# pulse = instantiate_pulse(n_points=n_points,
#                           v_min=v_min,
#                           v_max=v_max,
#                           e_p=e_p,
#                           t_fwhm=t_fwhm)
# mode = load_waveguide(pulse, 103)
# pulse_out, z, a_t, a_v = simulate(pulse, mode, length=10e-3)
# wl = sc.c / pulse.v_grid
# ind_alias = aliasing_av(a_v)
#
#
# def video():
#     fig, ax = plt.subplots(1, 2)
#     for n, i in enumerate(abs(a_v) ** 2):
#         [i.clear() for i in ax]
#         ax[0].semilogy(wl * 1e6, i)
#         ax[0].axvline(4.07, color='r')
#         ax[1].plot(pulse.t_grid * 1e12, abs(a_t[n]) ** 2)
#         # ax[1].set_xlim(-.5, .5)
#         plt.pause(.1)
