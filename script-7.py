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
path = 'sim_output/07-19-2022/'
# path = 'sim_output/09-08-2022/'
names = [i.name for i in os.scandir(path + 'dispersion-curves/')]

w_limit = 1.245
[names.remove(i) for i in names.copy() if width(i) < w_limit]

names = sorted(names, key=depth)
names = sorted(names, key=width)

# %%____________________________________________________________________________________________________________________
get_disp = lambda n: np.load(path + 'dispersion-curves/' + names[n])
get_eps = lambda n: np.load(path + 'eps/' + names[n])
get_field = lambda n: np.squeeze(np.load(path + 'E-fields/' + names[n]))


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
        ax.set_title(f'{np.round(width(names[n]), 3)} x {np.round(depth(names[n]), 3)}' + ' $\mathrm{\mu m}$' '\n' +
                     "$\mathrm{\lambda = }$" + '%.2f' % wl + ' $\mathrm{\mu m}$' + '\n' +
                     '$\mathrm{A_{eff}}$ = %.3f' % mode_area(get_field(n)[k_index]) +
                     ' $\mathrm{\mu m^2}$')
    else:
        plt.title(f'{np.round(width(names[n]), 3)} x {np.round(depth(names[n]), 3)}' + ' $\mathrm{\mu m}$' '\n' +
                  "$\mathrm{\lambda = }$" + '%.2f' % wl + ' $\mathrm{\mu m}$' + '\n' +
                  '$\mathrm{A_{eff}}$ = %.3f' % mode_area(get_field(n)[k_index]) +
                  ' $\mathrm{\mu m^2}$')


def get_bp_ind(wl_grid, wl_ll, wl_ul):
    return np.where(np.logical_and(wl_grid >= wl_ll, wl_grid <= wl_ul), 1, 0)


def e_p_in_window(wl_grid, dv, a_v, wl_ll, wl_ul):
    h = get_bp_ind(wl_grid, wl_ll, wl_ul)
    a_v_filt = a_v * h
    return scint.simps(abs(a_v_filt) ** 2, axis=1, dx=dv)


# %% __________________________________________ RUN THROUGH PYNLO ______________________________________________________
path_save = r"/home/peterchang/SynologyDrive/Research_Projects/Waveguide Simulations/sim_output/10-05-2022/"
for ind in range(19 * 2, len(names)):
    # %% Pulse Properties ____________________________________________________________________________________________
    n_points = 2 ** 13
    v_min = sc.c / ((5000 - 10) * 1e-9)  # sc.c / 5000 nm
    v_max = sc.c / ((400 + 10) * 1e-9)  # sc.c / 400 nm
    v0 = sc.c / 1560e-9  # sc.c / 1550 nm
    e_p = 300e-3 * 1e-9  # 300 mW
    t_fwhm = 50e-15  # 50 fs

    pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
    pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing

    v_grid = pulse.v_grid
    t_grid = pulse.t_grid

    # %% Waveguide properties ________________________________________________________________________________________
    length = 3e-3  # 10 mm
    a_eff = mode_area(get_field(ind)[21]) * 1e-12  # um^2 -> m^2 @ lamda = 1560 nm

    b_data = get_disp(ind)
    b_data_dim = names[ind]
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

    # %% Mode ________________________________________________________________________________________________________
    mode = pynlo.media.Mode(
        v_grid=v_grid,
        beta_v=beta,
        g2_v=g2,
        g2_inv=None,
        g3_v=g3,
        z=0.0
    )

    # %% Model _______________________________________________________________________________________________________
    model = pynlo.model.SM_UPE(pulse, mode)
    local_error = 1e-6
    dz = model.estimate_step_size(n=20, local_error=local_error)

    z_grid = np.linspace(0, length, 100)
    pulse_out, z, a_t, a_v = model.simulate(z_grid, dz=dz, local_error=local_error, n_records=100, plot=None)

    # %% save data ___________________________________________________________________________________________________
    # np.save('sim_output/10-05-2022/time_domain/' + f'{width(names[ind])}_{depth(names[ind])}.npy', a_t)
    # np.save('sim_output/10-05-2022/frequency_domain/' + f'{width(names[ind])}_{depth(names[ind])}.npy', a_v)

    np.save(path_save + 'time_domain/' + f'{width(names[ind])}_{depth(names[ind])}.npy', a_t)
    np.save(path_save + 'frequency_domain/' + f'{width(names[ind])}_{depth(names[ind])}.npy', a_v)

# np.save('sim_output/10-05-2022/v_grid.npy', v_grid)
# np.save('sim_output/10-05-2022/t_grid.npy', t_grid)
# np.save('sim_output/10-05-2022/z.npy', z)

np.save(path_save + 'v_grid.npy', v_grid)
np.save(path_save + 't_grid.npy', t_grid)
np.save(path_save + 'z.npy', z)
# path_save = r"~/SynologyDrive/Research_Projects/Waveguide Simulations/sim_output/10-05-2022/"
# for ind in range(19 * 2, len(names)):
#     # %% Pulse Properties ____________________________________________________________________________________________
#     n_points = 2 ** 13
#     v_min = sc.c / ((5000 - 10) * 1e-9)  # sc.c / 5000 nm
#     v_max = sc.c / ((400 + 10) * 1e-9)  # sc.c / 400 nm
#     v0 = sc.c / 1560e-9  # sc.c / 1550 nm
#     e_p = 300e-3 * 1e-9  # 300 mW
#     t_fwhm = 50e-15  # 50 fs
#
#     pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
#     pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing
#
#     v_grid = pulse.v_grid
#     t_grid = pulse.t_grid
#
#     # %% Waveguide properties ________________________________________________________________________________________
#     length = 3e-3  # 10 mm
#     a_eff = mode_area(get_field(ind)[21]) * 1e-12  # um^2 -> m^2 @ lamda = 1560 nm
#
#     b_data = get_disp(ind)
#     b_data_dim = names[ind]
#     wl, b = 1 / b_data[:, 0], b_data[:, 1]
#     k = b * 1e6 / (2 * np.pi)  # 1/m
#     nu = sc.c / (wl * 1e-6)
#     n = sc.c * k / nu
#     n_wvgd = interp1d(nu, n, kind='cubic', bounds_error=True)
#
#     n_eff = n_wvgd(v_grid)
#     beta = n_eff * 2 * np.pi * v_grid / sc.c  # n * w / sc.c
#
#     # 2nd order nonlinearity
#     d_eff = 27e-12  # 27 pm / V
#     chi2_eff = 2 * d_eff
#     g2 = utils.chi2.g2_shg(v0, v_grid, n_eff, a_eff, chi2_eff)
#
#     # 3rd order nonlinearity
#     chi3_eff = 5200e-24
#     g3 = utils.chi3.g3_spm(n_eff, a_eff, chi3_eff)
#
#     # %% Mode ________________________________________________________________________________________________________
#     mode = pynlo.media.Mode(
#         v_grid=v_grid,
#         beta_v=beta,
#         g2_v=g2,
#         g2_inv=None,
#         g3_v=g3,
#         z=0.0
#     )
#
#     # %% Model _______________________________________________________________________________________________________
#     model = pynlo.model.SM_UPE(pulse, mode)
#     local_error = 1e-6
#     dz = model.estimate_step_size(n=20, local_error=local_error)
#
#     z_grid = np.linspace(0, length, 100)
#     pulse_out, z, a_t, a_v = model.simulate(z_grid, dz=dz, local_error=local_error, n_records=100, plot=None)
#
#     # %% save data ___________________________________________________________________________________________________
#     # np.save('sim_output/10-05-2022/time_domain/' + f'{width(names[ind])}_{depth(names[ind])}.npy', a_t)
#     # np.save('sim_output/10-05-2022/frequency_domain/' + f'{width(names[ind])}_{depth(names[ind])}.npy', a_v)
#
#     np.save(path_save + 'time_domain/' + f'{width(names[ind])}_{depth(names[ind])}.npy', a_t)
#     np.save(path_save + 'frequency_domain/' + f'{width(names[ind])}_{depth(names[ind])}.npy', a_v)
#
# # np.save('sim_output/10-05-2022/v_grid.npy', v_grid)
# # np.save('sim_output/10-05-2022/t_grid.npy', t_grid)
# # np.save('sim_output/10-05-2022/z.npy', z)
#
# np.save(path_save + 'v_grid.npy', v_grid)
# np.save(path_save + 't_grid.npy', t_grid)
# np.save(path_save + 'z.npy', z)

# %% ________________________________________ Plotting PYNLO SIMULATION RESULTS ________________________________________
path_ = 'sim_output/10-05-2022/'
names_ = [i.name for i in os.scandir('sim_output/10-05-2022/frequency_domain/')]

names_.sort(key=depth)
names_.sort(key=width)

z = np.load(path_ + 'z.npy') * 1e3
z_ind = np.argmin(abs(z - 1))
v_grid = np.load(path_ + 'v_grid.npy') * 1e-12
wl_grid = sc.c * 1e6 * 1e-12 / v_grid

fig, ax = plt.subplots(2, 2, figsize=np.array([13.69, 4.8 * 2]))
ax = ax.flatten()
save = False
for i in range(len(names_)):
    data = np.load(path_ + 'frequency_domain/' + names_[i])
    data = abs(data) ** 2
    data /= data.max()

    [i.clear() for i in ax]

    ax[0].pcolormesh(wl_grid, z, data)
    ax[0].set_xlabel("wavelength ($\mathrm{\mu m}$)")

    ax[1].plot(wl_grid, 10 * np.log10(data[z_ind] / data[-1].max()))
    ax[1].set_xlabel("wavelength ($\mathrm{\mu m}$)")

    plot_mode(i + 19 * 2, 3, False, ax[2])
    assert names[i + 19 * 2] == names_[i]
    freq, b, b1, b2 = get_disp(i).T
    wl = 1 / freq
    b2 *= conversion
    ax[3].plot(wl, b2, 'o-')
    ax[3].axhline(0, linestyle='--', color='k')
    ax[3].set_ylim(-1000, 5500)

    if save:
        plt.savefig(f'fig/{i}.png')
    else:
        plt.pause(.01)

# %% ________________________________________ Analyzing PyNLO simulation results _______________________________________
# path_ = 'sim_output/10-05-2022/'
# names_ = [i.name for i in os.scandir('sim_output/10-05-2022/frequency_domain/')]
#
# names_.sort(key=depth)
# names_.sort(key=width)
#
# z = np.load(path_ + 'z.npy') * 1e3
# ind_lim = np.argmin(abs(z - 1))
# v_grid = np.load(path_ + 'v_grid.npy')
# t_grid = np.load(path_ + 't_grid.npy')
# wl_grid = sc.c / v_grid
# dv = np.diff(v_grid)[0]
# dt = np.diff(t_grid)[0]
# frep = 1e9
#
# POWER = np.zeros((len(names_), len(z)))
# for n in range(len(POWER)):
#     a_v = np.load(path_ + "frequency_domain/" + names_[n])
#     a_t = np.load(path_ + "time_domain/" + names_[n])
#     POWER[n] = e_p_in_window(wl_grid, dv, a_v, 3e-6, 5e-6) * frep * 1e3  # mW
#
# # the most power
# max_power = np.max(POWER, axis=1)
# ind_best = np.argmax(max_power)
# a_v_best = np.load(path_ + "frequency_domain/" + names_[ind_best])
# a_t_best = np.load(path_ + "time_domain/" + names_[ind_best])
#
# # fig, ax = plt.subplots(1, 1)
# # for i in abs(a_v_best) ** 2:
# #     ax.clear()
# #     ax.plot(wl_grid * 1e6, i)
# #     plt.pause(.2)
