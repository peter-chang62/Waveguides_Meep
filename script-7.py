"""sim data analysis the arrays were saved as np.c_[res.freq, beta, beta1, beta2] """

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
                     ' $\mathrm{\mu m^2}$' + '\n' + s)
    else:
        plt.title(f'{np.round(width(names[n]), 3)} x {np.round(depth(names[n]), 3)}' + ' $\mathrm{\mu m}$' '\n' +
                  "$\mathrm{\lambda = }$" + '%.2f' % wl + ' $\mathrm{\mu m}$' + '\n' +
                  '$\mathrm{A_{eff}}$ = %.3f' % mode_area(get_field(n)[k_index]) +
                  ' $\mathrm{\mu m^2}$' + '\n' + s)


for ind in range(19 * 2, len(names)):
    # %% Pulse Properties ______________________________________________________________________________________________
    n_points = 2 ** 13
    v_min = sc.c / 4500e-9  # sc.c / 4500 nm
    v_max = sc.c / 800e-9  # sc.c / 815 nm
    v0 = sc.c / 1560e-9  # sc.c / 1550 nm
    e_p = 300e-3 * 1e-9  # 300 mW
    t_fwhm = 50e-15  # 50 fs

    pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
    pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing

    v_grid = pulse.v_grid
    t_grid = pulse.t_grid

    # %% Waveguide properties __________________________________________________________________________________________
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

    # %% Mode __________________________________________________________________________________________________________
    mode = pynlo.media.Mode(
        v_grid=v_grid,
        beta_v=beta,
        g2_v=g2,
        g2_inv=None,
        g3_v=g3,
        z=0.0
    )

    # %% Model _________________________________________________________________________________________________________
    model = pynlo.model.SM_UPE(pulse, mode)
    local_error = 1e-6
    dz = model.estimate_step_size(n=20, local_error=local_error)

    z_grid = np.linspace(0, length, 100)
    pulse_out, z, a_t, a_v = model.simulate(z_grid, dz=dz, local_error=local_error, n_records=100, plot=None)

    # %% save data _____________________________________________________________________________________________________
    arr = np.c_[z, a_t, a_v]
    np.save('sim_output/10-05-2022/' + f'{width(names[ind])}_{depth(names[ind])}.npy', arr)

    # Plotting _________________________________________________________________________________________________________
    # fig = plt.figure("Simulation Results", clear=True)
    # ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1)
    # ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
    # ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, sharex=ax0)
    # ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2, sharex=ax1)
    #
    # p_v_dB = 10 * np.log10(np.abs(a_v) ** 2)
    # p_v_dB -= p_v_dB.max()
    # ax0.plot(1e-12 * v_grid, p_v_dB[0], color="b")
    # ax0.plot(1e-12 * v_grid, p_v_dB[-1], color="g")
    # ax2.pcolormesh(1e-12 * v_grid, 1e3 * z, p_v_dB, vmin=-40.0, vmax=0, shading="auto")
    # ax0.set_ylim(bottom=-50, top=10)
    # ax2.set_xlabel('Frequency (THz)')
    #
    # p_t_dB = 10 * np.log10(np.abs(a_t) ** 2)
    # p_t_dB -= p_t_dB.max()
    # ax1.plot(1e12 * t_grid, p_t_dB[0], color="b")
    # ax1.plot(1e12 * t_grid, p_t_dB[-1], color="g")
    # ax3.pcolormesh(1e12 * t_grid, 1e3 * z, p_t_dB, vmin=-40.0, vmax=0, shading="auto")
    # ax1.set_ylim(bottom=-50, top=10)
    # ax3.set_xlabel('Time (ps)')
    #
    # ax0.set_ylabel('Power (dB)')
    # ax2.set_ylabel('Propagation Distance (mm)')
    # fig.tight_layout()
    # fig.show()
