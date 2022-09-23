"""sim data analysis the arrays were saved as np.c_[res.freq, beta, beta1, beta2] """

import numpy as np
import matplotlib.pyplot as plt
import os
from pynlo.media.crystals.XTAL_PPLN import Gayer5PctSellmeier
import clipboard_and_style_sheet
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.constants as sc
import scipy.integrate as scint
from numpy import ma
import os

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
# path = 'sim_output/07-19-2022/'
path = 'sim_output/09-08-2022/'
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


# %%____________________________________________________________________________________________________________________
# fig, ax = plt.subplots(2, 4)
# ax = ax.flatten()
# ax[-1].axis(False)
# for n in range(7):
#     ind = np.arange(len(names))[n::7]
#     for m in ind:
#         wl = 1 / get_disp(m)[:, 0]
#         beta2 = get_disp(m)[:, 3]
#         ax[n].plot(wl, beta2)
#         print(names[m])

# %%____________________________________________________________________________________________________________________
fig, ax = plt.subplots(1, 3, figsize=np.array([12.88, 4.5]))
save = False
# save = True
for n in range(len(names)):
    [i.clear() for i in ax]
    ax[0].set_xlabel("wavelength $\mathrm{\mu m}$")
    ax[0].set_ylabel("$\mathrm{\\beta_2 \; (ps^2/km})$")
    wl = 1 / get_disp(n)[:, 0]

    # beta2 = get_disp(n)[:, 3] * conversion
    beta2 = get_disp(n)[:, 3]

    ax[0].plot(wl, beta2, 'o-')
    ax[0].axhline(0, linestyle='--', color='k')
    ax[0].axhline(1.55, linestyle='--', color='k')
    ax[0].set_ylim(-1000, 5500)
    plot_mode(n, 0, False, ax[1])
    plot_mode(n, 21, False, ax[2])
    if save:
        plt.savefig(f'fig/{n}.png')
    else:
        plt.pause(.1)
