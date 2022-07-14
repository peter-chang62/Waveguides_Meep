"""sim data analysis the arrays were saved as np.c_[kx, freq, vg] """

import numpy as np
import matplotlib.pyplot as plt
import os
from pynlo.media.crystals.XTAL_PPLN import Gayer5PctSellmeier
import clipboard_and_style_sheet
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.constants as sc
import scipy.integrate as scint
from numpy import ma

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


def height(s):
    return float(s.split('_')[1].split('.npy')[0])


def eps_func_sbstrt(freq):
    assert isinstance(freq, float) or isinstance(freq, int) or \
           all([isinstance(freq, np.ndarray), len(freq.shape) == 1, freq.shape[0] > 1])  # 1D array with length > 1

    if isinstance(freq, float) or isinstance(freq, int):
        if on_linux:
            return mtp.Al2O3.epsilon(freq)[2, 2]
        else:
            return Al2O3(freq)
    else:
        if on_linux:
            return mtp.Al2O3.epsilon(freq)[:, 2, 2]
        else:
            return Al2O3(freq)


def is_guided(kx, freq):
    # __________________________________________________________________________________________________________________
    # bulk propagation is omega = c * k / n
    # to be guided we need omega < omega_bulk
    # __________________________________________________________________________________________________________________

    index_substrate = eps_func_sbstrt(freq) ** 0.5
    freq_substrate = kx / index_substrate
    print(freq, freq_substrate, freq - freq_substrate)
    return freq < freq_substrate


def mode_area(I):
    # integral(I * dA) ** 2  / integral(I ** 2 * dA) is the common definition that is used
    # reference: https://www.rp-photonics.com/effective_mode_area.html
    # this gives an overall dimension of dA in the numerator
    area = scint.simpson(scint.simpson(I)) ** 2 / scint.simpson(scint.simpson(I ** 2))
    area /= resolution ** 2
    return area


eps_func_wvgd = lambda omega: Gayer5PctSellmeier(24.5).n((1 / omega) * 1e3) ** 2
conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9

# %%____________________________________________________________________________________________________________________
# Dispersion Simulation Data
path_disp = 'sim_output/06-16-2022/dispersion-curves/'
name_disp = [i.name for i in os.scandir(path_disp)]
name_disp = sorted(name_disp, key=height)
name_disp = sorted(name_disp, key=width)

get_disp = lambda n: np.load(path_disp + name_disp[n])
plot_nu_eps = lambda n: plt.plot(get_disp(n)[:, 1], (get_disp(n)[:, 0] / get_disp(n)[:, 1]) ** 2, '.-')
plot_wl_eps = lambda n: plt.plot(1 / get_disp(n)[:, 1], (get_disp(n)[:, 0] / get_disp(n)[:, 1]) ** 2, '.-')

# %%____________________________________________________________________________________________________________________
# E-field Simulation Data corresponding to the dispersion simulation data
path_fields = 'sim_output/06-16-2022/E-fields/'
name_fields = [i.name for i in os.scandir(path_fields)]
name_fields = sorted(name_fields, key=height)
name_fields = sorted(name_fields, key=width)

get_field = lambda n: np.squeeze(np.load(path_fields + name_fields[n]))
plot_field = lambda n, k_index, alpha=0.9: plt.imshow(get_field(n)[k_index][::-1, ::-1].T, cmap='RdBu', alpha=alpha)

# %%____________________________________________________________________________________________________________________
# epsilon grid simulation data corresponding to the dispersion simulation data
path_eps = 'sim_output/06-16-2022/eps/'
name_eps = [i.name for i in os.scandir(path_eps)]
name_eps = sorted(name_eps, key=height)
name_eps = sorted(name_eps, key=width)

get_eps = lambda n: np.load(path_eps + name_eps[n])
plot_eps = lambda n: plt.imshow(get_eps(n)[::-1, ::-1].T, interpolation='spline36', cmap='binary')


# %%____________________________________________________________________________________________________________________
def plot_mode(n, k_index):
    plot_eps(n)
    plot_field(n, k_index)
    data = get_disp(n)  # np.c_[res.kx, res.freq, res.v_g[:, 0, 0]]
    kx = data[:, 0]
    freq = data[:, 1]
    guided = is_guided(kx[k_index], freq[k_index])
    wl = 1 / freq[k_index]
    if guided:
        s = "guided"
    else:
        s = "NOT guided"
    plt.title(f'{np.round(width(name_eps[n]), 3)} x {np.round(height(name_eps[n]), 3)}' + ' $\mathrm{\mu m}$' '\n' +
              "$\mathrm{\lambda = }$" + '%.2f' % wl + ' $\mathrm{\mu m}$' + '\n' +
              '$\mathrm{A_{eff}}$ = %.3f' % mode_area(get_field(n)[k_index]) +
              ' $\mathrm{\mu m^2}$' + '\n' + s)


# %%____________________________________________________________________________________________________________________
def get_betas(n):
    # simulation data was saved as: np.c_[res.kx, res.freq, res.v_g[:, 0, 0]]
    data = get_disp(n)
    kx = data[:, 0]
    freq = data[:, 1]

    # beta = n * omega / c = 2 * np.pi * kx (the propagation constant)
    omega = freq * 2 * np.pi
    beta = kx * 2 * np.pi
    beta1 = np.gradient(beta, omega, edge_order=2)
    beta2 = np.gradient(beta1, omega, edge_order=2)

    spl_beta = InterpolatedUnivariateSpline(omega, beta, k=3)
    spl_beta1 = InterpolatedUnivariateSpline(omega, beta1, k=3)
    spl_beta2 = InterpolatedUnivariateSpline(omega, beta2, k=3)

    return beta, beta1, beta2, spl_beta, spl_beta1, spl_beta2


# %%____________________________________________________________________________________________________________________
# values fixed by the simulation data:
freq = get_disp(0)[:, 1]  # np.c_[kx, freq, vg]
omega = 2 * np.pi * freq
wl = 1 / freq
resolution = 30  # pixels / um

# %%____________________________________________________________________________________________________________________
# truncate the simulation data
[name_disp.remove(i) for i in name_disp.copy() if height(i) > 2.0]
[name_eps.remove(i) for i in name_eps.copy() if height(i) > 2.0]
[name_fields.remove(i) for i in name_fields.copy() if height(i) > 2.0]

# %%__________________________________________ Analyzing Results _______________________________________________________
wl_roots = np.zeros(len(name_disp), dtype=object)
BETA2 = np.zeros((len(name_disp), len(freq)))
fig, ax = plt.subplots(1, 1)
savefig = False
animate = False
for n in range(len(name_disp)):
    beta, beta1, beta2, spl_beta, spl_beta1, spl_beta2 = get_betas(n)
    wl_roots[n] = 2 * np.pi / spl_beta2.roots()
    BETA2[n] = beta2

    # ___________________________________ plotting _____________________________________________________________________
    # if you want to plot
    omega_plt = np.linspace(*omega[[0, -1]], 5000)
    beta2_plt = spl_beta2(omega_plt)
    if animate:
        ax.clear()
    ax.plot(2 * np.pi / omega_plt, beta2_plt * conversion)
    ax.set_title(f'{np.round(width(name_disp[n]), 3)} x {np.round(height(name_disp[n]), 3)}' + ' $\mathrm{\mu m}$')
    ax.set_xlabel("wavelength $\mathrm{\mu m}$")
    ax.set_ylabel("$\mathrm{\\beta_2 \; (ps^2/km})$")
    ax.axhline(0, color='k', linestyle='--')
    if savefig:
        plt.savefig(f'fig/{n}.png')
        print(n)
    elif animate:
        plt.pause(.1)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("wavelength $\mathrm{\mu m}$")
plt.ylabel("$\mathrm{\\beta_2 \; (ps^2/km})$")

# %%____________________________________________________________________________________________________________________
# plotting the analysis of the results
wl_zdw_long = []
wl_zdw_short = []
ind_zdw_long = []
ind_zdw_short = []
for n, i in enumerate(wl_roots):
    if len(i) > 1:  # if there are two roots
        wl_zdw_long.append(i[0])
        wl_zdw_short.append(i[1])
        ind_zdw_long.append(n)
        ind_zdw_short.append(n)
    elif len(i) > 0:  # if there is only one root
        wl_zdw_short.append(i[0])
        ind_zdw_short.append(n)
wl_zdw_long = np.array(wl_zdw_long)
wl_zdw_short = np.array(wl_zdw_short)
ind_zdw_long = np.array(ind_zdw_long)
ind_zdw_short = np.array(ind_zdw_short)

check_if_guided = lambda n: is_guided(get_disp(n)[:, 0], freq)

w = np.array([width(i) for i in name_disp])
h = np.array([height(i) for i in name_disp])
shape = (len(set(w)), len(set(h)))
w.resize(shape)
h.resize(shape)
ind_zdw_long_2D = np.unravel_index(ind_zdw_long, w.shape)
ind_zdw_short_2D = np.unravel_index(ind_zdw_short, w.shape)
wl_zdw_short_2D = np.zeros(w.shape)
wl_zdw_short_2D[ind_zdw_short_2D] = wl_zdw_short
wl_zdw_short_2D = ma.masked_values(wl_zdw_short_2D, 0)
wl_zdw_long_2D = np.zeros(w.shape)
wl_zdw_long_2D[ind_zdw_long_2D] = wl_zdw_long
wl_zdw_long_2D = ma.masked_values(wl_zdw_long_2D, 0)

cmap = 'afmhot'
fig, ax = plt.subplots(1, 1)
img = ax.pcolormesh(w, h, wl_zdw_short_2D, cmap=cmap)
plt.colorbar(img)
ax.set_xlabel("width ($\mathrm{\mu m}$)")
ax.set_ylabel("height ($\mathrm{\mu m}$)")
ax.set_title("$\mathrm{\lambda_{ZDW}}$ shortest")

fig, ax = plt.subplots(1, 1)
img = ax.pcolormesh(w, h, wl_zdw_long_2D, cmap=cmap)
plt.colorbar(img)
ax.set_xlabel("width ($\mathrm{\mu m}$)")
ax.set_ylabel("height ($\mathrm{\mu m}$)")
ax.set_title("$\mathrm{\lambda_{ZDW}}$ longest")

# %%____________________________________________________________________________________________________________________
w_unique = np.array(list(set([width(i) for i in name_disp])))
h_unique = np.array(list(set([height(i) for i in name_disp])))
w_unique.sort()
h_unique.sort()

ind_wl = np.argmin(abs(wl - 1.55))

# fixed width
ind_w = np.argmin(abs(w_unique - 2.19))
step = 30
fig, ax = plt.subplots(1, 1)
for i in range(step):
    ax.clear()
    plot_mode(i + step * ind_w, ind_wl)
    plt.pause(.1)
    # plt.savefig(f'fig/{i}.png')

# fixed height
ind_h = np.argmin(abs(h_unique - 1.15))
fig, ax = plt.subplots(1, 1)
h = 0
for i in range(ind_h, len(name_disp), step):
    ax.clear()
    plot_mode(i, ind_wl)
    plt.pause(.1)
    # plt.savefig(f'fig/{h}.png')
    h += 1
