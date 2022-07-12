"""sim data analysis the arrays were saved as np.c_[kx, freq, vg] """

import numpy as np
import matplotlib.pyplot as plt
import os
from pynlo.media.crystals.XTAL_PPLN import Gayer5PctSellmeier
import clipboard_and_style_sheet
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.constants as sc

clipboard_and_style_sheet.style_sheet()


def width(s):
    return float(s.split('_')[0])


def height(s):
    return float(s.split('_')[1].split('.npy')[0])


eps_func_wvgd = lambda omega: Gayer5PctSellmeier(24.5).n((1 / omega) * 1e3) ** 2

# %%____________________________________________________________________________________________________________________
path_disp = 'sim_output/06-16-2022/dispersion-curves/'
name_disp = [i.name for i in os.scandir(path_disp)]
name_disp = sorted(name_disp, key=height)
name_disp = sorted(name_disp, key=width)

get_disp = lambda n: np.load(path_disp + name_disp[n])
plot_nu_eps = lambda n: plt.plot(get_disp(n)[:, 1], (get_disp(n)[:, 0] / get_disp(n)[:, 1]) ** 2, '.-')
plot_wl_eps = lambda n: plt.plot(1 / get_disp(n)[:, 1], (get_disp(n)[:, 0] / get_disp(n)[:, 1]) ** 2, '.-')

# %%____________________________________________________________________________________________________________________
path_fields = 'sim_output/06-16-2022/E-fields/'
name_fields = [i.name for i in os.scandir(path_fields)]
name_fields = sorted(name_fields, key=height)
name_fields = sorted(name_fields, key=width)

get_field = lambda n: np.load(path_fields + name_fields[n])
plot_field = lambda n, k_index, alpha=0.9: plt.imshow(get_field(n)[k_index, 0][::-1, ::-1].T, cmap='RdBu', alpha=alpha)

# %%____________________________________________________________________________________________________________________
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


# %%____________________________________________________________________________________________________________________
plt.figure()
[plot_nu_eps(i) for i in range(len(name_disp))]
plt.xlabel("$\mathrm{\\nu \; (1 / \mu m)}$")
plt.ylabel("$\mathrm{\\epsilon}$")

# %%____________________________________________________________________________________________________________________
N = 115
fig, ax = plt.subplots(1, 1)
wl = 1 / get_disp(N)[:, 1]
for n in range(26):
    ax.clear()
    plot_mode(N, n)
    ax.set_title('%.3f' % wl[n] + " $\mathrm{\\mu m}$")
    # plt.savefig(f'fig/{n}.png')
    plt.pause(.1)

# %%____________________________________________________________________________________________________________________
# beta is calculated in (2 pi / um) with c = 1, so the conversion goes from
# s^2 um / m^2 -> ps^2 um / km^2 -> ps^2 / km
fig, ax = plt.subplots(1, 1)
conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9
BETA2 = np.zeros((len(name_disp), len(get_disp(0))))
for i in range(len(name_disp)):
    # ax.clear()
    data = get_disp(i)
    kx = data[:, 0]
    freq = data[:, 1]

    omega = freq * 2 * np.pi
    beta = kx * 2 * np.pi
    beta1 = np.gradient(beta, omega, edge_order=2)
    beta2 = np.gradient(beta1, omega, edge_order=2)
    spl_beta2 = InterpolatedUnivariateSpline(omega, beta2, k=3)

    omega_plot = np.linspace(*freq[[0, -1]], 5000) * 2 * np.pi
    ax.plot((2 * np.pi / omega_plot), spl_beta2(omega_plot) * conversion)
    BETA2[i] = spl_beta2(omega)
    # plt.pause(.1)

ax.set_xlabel("wavelength ($\mathrm{\mu m}$")
ax.set_ylabel("$\\beta_2$")
ax.axhline(0, color='k', linestyle='--')
