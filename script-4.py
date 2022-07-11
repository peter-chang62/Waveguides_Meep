"""sim data analysis the arrays were saved as np.c_[kx, freq, vg] """

import numpy as np
import matplotlib.pyplot as plt
import os
from pynlo.media.crystals.XTAL_PPLN import Gayer5PctSellmeier
import clipboard_and_style_sheet
from scipy.interpolate import InterpolatedUnivariateSpline

clipboard_and_style_sheet.style_sheet()


def width(s):
    return float(s.split('_')[0])


def height(s):
    return float(s.split('_')[1].split('.npy')[0])


eps_func_wvgd = lambda omega: Gayer5PctSellmeier(24.5).n((1 / omega) * 1e3) ** 2

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

get_field = lambda n: np.load(path_fields + name_fields[n])
plot_field = lambda n, k_index, alpha=0.9: plt.imshow(get_field(n)[k_index, 0][::-1, ::-1].T, cmap='RdBu', alpha=alpha)

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


def vg_beta2_D(n):
    data = get_disp(n)
    kx = data[:, 0]
    freq = data[:, 1]

    # beta = n * omega / c = 2 * np.pi * kx (the propagation constant)
    omega = freq * 2 * np.pi
    beta = kx * 2 * np.pi
    spl_beta = InterpolatedUnivariateSpline(omega, beta, k=5)
    spl_beta1 = spl_beta.derivative(1)
    spl_beta2 = spl_beta.derivative(2)

    return spl_beta(freq), spl_beta1(freq), spl_beta2(freq), spl_beta, spl_beta1, spl_beta2


# %%____________________________________________________________________________________________________________________
roots = []
n_roots = np.zeros(len(name_disp))
for n in range(len(name_disp)):
    beta, beta1, beta2, spl_beta, spl_beta1, spl_beta2 = vg_beta2_D(n)
    roots.append(spl_beta2.roots())
    n_roots[n] = (len(roots[n]))

# %%____________________________________________________________________________________________________________________
# TODO restrict analysis to guided modes
