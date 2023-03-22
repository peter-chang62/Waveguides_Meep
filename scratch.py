# %% package imports
import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.interpolate as spi
from numpy import ma
import pynlo_extras as pe


cr.style_sheet()

unguided = np.load("unguided_index.npy")
data = np.vstack(
    [np.load("unguided_dispersion.npy")[::-1], np.load("guided_dispersion.npy")]
)
k_points = np.load("k_points.npy")

um = 1e-6

nu_grid = k_points[:, 0] * sc.c / um
w_grid = nu_grid * 2 * np.pi
b = k_points[:, 1] * 2 * np.pi / um
b_1 = np.gradient(b, w_grid, edge_order=2)
loss = data[:, 1] * sc.c / um * b_1[: len(data)]

fig = plt.figure(num="propagation loss")
plt.plot(1 / data[:, 0], loss, "o")
data = np.append(data[:12], data[20:], axis=0)
loss = np.append(loss[:12], loss[20:], axis=0)
f = spi.interp1d(data[:, 0], loss, kind="cubic", fill_value="extrapolate")
nu = np.linspace(*data[:, 0][[0, -1]], 5000)
plt.plot(1 / nu, f(nu))
plt.xlabel("wavelength($\\mathrm{\\mu m}$)")
plt.ylabel("propagation loss (1 / m)")
plt.tight_layout()

# v_grid = np.linspace(sc.c / 6e-6, sc.c / 3e-6, 5000)
# alpha = pe.materials.ppln_alpha(v_grid)
# wl_grid = sc.c * 1e6 / v_grid
# plt.plot(wl_grid, alpha)
