"""
First attempt to run waveguide dispersion parameters through Connor's PyNLO class
"""

# Imports
import numpy as np
from scipy.constants import pi, c
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pynlo_connor as pynlo
from pynlo_connor import utility as utils

# importing dispersion data ____________________________________________________________________________________________
b_data = np.load("sim_output/10-03-2022/3x.7_wl_omega_b_b1_b2.npy")
wl, omega, b, b1, b2 = b_data[:, 0], b_data[:, 1], b_data[:, 2], b_data[:, 3], b_data[:, 4]
k = b * 1e6 / (2 * np.pi)  # 1 / m
freq = c / (wl * 1e-6)  # Hz
n = c * k / freq  # n = k * c / freq
n_wvgd = interp1d(freq, n, kind='cubic', bounds_error=True)

# %% Pulse Properties __________________________________________________________________________________________________
n_points = 2 ** 13
v_min = c / 4500e-9  # c / 4500 nm
v_max = c / 800e-9  # c / 815 nm
v0 = c / 1560 - 9  # c / 1550 nm
e_p = 1e-9  # 1 nJ
t_fwhm = 50e-15  # 50 fs

pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
pulse.rtf_grids(n_harmonic=1, update=True)  # anti-aliasing

v_grid = pulse.v_grid
t_grid = pulse.t_grid

# %% ___________________________________________________________________________________________________________________
# Waveguide properties
length = 20e-3  # 10 mm
a_eff = 2.56e-12

n_eff = n_wvgd(v_grid)
beta = n_eff * 2 * np.pi * v_grid / c  # n * w / c

# 2nd order nonlinearity
d_eff = 27e-12  # 27 pm / V
chi2_eff = 2 * d_eff
g2 = utils.chi2.g2_shg(v0, v_grid, n_eff, a_eff, chi2_eff)

# 3rd order nonlinearity
chi3_eff = 5200e-24
g3 = utils.chi3.g3_spm(n_eff, a_eff, chi3_eff)

# Mode
mode = pynlo.media.Mode(
    v_grid=v_grid,
    beta_v=beta,
    g2_v=g2,
    g2_inv=None,
    g3_v=g3,
    z=0.0
)

# Model
model = pynlo.model.SM_UPE(pulse, mode)
local_error = 1e-6
dz = model.estimate_step_size(n=20, local_error=local_error)

# Simulate
z_grid = np.linspace(0, length, 100)
pulse_out, z, a_t, a_v = model.simulate(z_grid, dz=dz, local_error=local_error, n_records=100, plot=None)

fig = plt.figure("Simulation Results", clear=True)
ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2, sharex=ax1)

p_v_dB = 10 * np.log10(np.abs(a_v) ** 2)
p_v_dB -= p_v_dB.max()
ax0.plot(1e-12 * v_grid, p_v_dB[0], color="b")
ax0.plot(1e-12 * v_grid, p_v_dB[-1], color="g")
ax2.pcolormesh(1e-12 * v_grid, 1e3 * z, p_v_dB, vmin=-40.0, vmax=0, shading="auto")
ax0.set_ylim(bottom=-50, top=10)
ax2.set_xlabel('Frequency (THz)')

p_t_dB = 10 * np.log10(np.abs(a_t) ** 2)
p_t_dB -= p_t_dB.max()
ax1.plot(1e12 * t_grid, p_t_dB[0], color="b")
ax1.plot(1e12 * t_grid, p_t_dB[-1], color="g")
ax3.pcolormesh(1e12 * t_grid, 1e3 * z, p_t_dB, vmin=-40.0, vmax=0, shading="auto")
ax1.set_ylim(bottom=-50, top=10)
ax3.set_xlabel('Time (ps)')

ax0.set_ylabel('Power (dB)')
ax2.set_ylabel('Propagation Distance (mm)')
fig.tight_layout()
fig.show()
