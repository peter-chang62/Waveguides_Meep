"""
Combine MEEP simulations and PyNLO into one script (will only run on a linux
computer). This is good for sort of point sampling of the big simulation
outputs. You can make small adjustment to parameters here
"""

# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import meep.materials as mt
from TFW_meep import materials as mtp
from TFW_meep import waveguide_dispersion as wg
from scipy.constants import c
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline
import pynlo

# %% global variables
resolution = 20

um = 1e-6
ps = 1e-12
km = 1e3


# %% ----- function defs
def mode_area(I):
    # integral(I * dA) ** 2  / integral(I ** 2 * dA) is the
    # common definition that is used
    # reference: https://www.rp-photonics.com/effective_mode_area.html
    # this gives an overall dimension of dA in the numerator
    area = simpson(simpson(I)) ** 2 / simpson(simpson(I**2))
    area /= resolution**2
    return area


# %% ----- Gayer paper Sellmeier equation for ne (taken from PyNLO 1 / omega is in um
# n = squrt[epsilon] so epsilon = n^2
def eps_func_wvgd(omega):
    # omega is in inverse micron
    um = 1e-6
    v = omega * c / um
    # return pynlo.materials.n_MgLN_G(v, T=24.5, axis="e") ** 2
    return pynlo.materials.n_cLN(v, T=24.5, axis="e") ** 2


# %% ----- create sim instance ------------------------------------------------
etch_width = 2.0
etch_depth = 0.70
film_thickness = 0.850
sim = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=film_thickness,
    substrate_medium=mtp.Al2O3,
    waveguide_medium=mt.LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    num_bands=1,
)

# %% ----- waveguide simulation -----------------------------------------------
res = sim.calc_dispersion(
    0.7, 5, 100, eps_func_wvgd=eps_func_wvgd, etch_angle=80.0
)  # simulate

beta = res.kx.flatten() * 2 * np.pi / um

# %% ----- PyNLO simulation ---------------------------------------------------
n_points = 256
v_min = c / 5.5e-6  # c / 5000 nm
v_max = c / 0.7e-6  # c / 400 nm
e_p = 300e-12
t_fwhm = 100e-15
pulse = pynlo.light.Pulse.Sech(
    n=n_points,
    v_min=v_min,
    v_max=v_max,
    v0=c / 1560e-9,
    e_p=e_p,
    t_fwhm=t_fwhm,
    min_time_window=10e-12,
)

ppln = pynlo.materials.MgLN()
I = sim.E[:, 0, :, :, 1]
I = abs(I[abs(res.freq - 1 / 1.55).argmin()]) ** 2
a_eff = mode_area(I) * um**2

beta_grid = InterpolatedUnivariateSpline(res.freq * c / um, beta)(pulse.v_grid)

# %% ---- non-chirped poling --------------------------------------------------
length = 10e-3
model = ppln.generate_model(
    pulse,
    a_eff,
    length,
    g2_inv=None,
    beta=beta_grid,
    is_gaussian_beam=False,
)

dz = model.estimate_step_size()
sim_pynlo = model.simulate(length, dz=dz, n_records=100, plot="wvl")

# %%  ----- plotting ----------------------------------------------------------
sim_pynlo.plot("wvl")

(idx,) = np.logical_and(1e-6 < pulse.wl_grid, pulse.wl_grid < 3.5e-6).nonzero()

fig, ax = plt.subplots(1, 1)
ax.plot(
    pulse.wl_grid[idx] * 1e9, model.dispersive_wave_dk[idx] / (2 * np.pi), linewidth=2
)
ax.grid(True)
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_ylabel("dispersive wave phase mismatch (1/m)")
fig.tight_layout()

fig, ax = plt.subplots(1, 1)
beta2 = InterpolatedUnivariateSpline(model.w_grid, model.beta).derivative(2)(
    model.w_grid
)
ax.plot(pulse.wl_grid[idx] * 1e9, beta2[idx] / (ps**2 / km), linewidth=2)
ax.grid(True)
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_ylabel("$\\mathrm{\\beta_2 \\; (ps^2/km)}$")
fig.tight_layout()

wl = 1 / res.freq
fig, ax = sim.plot_mode(0, np.argmin(abs(wl - 1.0)))
ax.title.set_text(
    ax.title.get_text()
    + "\n"
    + "$\\mathrm{\\lambda = }$"
    + "%.2f" % wl[np.argmin(abs(wl - 1.0))]
    + " $\\mathrm{\\mu m}$"
)
fig.tight_layout()

fig, ax = sim.plot_mode(0, np.argmin(abs(wl - 1.55)))
ax.title.set_text(
    ax.title.get_text()
    + "\n"
    + "$\\mathrm{\\lambda = }$"
    + "%.2f" % wl[np.argmin(abs(wl - 1.55))]
    + " $\\mathrm{\\mu m}$"
)
fig.tight_layout()

fig, ax = sim.plot_mode(0, np.argmin(abs(wl - 2.0)))
ax.title.set_text(
    ax.title.get_text()
    + "\n"
    + "$\\mathrm{\\lambda = }$"
    + "%.2f" % wl[np.argmin(abs(wl - 2.0))]
    + " $\\mathrm{\\mu m}$"
)
fig.tight_layout()

fig, ax = sim.plot_mode(0, np.argmin(abs(wl - 3.0)))
ax.title.set_text(
    ax.title.get_text()
    + "\n"
    + "$\\mathrm{\\lambda = }$"
    + "%.2f" % wl[np.argmin(abs(wl - 3.0))]
    + " $\\mathrm{\\mu m}$"
)
fig.tight_layout()

fig, ax = sim.plot_mode(0, np.argmin(abs(wl - 3.5)))
ax.title.set_text(
    ax.title.get_text()
    + "\n"
    + "$\\mathrm{\\lambda = }$"
    + "%.2f" % wl[np.argmin(abs(wl - 3.5))]
    + " $\\mathrm{\\mu m}$"
)
fig.tight_layout()
