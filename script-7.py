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
import scipy.constants as sc
import scipy.integrate as scint
from scipy.interpolate import interp1d
import pynlo
from pynlo import utility as utils
import time
from scipy.interpolate import UnivariateSpline

# %% global variables
resolution = 20
time_0 = time.time()


# %% ----- function defs
def get_bp_ind(wl_grid, wl_ll, wl_ul):
    return np.where(np.logical_and(wl_grid >= wl_ll, wl_grid <= wl_ul), 1, 0)


def e_p_in_window(wl_grid, dv, a_v, wl_ll, wl_ul):
    h = get_bp_ind(wl_grid, wl_ll, wl_ul)
    a_v_filt = a_v * h
    return scint.simps(abs(a_v_filt) ** 2, axis=1, dx=dv)


def mode_area(I):
    # integral(I * dA) ** 2  / integral(I ** 2 * dA) is the
    # common definition that is used
    # reference: https://www.rp-photonics.com/effective_mode_area.html
    # this gives an overall dimension of dA in the numerator
    area = scint.simpson(scint.simpson(I)) ** 2 / scint.simpson(scint.simpson(I**2))
    area /= resolution**2
    return area


def load_waveguide(pulse, res, sim):
    pulse: pynlo.light.Pulse
    v_grid = pulse.v_grid
    v0 = pulse.v0

    wl, b = 1 / res.freq, res.kx.flatten() * 2 * np.pi
    k = b * 1e6 / (2 * np.pi)  # 1/m
    nu = sc.c / (wl * 1e-6)
    n = sc.c * k / nu
    n_wvgd = interp1d(nu, n, kind="cubic", bounds_error=False, fill_value=0)
    n_eff = n_wvgd(v_grid)
    beta = n_eff * 2 * np.pi * v_grid / sc.c  # n * w / sc.c

    ind = np.argmin(abs(wl - 1.55))
    field = sim.E[ind][0][:, :, 1].__abs__() ** 2  # mp.Ey = 1
    a_eff = mode_area(field) * 1e-12  # um^2 -> m^2 @ lamda = 1560 nm

    # 2nd order nonlinearity
    d_eff = 27e-12  # 27 pm / V
    chi2_eff = 2 * d_eff
    g2 = utils.chi2.g2_shg(v0, v_grid, n_eff, a_eff, chi2_eff)

    # 3rd order nonlinearity
    chi3_eff = 5200e-24
    g3 = utils.chi3.g3_spm(n_eff, a_eff, chi3_eff)

    mode = pynlo.media.Mode(v_grid, beta, g2=g2, g3=g3)
    return mode


def simulate(pulse, mode, length=3e-3, npts=100):
    # model = pynlo.model.SM_UPE(pulse, mode)
    model = pynlo.model.UPE(pulse, mode)
    local_error = 1e-6
    dz = model.estimate_step_size(n=20, local_error=local_error)

    z_grid = np.linspace(0, length, npts)
    # pulse_out, z, a_t, a_v = model.simulate(
    #     z_grid, dz=dz, local_error=local_error, n_records=None, plot=None
    # )
    sim = model.simulate(
        z_grid, dz=dz, local_error=local_error, n_records=None, plot=None
    )
    return sim


# %% ----- Gayer paper Sellmeier equation for ne (taken from PyNLO 1 / omega is in um
# n = squrt[epsilon] so epsilon = n^2
def eps_func_wvgd(omega):
    # omega is in inverse micron
    um = 1e-6
    v = omega * sc.c / um
    return n_MgLN_G(v, T=24.5, axis="e") ** 2


def n_MgLN_G(v, T=24.5, axis="e"):
    """
    Range of Validity:
        - 500 nm to 4000 nm
        - 20 C to 200 C
        - 48.5 mol % Li
        - 5 mol % Mg

    Gayer, O., Sacks, Z., Galun, E. et al. Temperature and wavelength
    dependent refractive index equations for MgO-doped congruent and
    stoichiometric LiNbO3 . Appl. Phys. B 91, 343–348 (2008).

    https://doi.org/10.1007/s00340-008-2998-2

    """
    if axis == "e":
        a1 = 5.756  # plasmons in the far UV
        a2 = 0.0983  # weight of UV pole
        a3 = 0.2020  # pole in UV
        a4 = 189.32  # weight of IR pole
        a5 = 12.52  # pole in IR
        a6 = 1.32e-2  # phonon absorption in IR
        b1 = 2.860e-6
        b2 = 4.700e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
    elif axis == "o":
        a1 = 5.653  # plasmons in the far UV
        a2 = 0.1185  # weight of UV pole
        a3 = 0.2091  # pole in UV
        a4 = 89.61  # weight of IR pole
        a5 = 10.85  # pole in IR
        a6 = 1.97e-2  # phonon absorption in IR
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.188e-6

    else:
        raise ValueError("axis needs to be o or e")

    wvl = sc.c / v * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        (a1 + b1 * f)
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


# %% ----- create sim instance ------------------------------------------------
etch_width = 1.245
etch_depth = 0.7
sim = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=1,  # I'll fix the height at 1 um now
    substrate_medium=mtp.Al2O3,
    waveguide_medium=mt.LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    num_bands=1,
)

# %% ----- waveguide simulation -----------------------------------------------
# individual sampling (comment out if running the for loop block instead)
res = sim.calc_dispersion(0.4, 5, 100, eps_func_wvgd=eps_func_wvgd)  # simulate

wl = 1 / res.freq
omega = res.freq * 2 * np.pi
conversion = sc.c**-2 * 1e12**2 * 1e3**2 * 1e-9
beta = res.kx.flatten() * 2 * np.pi
beta1 = np.gradient(beta, omega, edge_order=2)
beta2 = np.gradient(beta1, omega, edge_order=2) * conversion

# %% ----- PyNLO simulation ---------------------------------------------------
# n_points = 2**13
n_points = 256
v_min = sc.c / 3.5e-6  # sc.c / 5000 nm
v_max = sc.c / 700e-9  # sc.c / 400 nm
e_p = 100e-12
t_fwhm = 50e-15
pulse = pynlo.light.Pulse.Sech(
    n=n_points,
    v_min=v_min,
    v_max=v_max,
    v0=sc.c / 1560e-9,
    e_p=e_p,
    t_fwhm=t_fwhm,
    min_time_window=10e-12,
)
mode = load_waveguide(pulse, res, sim)
sim_wvgd = simulate(pulse, mode, length=10e-3, npts=100)
pulse_out, z, a_t, a_v = sim_wvgd.pulse_out, sim_wvgd.z, sim_wvgd.a_t, sim_wvgd.a_v
wl_grid = sc.c / pulse.v_grid

frep = 1e9
power = (
    e_p_in_window(wl_grid=wl_grid, dv=pulse.dv, a_v=a_v, wl_ll=3e-6, wl_ul=5e-6) * frep
)

# %%  ----- plotting ----------------------------------------------------------
fig, ax = plt.subplots(1, 1)
# ax.plot(wl[:-10], beta2[:-10], "o")
s = UnivariateSpline(wl[wl > 0.6][::-1], beta2[wl > 0.6][::-1], s=0)
_ = np.linspace(wl[wl > 0.6].min(), wl[wl > 0.6].max(), 1000)
ax.plot(_, s(_), linewidth=2)
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_ylabel("$\\mathrm{\\beta_2 \\; (ps^2/km})$")
fig.tight_layout()

fig, ax = sim.plot_mode(0, np.argmin(abs(wl - 3.5)))
ax.title.set_text(
    ax.title.get_text()
    + "\n"
    + "$\\mathrm{\\lambda = }$"
    + "%.2f" % wl[np.argmin(abs(wl - 3.5))]
    + " $\\mathrm{\\mu m}$"
)

fig, ax = sim.plot_mode(0, np.argmin(abs(wl - 4.0)))
ax.title.set_text(
    ax.title.get_text()
    + "\n"
    + "$\\mathrm{\\lambda = }$"
    + "%.2f" % wl[np.argmin(abs(wl - 4.0))]
    + " $\\mathrm{\\mu m}$"
)

sim_wvgd.plot("wvl")

time_1 = time.time()

# %% ----- total run time
print(time_1 - time_0)
