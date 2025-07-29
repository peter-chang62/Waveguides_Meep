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
from scipy.integrate import simpson
import meep as mp
import scipy.constants as sc

# %% global variables
resolution = 20

um = 1e-6
ps = 1e-12
km = 1e3


# %% ----- Gayer paper Sellmeier equation for ne (taken from PyNLO 1 / omega is in um
# n = squrt[epsilon] so epsilon = n^2
def eps_func_wvgd(omega):
    # omega is in inverse micron
    um = 1e-6
    v = omega * sc.c / um
    return n_cLN(v, T=24.5, axis="e") ** 2


def n_cLN(v_grid, T=24.5, axis="e"):
    """
    Refractive index of congruent lithium niobate.

    References
    ----------
    Dieter H. Jundt, "Temperature-dependent Sellmeier equation for the index of
     refraction, ne, in congruent lithium niobate," Opt. Lett. 22, 1553-1555
     (1997). https://doi.org/10.1364/OL.22.001553

    """

    assert axis == "e"

    a1 = 5.35583
    a2 = 0.100473
    a3 = 0.20692
    a4 = 100.0
    a5 = 11.34927
    a6 = 1.5334e-2
    b1 = 4.629e-7
    b2 = 3.862e-8
    b3 = -0.89e-8
    b4 = 2.657e-5

    wvl = sc.c / v_grid * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        a1
        + b1 * f
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


def chi3_to_n2(n, chi3):
    return chi3 / (4 / 3 * sc.epsilon_0 * sc.c * n**2)


# ---- Effective Nonlinearity gamma and Kerr Parameter
def n2_to_gamma(v_grid, a_eff, n2):
    return n2 * (2 * np.pi * v_grid / (sc.c * a_eff))


def mode_area(I):
    # integral(I * dA) ** 2  / integral(I ** 2 * dA) is the
    # common definition that is used
    # reference: https://www.rp-photonics.com/effective_mode_area.html
    # this gives an overall dimension of dA in the numerator
    area = simpson(simpson(I)) ** 2 / simpson(simpson(I**2))
    area /= resolution**2
    return area


# %% ----- create sim instance ------------------------------------------------
etch_width = 2.0
etch_depth = 0.400
film_thickness = 0.850

Al2O3 = mtp.Al2O3
LiNbO3 = mt.LiNbO3
SiO2 = mt.SiO2

Al2O3.valid_freq_range = mp.FreqRange(0, mp.inf)
LiNbO3.valid_freq_range = mp.FreqRange(0, mp.inf)
SiO2.valid_freq_range = mp.FreqRange(0, mp.inf)
sim_al2o3 = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=film_thickness,
    z_offset_wvgd=1.5,  # some of the modes at the end are a bit leaky
    substrate_medium=Al2O3,
    waveguide_medium=LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    num_bands=1,
)

sim_sio2 = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=film_thickness,
    z_offset_wvgd=1.5,  # some of the modes at the end are a bit leaky
    substrate_medium=SiO2,
    waveguide_medium=LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    num_bands=1,
)

sim_air = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=film_thickness,
    z_offset_wvgd=1.5,  # some of the modes at the end are a bit leaky
    substrate_medium=mp.Medium(),  # air
    waveguide_medium=LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    num_bands=1,
)

# %% ----- waveguide simulation -----------------------------------------------
res_al2o3 = sim_al2o3.calc_dispersion(
    3,
    5,
    100,
    eps_func_wvgd=eps_func_wvgd,
    etch_angle=80.0,
)

res_sio2 = sim_sio2.calc_dispersion(
    3,
    5,
    100,
    eps_func_wvgd=eps_func_wvgd,
    etch_angle=80.0,
)

res_air = sim_air.calc_dispersion(
    3,
    5,
    100,
    eps_func_wvgd=eps_func_wvgd,
    etch_angle=80.0,
)

# %% -----
figsize = np.array([2.99, 2.64])

fig, ax = plt.subplots(1, 1, num="Al2O3", figsize=figsize)
ax.plot(
    1 / res_al2o3.freq,
    res_al2o3.kx[:, 0],
)
ax.plot(
    1 / res_al2o3.freq,
    res_al2o3.freq * res_al2o3._index_sbstrt,
)
ax.set_xlabel("wavelength ($\\mu m$)")
ax.set_ylabel("k ($1 / \\mu m$)")
fig.tight_layout()
fig.tight_layout()
fig.tight_layout()

fig, ax = plt.subplots(1, 1, num="SiO2", figsize=figsize)
ax.plot(
    1 / res_sio2.freq,
    res_sio2.kx[:, 0],
)
ax.plot(
    1 / res_sio2.freq,
    res_sio2.freq * res_sio2._index_sbstrt,
)
ax.set_xlabel("wavelength ($\\mu m$)")
ax.set_ylabel("k ($1 / \\mu m$)")
fig.tight_layout()
fig.tight_layout()
fig.tight_layout()

fig, ax = plt.subplots(1, 1, num="air", figsize=figsize)
ax.plot(
    1 / res_air.freq,
    res_air.kx[:, 0],
)
ax.plot(
    1 / res_air.freq,
    res_air.freq * res_air._index_sbstrt,
)
ax.set_xlabel("wavelength ($\\mu m$)")
ax.set_ylabel("k ($1 / \\mu m$)")
fig.tight_layout()
fig.tight_layout()
fig.tight_layout()

# %% ----- some random thinking -----------------------------------------------
# what's the effective gamma?
res = sim_al2o3.calc_dispersion(
    freq_array=np.array([1 / 1.55]), eps_func_wvgd=eps_func_wvgd, etch_angle=80.0
)
n = res.kx.flatten() / res.freq


chi3_eff = 5200e-24  # pm**2/V**2
n2 = chi3_to_n2(n, chi3_eff)

a_eff = mode_area(res.power[:, :, 1]) * um**2
gamma = n2_to_gamma(sc.c / 1.55e-6, a_eff, n2)
print(f"gamma is {np.round(gamma[0] * 1e3, 1)} 1/(Wkm)")
