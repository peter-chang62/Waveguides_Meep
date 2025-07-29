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
import multiprocessing
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
sim = wg.ThinFilmWaveguide(
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

# %% ----- multiprocessing version --------------------------------------------
wl_min = 0.8e-6
wl_max = 2.4e-6


def func(etch_depth):
    sim.etch_depth = etch_depth
    res = sim.calc_dispersion(
        wl_min * 1e6,
        wl_max * 1e6,
        100,
        eps_func_wvgd=eps_func_wvgd,
        etch_angle=80.0,
    )

    beta = res.kx.flatten() * 2 * np.pi / um
    omega = res.freq * 2 * np.pi * sc.c / um

    polyfit = np.polyfit(
        omega, beta, 9
    )  # higher than 9 doesn't alter the fit (poorly conditioned)
    polyfit_2 = np.polyder(polyfit, 2)
    poly1d = np.poly1d(polyfit)
    poly1d_2 = np.poly1d(polyfit_2)

    return np.c_[res.freq, beta, poly1d_2(omega)]


if __name__ == "__main__":
    with multiprocessing.Pool(8) as p:
        data = p.map(func, np.arange(0.1, 0.7 + 0.1, 0.1))

    data = np.asarray(data)

    # -------------------------------------------------------------------------
    etch_depth=np.arange(0.1, 0.7 + 0.1, 0.1)
    colors = plt.cm.gray(np.linspace(0, 0.7, etch_depth.size))

    fig, ax = plt.subplots(1, 1)
    for n in range(data[:, :, 2].shape[0]):
        ax.plot(1 / data[0, :, 0], data[:, :, 2][n] / (ps**2 / km), color=colors[n])

    ax.set_xlabel("wavelength ($\\mu m$)")
    ax.set_ylabel("$\\beta_2 \\; (ps^{2}/km)$")
