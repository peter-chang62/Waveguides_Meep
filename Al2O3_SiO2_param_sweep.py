"""
sweep etch depth, etch width, and film thickness. save dispersion and mode area
for each one. Run this script to calculate Al2O3 and again to calculate SiO2
(switch out the relevant commented lines).

once run, don't change the frequency and waveguide width, unless you also
change it in plotting_Al2O3_SiO2_param_sweep.py
"""

# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import meep as mp
from meep import materials as mt
from TFW_meep import materials as mtp
from TFW_meep import waveguide_dispersion as wg
import scipy.constants as sc
from scipy.constants import c
from tqdm import tqdm


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
etch_width = 0.5
etch_depth = 0.1
LiN_thickness = 1.0
resolution = 20
sim = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=LiN_thickness,
    # substrate_medium=mt.SiO2,
    substrate_medium=mtp.Al2O3,
    waveguide_medium=mt.LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    z_offset_wvgd=1.5,
    num_bands=1,
)

# %% ----- waveguide simulation -----------------------------------------------
step = 0.1
# for film_thickness in np.arange(0.6, 1.0 + step, step):
film_thickness = 0.4
size_d = np.arange(0.1, film_thickness + step, step).size
size_w = np.arange(0.5, 2.0 + step, step).size
data = np.zeros((size_d, size_w, 101, 2))

for n_d, etch_depth in enumerate(tqdm(np.arange(0.1, film_thickness + step, step))):
    for n_w, etch_width in enumerate(tqdm(np.arange(0.5, 2.0 + step, step))):
        sim.height = np.round(film_thickness, 3)
        sim.etch_depth = np.round(etch_depth, 3)
        sim.etch_width = np.round(etch_width, 3)

        res = sim.calc_dispersion(0.8, 4.5, 100, eps_func_wvgd=eps_func_wvgd)
        # res = sim.calc_dispersion(
        #     1 / mt.SiO2.valid_freq_range.max,
        #     1 / mt.SiO2.valid_freq_range.min,
        #     100,
        #     eps_func_wvgd=eps_func_wvgd,
        # )

        um = 1e-6
        ps = 1e-12
        km = 1e3

        omega = res.freq * 2 * np.pi * c / um
        beta = res.kx.flatten() * 2 * np.pi / um
        area = np.zeros(res.freq.size)
        for n_f in range(res.freq.size):
            x = sim.E[n_f][0][:, :, mp.Ey].__abs__() ** 2
            area[n_f] = wg.mode_area(x, sim.resolution[0])

        data[n_d, n_w] = np.c_[area, beta]

np.save(
    f"sim_output/02-23-2024/Al2O3/{film_thickness}_t_{0.1}_dstart_{0.1}_dstep_"
    # f"sim_output/02-23-2024/SiO2/{film_thickness}_t_{0.1}_dstart_{0.1}_dstep_"
    + f"{0.5}_wstart_{2.0}_wend_{0.1}_wstep.npy",
    data,
)
