# -*- coding: utf-8 -*-
# Materials Library

import meep as mp

# default unit length is 1 μm
um_scale = 1.0

# conversion factor for eV to 1/μm [=1/hc]
eV_um_scale = um_scale / 1.23984193

# ------------------------------------------------------------------
# Sapphire (Al2O3) from Malitson and Dodge 1972
# ref: https://refractiveindex.info/?shelf=main&book=Al2O3&page=Malitson-o
# ref: https://refractiveindex.info/?shelf=main&book=Al2O3&page=Malitson-e
# wavelength range: 0.2 - 5.0 μm

# NOTE: ordinary (o) axes in X and Y, extraordinary (e) axis in Z

Al2O3_range = mp.FreqRange(min=um_scale / 5.0, max=um_scale / 0.2)

Al2O3_frq1 = 1 / (0.0726631 * um_scale)
Al2O3_gam1 = 0
Al2O3_sig1 = 1.4313493
Al2O3_frq2 = 1 / (0.1193242 * um_scale)
Al2O3_gam2 = 0
Al2O3_sig2 = 0.65054713
Al2O3_frq3 = 1 / (18.028251 * um_scale)
Al2O3_gam3 = 0
Al2O3_sig3 = 5.3414021

Al2O3_susc_o = [
    mp.LorentzianSusceptibility(
        frequency=Al2O3_frq1,
        gamma=Al2O3_gam1,
        sigma_diag=Al2O3_sig1 * mp.Vector3(1, 1, 0),
    ),
    mp.LorentzianSusceptibility(
        frequency=Al2O3_frq2,
        gamma=Al2O3_gam2,
        sigma_diag=Al2O3_sig2 * mp.Vector3(1, 1, 0),
    ),
    mp.LorentzianSusceptibility(
        frequency=Al2O3_frq3,
        gamma=Al2O3_gam3,
        sigma_diag=Al2O3_sig3 * mp.Vector3(1, 1, 0),
    ),
]

Al2O3_frq1 = 1 / (0.0740288 * um_scale)
Al2O3_gam1 = 0
Al2O3_sig1 = 1.5039759
Al2O3_frq2 = 1 / (0.1216529 * um_scale)
Al2O3_gam2 = 0
Al2O3_sig2 = 0.55069141
Al2O3_frq3 = 1 / (20.072248 * um_scale)
Al2O3_gam3 = 0
Al2O3_sig3 = 6.5927379

Al2O3_susc_e = [
    mp.LorentzianSusceptibility(
        frequency=Al2O3_frq1,
        gamma=Al2O3_gam1,
        sigma_diag=Al2O3_sig1 * mp.Vector3(0, 0, 1),
    ),
    mp.LorentzianSusceptibility(
        frequency=Al2O3_frq2,
        gamma=Al2O3_gam2,
        sigma_diag=Al2O3_sig2 * mp.Vector3(0, 0, 1),
    ),
    mp.LorentzianSusceptibility(
        frequency=Al2O3_frq3,
        gamma=Al2O3_gam3,
        sigma_diag=Al2O3_sig3 * mp.Vector3(0, 0, 1),
    ),
]

Al2O3 = mp.Medium(
    epsilon=1.0,
    E_susceptibilities=Al2O3_susc_o + Al2O3_susc_e,
    valid_freq_range=Al2O3_range,
)
