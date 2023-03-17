# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
from TFW_meep import waveguide_dispersion as wg
from TFW_meep import materials as mtp
from meep import materials as mt
import meep as mp
import pynlo_extras as pe
from scipy import constants as sc


# %% dimensions
etch_width = 2
etch_depth = 0.7
film_thickness = 1

wl_min = 4.0
wl_max = 5.0

# %% materials
ppln = pe.materials.PPLN()


def eps_func_wvgd(freq):
    um = 1e-6

    nu = freq * sc.c / um
    return ppln.n(nu) ** 2


substrate_medium = mtp.Al2O3
waveguide_medium = mt.LiNbO3

# %% create waveguide
TFW = wg.ThinFilmWaveguide(
    etch_width,
    etch_depth,
    film_thickness,
    substrate_medium,
    waveguide_medium,
    resolution=30,
    cell_width=8,
    cell_height=6,
    num_bands=1,
)


# %% run simulation to obtain a list of k_points the unguided modes are those
#    that lie above the light line, which are the ones that we want to solve
#    for propagation loss
res = TFW.calc_dispersion(
    wl_min, wl_max, 19, eps_func_wvgd=eps_func_wvgd, eps_func_sbstrt=None
)

# ----------------------- propagation loss calculation ------------------------
(unguided,) = (res.freq > np.squeeze(res.kx) / res._index_sbstrt).nonzero()
unguided = np.arange(len(unguided) + 2)  # add a few more
dispersion = np.c_[res.freq[unguided], np.squeeze(res.kx)[unguided]]

# %% get sim
sim = TFW.sim
pml_layers = [
    mp.PML(thickness=1, direction=mp.Z),
    mp.PML(thickness=1, direction=mp.Y),
]
sim.boundary_layers = pml_layers

# %% run simulation for each k_point
FREQ = []
DECAY = []
for fcen, kx in dispersion[::-1]:
    # reset
    sim.reset_meep()

    # sources
    src = mp.GaussianSource(frequency=fcen, width=5)
    sources = [mp.Source(src, mp.Ey, center=mp.Vector3())]
    sim.sources = sources

    # set waveguide epsilon
    eps_wvgd = eps_func_wvgd(fcen)
    TFW.blk_wvgd.material = mp.Medium(epsilon=eps_wvgd)
    TFW._blk_film.material = mp.Medium(epsilon=eps_wvgd)

    # set substrate epsilon
    eps_sbstrt = TFW.sbstrt_mdm.epsilon(fcen)[2, 2]
    TFW.blk_sbstrt.material = mp.Medium(epsilon=eps_sbstrt)
    sim.geometry = TFW.geometry

    # harminv monitoring: offset from the symmetry plane slightly in the
    # horizontal direction
    df = 100e-3 / (1 / fcen) ** 2  # 100 nm
    h = mp.Harminv(mp.Ey, mp.Vector3(0, 0.1234, 0), fcen, df)

    k_point = mp.Vector3(kx, 0, 0)
    sim.k_point = k_point
    sim.run(mp.after_sources(h), until_after_sources=300)

    if len(h.modes) == 0:
        break
    else:
        assert len(h.modes) == 1
        (mode,) = h.modes
        re = mode.freq
        im = mode.decay
        FREQ.append(re)
        DECAY.append(abs(im))

# %% sort stuff out

# guided starts from the next one after the last in unguided
guided = np.arange(len(res.freq))[unguided[-1] + 1 :]
guided_freq = res.freq[guided]
unguided_freq = res.freq[unguided]

# truncate at frequencies below which loss was too high to calculate
# index high -> low and truncate at low, then re-index low -> high
unguided_freq = unguided_freq[::-1][: len(DECAY)][::-1]

# DECAY was indexed high -> low
decay = np.zeros(len(unguided_freq))
decay[:] = np.asarray(DECAY)
decay = decay[::-1]
decay = np.append(decay, np.zeros(len(guided_freq)))

# savable simulation data!
N_throw = len(res.freq) - len(decay)
arr = np.c_[res.freq[N_throw:], res.kx.flatten()[N_throw:], decay]
