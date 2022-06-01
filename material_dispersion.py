import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import h5py

clipboard_and_style_sheet.style_sheet()

# %% an isotropic medium so simulation size can be of 0 size!
cell = mp.Vector3()
resolution = 20

# We'll use a dispersive material with two polarization terms, just for
# illustration.  The first one is a strong resonance at omega=1.1,
# which leads to a polaritonic gap in the dispersion relation.  The second
# one is a weak resonance at omega=0.5, whose main effect is to add a
# small absorption loss around that frequency.

susceptibilities = [
    mp.LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5),
    mp.LorentzianSusceptibility(frequency=0.5, gamma=0.1, sigma=2e-5)
]

default_material = mp.Medium(epsilon=2.25,
                             E_susceptibilities=susceptibilities)

# %%
fcen = 1.0
df = 2.0
sources = [mp.Source(mp.GaussianSource(fcen,
                                       fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3())]

# %% 454 nm to 3.33 um
kmin = 0.3
kmax = 2.2
k_interp = 99
kpts = mp.interpolate(k_interp, [mp.Vector3(kmin), mp.Vector3(kmax)])

# %%
sim = mp.Simulation(cell_size=cell,
                    geometry=[],
                    sources=sources,
                    default_material=default_material,
                    resolution=resolution)

all_freqs = np.array(sim.run_k_points(200, kpts))

# %%
band1 = all_freqs[:, 0]
band2 = all_freqs[:, 1]

# %%
plt.figure()
plt.plot([i.x for i in kpts], band1.real)
plt.plot([i.x for i in kpts], band2.real)

# %%
# fig, ax = plt.subplots(1, 2)
# freq = np.linspace(0, 2, 5000)
# eps = default_material.epsilon(freq)
# ax[0].plot(freq, eps[:, 0, 0].real)
# ax[1].semilogy(freq, eps[:, 0, 0].imag)
# ax[0].set_ylim(-4, 8)
# [i.set_xlabel("freq") for i in ax]
# ax[0].set_ylabel("real")
# ax[1].set_ylabel("imag")
