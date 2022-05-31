import matplotlib.pyplot as plt
import meep as mp
import clipboard_and_style_sheet
import numpy as np
import h5py

# %%
cell = mp.Vector3(16, 16, 0)
geometry = [mp.Block(mp.Vector3(12, 1, mp.inf),
                     center=mp.Vector3(-2.5, -3.5),
                     material=mp.Medium(epsilon=12)),
            mp.Block(mp.Vector3(1, 12, mp.inf),
                     center=mp.Vector3(3.5, 2),
                     material=mp.Medium(epsilon=12))]
pml_layers = [mp.PML(1.0)]
resolution = 10

# %%
sources = [mp.Source(mp.ContinuousSource(wavelength=2 * (11 ** 0.5), width=20),
                     component=mp.Ez,
                     center=mp.Vector3(-7, -3.5),
                     size=mp.Vector3(0, 1))]

# %%
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# %%
sim.run(  # mp.at_beginning(mp.output_epsilon),
    mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
    until=200)

# %%
with h5py.File('scratch1-ez.h5', 'r') as f:
    data = np.array(f[list(f.keys())[0]])

# %%
fig, ax = plt.subplots(1, 1)
for n in range(data.shape[-1]):
    ax.clear()
    ax.imshow(data[:, :, n].T)
    plt.pause(.01)
