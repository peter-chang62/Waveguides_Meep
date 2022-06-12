import meep as mp
from meep import mpb
import numpy as np
from matplotlib import pyplot as plt

Si = mp.Medium(index=3.45)
wavelength = 1.55
omega = 1 / wavelength
w = 1  # Si width (um)
h = 0.5  # Si height (um)
geometry = [mp.Block(size=mp.Vector3(mp.inf, w, h), material=Si)]
resolution = 32  # pixels/um
sc_y = 2  # supercell width (um)
sc_z = 2  # supercell height (um)
geometry_lattice = mp.Lattice(size=mp.Vector3(0, sc_y, sc_z))

num_modes = 4
# Setup a simulation object
ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    resolution=resolution,
    num_bands=num_modes
)

# ------------------------------------------------- #
# Solve for all the modes at once, then plot
# ------------------------------------------------- #

k = ms.find_k(
    p=mp.EVEN_Z,
    omega=omega,
    band_min=1,
    band_max=num_modes,
    korig_and_kdir=mp.Vector3(1, 0, 0),
    tol=1e-4,
    kmag_guess=omega * 3.45,
    kmag_min=omega * 0.1,
    kmag_max=omega * 4
)
print(wavelength * np.array(k))
eps = ms.get_epsilon()
plt.figure(figsize=(8, 4))
# Plot the E fields
for mode in range(num_modes):
    print('Current band: {}'.format(mode + 1))
    E = ms.get_efield(which_band=mode + 1, bloch_phase=False)
    plt.subplot(1, num_modes, mode + 1)
    plt.imshow(eps.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(np.abs(np.squeeze(E[:, :, 0, 1]).transpose()) ** 2, cmap='RdBu', alpha=0.9)
    plt.axis('off')
    plt.title('$E_{{y{}}}$'.format(mode))
plt.tight_layout()
plt.savefig('errorFields.png')

# ------------------------------------------------- #
# Solve for all the modes one by one
# ------------------------------------------------- #

plt.figure(figsize=(8, 4))
for mode in range(num_modes):
    k = ms.find_k(
        p=mp.EVEN_Z,
        omega=omega,
        band_min=mode + 1,
        band_max=mode + 1,
        korig_and_kdir=mp.Vector3(1, 0, 0),
        tol=1e-4,
        kmag_guess=omega * 3.45,
        kmag_min=omega * 0.1,
        kmag_max=omega * 4
    )
    E = ms.get_efield(which_band=mode + 1, bloch_phase=False)
    plt.subplot(1, num_modes, mode + 1)
    plt.imshow(eps.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(np.abs(np.squeeze(E[:, :, 0, 1]).transpose()) ** 2, cmap='RdBu', alpha=0.9)
    plt.axis('off')
    plt.title('$E_{{y{}}}$'.format(mode))
plt.tight_layout()
plt.savefig('correctFields.png')

plt.show()
