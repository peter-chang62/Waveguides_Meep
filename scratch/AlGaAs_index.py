import numpy as np
import matplotlib.pyplot as plt
import clipboard
import scipy.constants as sc

a_0 = lambda x: 6.3 + 19.0 * x
b_0 = lambda x: 9.4 - 10.2 * x
e_0 = lambda x: (1.425 + 1.155 * x + 0.37 * x**2) * sc.eV
d_0 = lambda x: (1.765 + 1.115 * x + 0.37 * x**2) * sc.eV

f = lambda chi: (2 - np.sqrt(1 + chi) - np.sqrt(1 - chi)) / chi**2
chi = lambda wl, x: sc.h * sc.c / wl / e_0(x)
chi_s0 = lambda wl, x: sc.h * sc.c / wl / d_0(x)
n = lambda wl, x: np.sqrt(
    a_0(x) * (f(chi(wl, x) + f(chi_s0(wl, x)) / 2 * (e_0(x) / d_0(x)) ** 1.5)) + b_0(x)
)

# %% ----- test plot looks good!
x = np.arange(0, 1.1, 0.1)
wl = np.linspace(0.4, 2.0, 1000) * 1e-6

fig, ax = plt.subplots(num="AlGaAs index")
for i in range(len(x)):
    ax.plot(wl * 1e6, n(wl, x[i]))
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_ylabel("n")
fig.tight_layout()
