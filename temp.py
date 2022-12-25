"""
You should be able to tell where the dispersive wave will appear given the dispersion and peak power.
You should be able to do this from what you're already doing in load_waveguide() in script-7.py
"""

# %%
import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt

cr.style_sheet()
plt.ion()


# taken from:
# X. Liu, A. S. Svane, J. Lægsgaard, H. Tu, S. A. Boppart, and D. Turchinovich,
# J. Phys. D: Appl. Phys. 49, 023001 (2016).
def dk_func(omega, omega_p, beta, beta_1, gamma, P):
    """Summary

    Args:
        omega (TYPE): frequency
        omega_p (TYPE): center frequency
        beta (TYPE): function beta(omega)
        beta_1 (TYPE): Description
        gamma (TYPE): function dbeta/domega
        P (TYPE): peak power

    Returns:
        TYPE: phase mismatch
    """

    return beta(omega) - beta(omega_p) - beta_1(omega_p) * (omega - omega_p) - gamma * P
