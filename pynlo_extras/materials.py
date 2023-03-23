import numpy as np
import scipy.constants as sc
import pynlo
from pynlo_extras import utilities as util
from scipy.special import erf


# absorption coefficient of PPLN
# TODO I forget where Alex got this from, add a reference to the paper here
# TODO I think you're okay, but check an overall factor of 2 (alpha or 2 * alpha)
#  Connor defines it as twice the imaginary part of the propagation constant
#  swhich I think is consistent with how it's defined in the literature
def ppln_alpha(v_grid):
    """
    This is the absorption coefficient of ppln. When doing DFG and such, you
    deal with pretty low powers in the MIR, so it actually becomes problematic
    if you let the simulation generate arbitrarily long wavelengths. I ended
    uphaving to include the absorption coefficient, which is defined here.

    Args:
        v_grid (1D array):
            frequency grid

    Returns:
        1D array:
            absorption coefficient
    """
    w_grid = v_grid * 2 * np.pi
    w_grid /= 1e12
    return 1e6 * (1 + erf(-(w_grid - 300.0) / (10 * np.sqrt(2))))


def gbeam_area_scaling(z_to_focus, v0, a_eff):
    """
    A gaussian beam can be accounted for by scaling the chi2 and chi3 parameter

    Args:
        z_to_focus (float):
            the distance from the focus
        v0 (float):
            center frequency
        a_eff (float):
            effective area (pi * w_0^2)

    Returns:
        float:
            the ratio of areas^1/2:
                1 / sqrt[ (pi * w^2) / (pi * w_0^2) ]

    Notes:
        returns 1 / (current area / original area)
    """
    w_0 = np.sqrt(a_eff / np.pi)  # beam radius
    wl = sc.c / v0
    z_R = np.pi * w_0**2 / wl  # rayleigh length
    w = w_0 * np.sqrt(1 + (z_to_focus / z_R) ** 2)
    return 1 / (np.pi * w**2 / a_eff)


def chi2_gbeam_scaling(z_to_focus, v0, a_eff):
    """
    scaling for the chi2 parameter for gaussian beam

    Args:
        z_to_focus (float):
            the distance from the focus
        v0 (float):
            center frequency
        a_eff (float):
            effective area (pi * w_0^2)

    Returns:
        float:
            the ratio of areas^1/2:
                1 / sqrt[ (pi * w^2) / (pi * w_0^2) ]

    Notes:
        The chi2 parameter scales as 1 / sqrt[a_eff]
    """
    return gbeam_area_scaling(z_to_focus, v0, a_eff) ** 0.5


def chi3_gbeam_scaling(z_to_focus, v0, a_eff):
    """
    scaling for the chi3 parameter for gaussian beam

    Args:
        z_to_focus (float):
            the distance from the focus
        v0 (float):
            center frequency
        a_eff (float):
            effective area (pi * w_0^2)

    Returns:
        float:
            the ratio of areas^1/2:
                1 / sqrt[ (pi * w^2) / (pi * w_0^2) ]

    Notes:
        The chi3 parameter scales as 1 / a_eff. This is the same as chi2 but
        without the square root
    """
    return gbeam_area_scaling(z_to_focus, v0, a_eff)


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


class PPLN:
    def __init__(self, T=24.5, axis="e"):
        self._T = T
        self._axis = axis

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, val):
        """
        set the temperature in Celsius

        Args:
            val (float):
                the temperature in Celsius
        """
        assert isinstance(val, float)
        self._T = val

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, val):
        """
        set the axis to be either extraordinary or ordinary

        Args:
            val (string):
                either "e" or "o"
        """
        assert np.any([val == "e", val == "o"]), 'the axis must be either "e" or "o"'
        self._axis = val

    @property
    def n(self):
        """
        Returns:
            callable:
                a function that calculates the index of refraction as a
                function of frequency
        """
        return lambda v: n_MgLN_G(v, T=self.T, axis=self.axis)

    @property
    def beta(self):
        """
        Returns:
            callable:
                a function that calculates the angular wavenumber as a function
                of frequency
        """
        # n * omega * c
        return lambda v: n_MgLN_G(v, T=self.T, axis=self.axis) * 2 * np.pi * v / sc.c

    @property
    def d_eff(self):
        """
        d_eff of magnesium doped lithium niobate

        Returns:
            float: d_eff
        """
        return 27e-12  # 27 pm / V

    @property
    def chi2_eff(self):
        """
        effective chi2 of magnesium doped lithium niobate

        Returns:
            float: 2 * d_eff
        """
        return 2 * self.d_eff

    @property
    def chi3_eff(self):
        """
        3rd order nonlinearity of magnesium doped lithium niobate

        Returns:
            float
        """
        return 5200e-24  # 5200 pm ** 2 / V ** 2

    def g2_shg(self, v_grid, v0, a_eff):
        """
        The 2nd order nonlinear parameter weighted for second harmonic
        generation driven by the given input frequency.

        Args:
            v_grid (1D array):
                frequency grid
            v0 (float):
                center frequency
            a_eff (float):
                effective area

        Returns:
            1D array
        """
        return pynlo.utility.chi2.g2_shg(
            v0, v_grid, self.n(v_grid), a_eff, self.chi2_eff
        )

    def g3(self, v_grid, a_eff):
        """
        The 3rd order nonlinear parameter weighted for self-phase modulation.

        Args:
            v_grid (1D array):
                frequency grid
            a_eff (float):
                effective area

        Returns:
            1D array
        """
        n_eff = self.n(v_grid)
        return pynlo.utility.chi3.g3_spm(n_eff, a_eff, self.chi3_eff)

    def generate_model(
        self,
        pulse,
        a_eff,
        length,
        polling_period=None,
        polling_sign_callable=None,
        beta=None,
        alpha=None,
        is_gaussian_beam=False,
    ):
        """
        generate PyNLO model instance

        Args:
            pulse (object):
                PyNLO pulse instance
            a_eff (float):
                effective area
            length (float):
                crystal or waveguide length
            polling_period (None, optional):
                polling period, default is None which is no polling
            polling_sign_callable (None, optional):
                a callable that gives the current polling sign as a function of
                z, default is None
            beta (1D array, optional):
                the beta curve calculated over the pulse's frequency grid. The
                default is None, in which case beta is calculated from Mg-LN's
                material dispersion.
            alpha (1D array, optional):
                the absorption curve calculated over the pulse's frequency
                grid. The default is None in which case MgOLn's material loss
                is used
            is_gaussian_beam (bool, optional):
                whether the mode is a gaussian beam, default is False

        Returns:
            model (object):
                a PyNLO model instance

        Notes:
            polling_period and polling_sign_callable cannot both be provided,
            if "is_gaussian_beam" is set to True, then the chi2 parameter is
            scaled by the ratio of effective areas^1/2 as a function of z

            If is_gaussian_beam is not True, then it is assumed that
            propagation occurs inside a waveguide, in which case an assert
            statement checks that the beta curve was provided
        """
        # --------- assert statements ---------
        assert isinstance(pulse, pynlo.light.Pulse)
        pulse: pynlo.light.Pulse
        assert not np.all(
            [polling_period is not None, polling_sign_callable is not None]
        ), "cannot both set a polling period and a callable function for the polling sign"

        # -------- define polling_sign_callable callable ---------
        # polling period is provided
        if polling_period is not None:
            assert isinstance(polling_period, float)
            polling_sign_callable = lambda z: np.sign(
                np.cos(2 * np.pi * z / polling_period)
            )

        # a callable function for the polling sign was provided
        elif polling_sign_callable is not None:
            assert callable(polling_sign_callable)  # great, we already have it

        # nothing was provided (no polling)
        else:
            polling_sign_callable = lambda z: 1

        # ------ g2 and g3---------
        g2_array = self.g2_shg(pulse.v_grid, pulse.v0, a_eff)
        g3_array = self.g3(pulse.v_grid, a_eff)

        # make g2 and g3 callable if the mode is a gaussian beam
        if is_gaussian_beam:

            def g2_func(z):
                z_to_focus = z - length / 2
                return g2_array * chi2_gbeam_scaling(z_to_focus, pulse.v0, a_eff)

            def g3_func(z):
                z_to_focus = z - length / 2
                return g3_array * chi3_gbeam_scaling(z_to_focus, pulse.v0, a_eff)

            g2 = g2_func
            g3 = g3_func

        else:
            g2 = g2_array
            g3 = g3_array

            assert beta is not None, (
                "if not gaussian beam, waveguide "
                "dispersion needs to be acounted for by providing the beta curve"
            )

        # ----- mode and model ---------
        if beta is None:
            # calculate beta from material dispersion
            beta = self.beta(pulse.v_grid)
        else:
            # beta is already provided
            assert isinstance(beta, np.ndarray) and beta.shape == pulse.v_grid.shape

        if alpha is None:
            # calculate MgOLn's material loss
            alpha = -ppln_alpha(pulse.v_grid)
        else:
            # alpha is already provided
            assert isinstance(alpha, np.ndarray) and alpha.shape == pulse.v_grid.shape

        mode = pynlo.media.Mode(
            pulse.v_grid,
            beta,
            alpha_v=alpha,
            g2_v=g2,  # callable if gaussian beam
            g2_inv=polling_sign_callable,  # callable
            g3_v=g3,  # callable if gaussian beam
            z=0.0,
        )

        model = util.SM_UPE(pulse, mode)
        return model


# -------------------------- fiber parameters ---------------------------------


def Ds_to_beta_n(D, dD_dwl, wl_0):
    """
    convert D, D' to beta2, beta3

    Args:
        D (float):
            D (s / m^2)
        dD_dwl (float):
            D' (s / m^3)
        wl_0 (float):
            center wavelength

    Returns:
        tuple: beta2 (s^2/m), beta3 (s^3/m)

    Notes:
        You can derive the terms below starting from
            D = (-2 pi c / wl^2) beta_2
    """
    # D / (- 2 pi c / wl^2)
    beta_2 = D / (-2 * np.pi * sc.c / wl_0**2)

    # (D' + 2D/wl) / (2 pi c / wl^2)^2
    beta_3 = (dD_dwl + 2 * D / wl_0) * (2 * np.pi * sc.c / wl_0**2) ** -2
    return beta_2, beta_3


def beta_n_to_beta(v0, beta_n):
    """
    get beta(v_grid) from beta_n's

    Args:
        v0 (float):
            center frequency
        beta_n (list of floats):
            list of beta derivatives starting from beta_2

    Returns:
        callable:
            beta(v_grid)

    Notes:
        realize that in literature, and if retrieved from D's that beta
        derivatives are given for beta(2 * np.pi * v_grid), this is taken care
        of here
    """
    beta_omega = pynlo.utility.taylor_series(v0 * 2 * np.pi, [0, 0, *beta_n])

    beta = lambda v_grid: beta_omega(v_grid * 2 * np.pi)
    return beta


class Fiber:
    def __init__(self):
        # Q. Lin and G. P. Agrawal, Raman Response Function for Silica Fibers,
        # Opt. Lett. 31, 3086 (2006).
        self._r_weights = [0.245 * (1 - 0.21), 12.2e-15, 32e-15]
        self._b_weights = [0.245 * 0.21, 96e-15]

        self._beta = None
        self._gamma = None

    @property
    def beta(self):
        assert self._beta is not None, "no beta has been defined yet"
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def gamma(self):
        assert self._gamma is not None, "no gamma has been defined yet"
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def r_weights(self):
        return self._r_weights

    @r_weights.setter
    def r_weights(self, r_weights):
        """
        r_weights : array_like of float
            The contributions due to vibrational resonances in the material. Must
            be given as ``[fraction, tau_1, tau_2]``, where `fraction` is the
            fractional contribution of the resonance to the total nonlinear
            response function, `tau_1` is the period of the vibrational frequency,
            and `tau_2` is the resonance's characteristic decay time. Enter more
            than one resonance using an (n, 3) shaped input array.
        """
        assert len(self._r_weights) == 3
        self._r_weights = r_weights

    @property
    def b_weights(self):
        return self._b_weights

    @b_weights.setter
    def b_weights(self, b_weights):
        """
        b_weights : array_like of float, optional
            The contributions due to boson peaks found in amorphous materials.
            Must be given as ``[fraction, tau_b]``, where `fraction` is the
            fractional contribution of the boson peak to the total nonlinear
            response function, and `tau_b` is the boson peak's characteristic
            decay time. Enter more than one peak using an (n, 2) shaped input
            array. The default behavior is to ignore this term.
        """
        assert len(self._b_weights) == 2
        self._b_weights = b_weights

    def get_beta_from_beta_n(self, v0, beta_n):
        self.beta = beta_n_to_beta(v0, beta_n)

    def get_beta_from_D_n(self, wl_0, D, dD_dwl):
        beta_2, beta_3 = Ds_to_beta_n(D, dD_dwl, wl_0)
        v0 = sc.c / wl_0
        self.get_beta_from_beta_n(v0, [beta_2, beta_3])

    def load_fiber_from_dict(self, dict_fiber, axis="slow"):
        """
        load fiber parameters from the dictionaries below

        Args:
            dict_fiber (dict):
                dict containing fiber parameters, with keys following naming
                convention shown below
            axis (str, optional):
                "slow" or "fast"
        """
        assert np.any([axis == "slow", axis == "fast"])
        assert "center wavelength" in dict_fiber.keys()
        assert "nonlinear coefficient" in dict_fiber.keys()

        if axis == "slow":
            assert "D slow axis" in dict_fiber.keys()
            assert "D slope slow axis" in dict_fiber.keys()

            D = dict_fiber["D slow axis"]
            dD_dwl = dict_fiber["D slope slow axis"]
        if axis == "fast":
            assert "D fast axis" in dict_fiber.keys()
            assert "D slope fast axis" in dict_fiber.keys()

            D = dict_fiber["D fast axis"]
            dD_dwl = dict_fiber["D slope fast axis"]

        wl_0 = dict_fiber["center wavelength"]
        self.get_beta_from_D_n(wl_0, D, dD_dwl)
        self.gamma = dict_fiber["nonlinear coefficient"]

    def g3(self, v_grid, t_shock=None):
        """
        g3 nonlinear parameter

        Args:
            v_grid (1D array):
                frequency grid
            t_shock (float, optional):
                the characteristic time scale of optical shock formation, default is None
                in which case it is taken to be 1 / (2 pi v0)

        Returns:
            g3
        """
        return pynlo.utility.chi3.gamma_to_g3(v_grid, self.gamma, t_shock=t_shock)

    def raman(self, rt_grid, rdt):
        """
        Calculate the normalized frequency-domain Raman and instantaneous
        nonlinear response function.

        This calculates the Raman response in the time domain using approximate,
        analytic equations.

        Parameters
        ----------
        t_grid : array_like of float
            The time grid over which to calculate the nonlinear response function.
            This should be the same time grid as given by `Pulse.rt_grid`.
        dt : float
            The time grid step size. This should be the same time step as given by
            `Pulse.rdt`.

        Returns
        -------
        nonlinear_t : ndarray of float
            The time-domain nonlinear response function. This is defined over the
            same frequency grid as `Pulse.rv_grid`.

        Notes
        -----
        The equations used are the analytical formulations as summarized in
        section 2.3.3 Agrawal's Nonlinear Fiber Optics [1]_. More accurate
        simulations may be obtainable using digitized spectral measurements, such
        as from [2]_.

        References
        ----------
        .. [1] Agrawal GP. Nonlinear Fiber Optics. Sixth ed. London; San Diego,
            CA;: Academic Press; 2019.

            https://doi.org/10.1016/B978-0-12-817042-7.00009-9

        .. [2] R.H. Stolen in "Raman Amplifiers for Telecommunications". See
            figure 2.10 and comments in reference 28.

            https://doi.org/10.1007/978-0-387-21583-9_2

        """
        return pynlo.utility.chi3.nl_response_v(
            rt_grid, rdt, self.r_weights, b_weights=self.b_weights
        )

    def generate_model(self, pulse, t_shock=None):
        """
        generate pynlo.model.SM_UPE instance

        Args:
            pulse (object):
                instance of pynlo.light.Pulse
            t_shock (float, optional):
                time for optical shock formation, defaults to 1 / (2 pi pulse.v0)

        Returns:
            model
        """
        assert isinstance(pulse, pynlo.light.Pulse)
        pulse: pynlo.light.Pulse
        if t_shock is not None:
            assert isinstance(t_shock, float)
        else:
            t_shock = 1 / (2 * np.pi * pulse.v0)

        v_grid = pulse.v_grid
        rt_grid = pulse.rt_grid
        rv_grid = pulse.rv_grid
        rdt = pulse.rdt

        beta = self.beta(v_grid)
        g3 = self.g3(v_grid, t_shock=t_shock)
        raman = self.raman(rt_grid, rdt)

        mode = pynlo.media.Mode(
            v_grid,
            beta,
            g3_v=g3,
            rv_grid=rv_grid,
            r3_v=raman,
            z=0.0,
        )

        return util.SM_UPE(pulse, mode)


# --- unit conversions -----
ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0
dB_to_linear = lambda x: 10 ** (x / 10)

# ---------- OFS fibers ----

# fiber ID 15021110740001
hnlf_2p2 = {
    "D slow axis": 2.2 * ps / (nm * km),
    "D slope slow axis": 0.026 * ps / (nm**2 * km),
    "D fast axis": 1.0 * ps / (nm * km),
    "D slope fast axis": 0.024 * ps / (nm**2 * km),
    "nonlinear coefficient": 10.5 / (W * km),
    "center wavelength": 1550 * nm,
}

# fiber ID 15021110740002
hnlf_5p7 = {
    "D slow axis": 5.7 * ps / (nm * km),
    "D slope slow axis": 0.027 * ps / (nm**2 * km),
    "D fast axis": 5.1 * ps / (nm * km),
    "D slope fast axis": 0.026 * ps / (nm**2 * km),
    "nonlinear coefficient": 10.5 / (W * km),
    "center wavelength": 1550 * nm,
}

# fiber ID 15021110740002
hnlf_5p7_pooja = {
    "D slow axis": 4.88 * ps / (nm * km),
    "D slope slow axis": 0.0228 * ps / (nm**2 * km),
    "D fast axis": 5.1 * ps / (nm * km),
    "D slope fast axis": 0.026 * ps / (nm**2 * km),
    "nonlinear coefficient": 10.9 / (W * km),
    "center wavelength": 1550 * nm,
}

# fiber ID 15021110740002
pm1550 = {
    "D slow axis": 18 * ps / (nm * km),
    "D slope slow axis": 0.0612 * ps / (nm**2 * km),
    "nonlinear coefficient": 1.0 / (W * km),
    "center wavelength": 1550 * nm,
}
