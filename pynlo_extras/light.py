import pynlo
import numpy as np
import scipy
import scipy.constants as sc
import scipy.interpolate as spi
from pynlo_extras import python_phase_retrieval as pr


class Pulse(pynlo.light.Pulse):
    def __init__(self, n_points, v_max, dv, v0=None, a_v=None):
        super().__init__(n_points, v_max, dv, v0=v0, a_v=a_v)

    @classmethod
    def Sech(cls, n_points, v_min, v_max, v0, e_p, t_fwhm, min_time_window):
        """
        Initialize a squared hyperbolic secant pulse.

        Args:
            n_points (int):
                number of points on the time and frequency grid
            v_min (float):
                minimum frequency
            v_max (float):
                maximum frequency
            v0 (float):
                center frequency
            e_p (float):
                pulse energy
            t_fwhm (float):
                pulse duration (full width half max)
            min_time_window (float):
                time bandwidth

        Returns:
            object: pulse instance

        Notes:
            everything should be given in mks units

            v_min, v_max, and v0 set the desired limits of the frequency grid.
            min_time_window is used to set the desired time bandwidth. The
            product of the time and frequency bandwidth sets the minimum
            number of points. If the number of points is less than the minimum
            then the number of points is updated.

            Note that connor does not allow negative frequencies unlike PyNLO.
            So, if v_min is negative, then the whole frequency grid gets
            shifted up so that the first frequency bin occurs one dv away from
            DC (excludes the origin).
        """

        pulse: pynlo.light.Pulse
        bandwidth_v = v_max - v_min
        n_points_min = int(np.ceil(min_time_window * bandwidth_v))
        n_points_min = scipy.fftpack.next_fast_len(n_points_min)  # faster fft's
        if n_points_min > n_points:
            msg = (
                f"changing n_points from {n_points} to {n_points_min} to"
                " support both time and frequency bandwidths"
            )
            print(msg)
            n_points = n_points_min
        else:
            n_points_fast = scipy.fftpack.next_fast_len(n_points)
            if n_points_fast != n_points:
                msg = (
                    f"changing n_points from {n_points} to {n_points_fast}"
                    " for faster fft's"
                )
                print(msg)
                n_points = n_points_fast

        # from here it is the same as the Sech classmethod from
        # pynlo.light.Pulse, with the addition of a default call to rtf_grids
        pulse = super().Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
        pulse: Pulse
        pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing

        return pulse

    @property
    def wl_grid(self):
        """
        wavelength axis

        Returns:
            1D array:
                wavelength axis
        """
        return sc.c / self.v_grid

    def chirp_pulse_W(self, *chirp, v0=None):
        """
        chirp a pulse

        Args:
            *chirp (float):
                any number of floats representing gdd, tod, fod ... in seconds
            v0 (None, optional):
                center frequency for the taylor expansion, default is v0 of the
                pulse
        """
        assert len(chirp) > 0
        assert [isinstance(i, float) for i in chirp]

        if v0 is None:
            v0 = self.v0
        else:
            assert np.all([isinstance(v0, float), v0 > 0])

        v_grid = self.v_grid - v0
        w_grid = v_grid * 2 * np.pi

        factorial = np.math.factorial
        phase = 0
        for n, c in enumerate(chirp):
            n += 2  # start from 2
            phase += (c / factorial(n)) * w_grid**n
        self.a_v *= np.exp(1j * phase)

    def import_p_v(self, v_grid, p_v, phi_v=None):
        """
        import experimental spectrum

        Args:
            v_grid (1D array of floats):
                frequency grid
            p_v (1D array of floats):
                power spectrum
            phi_v (1D array of floats, optional):
                phase, default is transform limited, you would set this
                if you have a frog retrieval, for example
        """
        p_v = np.where(p_v > 0, p_v, 1e-100)
        amp_v = p_v**0.5
        amp_v = spi.interp1d(
            v_grid, amp_v, kind="cubic", bounds_error=False, fill_value=1e-100
        )(self.v_grid)

        if phi_v is not None:
            assert isinstance(phi_v, np.ndarray) and phi_v.shape == p_v.shape
            phi_v = spi.interp1d(
                v_grid, phi_v, kind="cubic", bounds_error=False, fill_value=0.0
            )(self.v_grid)
        else:
            phi_v = 0.0

        a_v = amp_v * np.exp(1j * phi_v)

        e_p = self.e_p
        self.a_v = a_v
        self.e_p = e_p

    @classmethod
    def clone_pulse(cls, pulse):
        """
        clone a pulse instance

        Args:
            pulse (Pulse instance)

        Returns:
            pulse
        """
        assert isinstance(pulse, pynlo.light.Pulse)
        pulse: pynlo.light.Pulse
        n_points = pulse.n
        v_min = pulse.v_grid[0]
        v_max = pulse.v_grid[-1]
        v0 = pulse.v0
        e_p = pulse.e_p
        time_window = np.diff(pulse.t_grid[[0, -1]])
        t_fwhm = 200e-15  # only affects power spectrum in the Sech call

        p = cls.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm, time_window)
        p.a_v[:] = pulse.a_v[:]
        return p

    def calculate_spectrogram(self, t_grid):
        """
        calculate a spectrogram for an input time delay axis

        Args:
            t_grid (1D array of floats):
                time delay axis

        Returns:
            v_grid (1D array), spectrogram (2D array):
                the calculated spectrogram, with time indexing row, and
                frequency indexing the column

        Notes:
            I have had issues with shifting using fft's if the power spectrum
            is not centered on the frequency grid. So, here I use the Pulse
            instance taken from python_phase_retrieval.py. Since the frequency
            grid there is different, I return the frequency grid here for
            reference
        """
        p = pr.Pulse.clone_pulse(self)
        s = pr.calculate_spectrogram(p, t_grid)
        ind = np.logical_and(self.v_grid.min() < p.v_grid, p.v_grid < self.v_grid.max())
        return p.v_grid[ind], s[:, ind]
