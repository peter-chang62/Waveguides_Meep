import pynlo
import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from pynlo_extras import light


def dispersive_wave_dk(w, w_p, b_w, b_w_p, b_1_w_p, gamma=0, P=0):
    return b_w - b_w_p - b_1_w_p * (w - w_p) - gamma * P / 2


def estimate_step_size(model, local_error=1e-6):
    """
    estimate the step size for PyNLO simulation, this is the same as connor's
    default call except n=20 instead of n=10 (just following some of his
    example files)

    Args:
        model (object):
            instance of pynlo.model.SM_UPE
        local_error (float, optional):
            local error, default is 10^-6

    Returns:
        float:
            estimated step size
    """
    model: pynlo.model.SM_UPE
    dz = model.estimate_step_size(n=20, local_error=local_error)
    return dz


def z_grid_from_polling_period(polling_period, length):
    """
    Generate the z grid points from a fixed polling period. The grid points are
    all the inversion points. I think this is important if including polling
    in a crystal to make sure that it doesn't "miss" any of the
    quasi-phasematching

    Args:
        polling_period (float):
            The polling period
        length (float):
            The length of the crystal / waveguide

    Returns:
        1D array: the array of z grid points
    """
    cycle_period = polling_period / 2.0
    n_cycles = np.ceil(length / cycle_period)
    z_grid = np.arange(0, n_cycles * cycle_period, cycle_period)
    z_grid = np.append(z_grid[z_grid < length], length)
    return z_grid


def plot_results(pulse_out, z, a_t, a_v, plot="frq"):
    """
    plot PyNLO simulation results

    Args:
        pulse_out (object):
            pulse instance that is used for the time and frequency grid
            (so actually could also be input pulse)
        z (1D array): simulation z grid points
        a_t (2D array): a_t at each z grid point
        a_v (2D array): a_v at each z grid point
        plot (string, optional):
            whether to plot the frequency domain with frequency or wavelength
            on the x axis, default is frequency
    """
    pulse_out: pynlo.light.Pulse
    assert np.any([plot == "frq", plot == "wvl"]), "plot must be 'frq' or 'wvl'"

    fig = plt.figure("Simulation Results", clear=True)
    ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, sharex=ax0)
    ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2, sharex=ax1)

    p_v_dB = 10 * np.log10(np.abs(a_v) ** 2)
    p_v_dB -= p_v_dB.max()
    if plot == "frq":
        ax0.plot(1e-12 * pulse_out.v_grid, p_v_dB[0], color="b")
        ax0.plot(1e-12 * pulse_out.v_grid, p_v_dB[-1], color="g")
        ax2.pcolormesh(
            1e-12 * pulse_out.v_grid,
            1e3 * z,
            p_v_dB,
            vmin=-40.0,
            vmax=0,
            shading="auto",
        )
        ax0.set_ylim(bottom=-50, top=10)
        ax2.set_xlabel("Frequency (THz)")
    elif plot == "wvl":
        wl_grid = sc.c / pulse_out.v_grid
        ax0.plot(1e6 * wl_grid, p_v_dB[0], color="b")
        ax0.plot(1e6 * wl_grid, p_v_dB[-1], color="g")
        ax2.pcolormesh(
            1e6 * wl_grid,
            1e3 * z,
            p_v_dB,
            vmin=-40.0,
            vmax=0,
            shading="auto",
        )
        ax0.set_ylim(bottom=-50, top=10)
        ax2.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")

    p_t_dB = 10 * np.log10(np.abs(a_t) ** 2)
    p_t_dB -= p_t_dB.max()
    ax1.plot(1e12 * pulse_out.t_grid, p_t_dB[0], color="b")
    ax1.plot(1e12 * pulse_out.t_grid, p_t_dB[-1], color="g")
    ax3.pcolormesh(
        1e12 * pulse_out.t_grid, 1e3 * z, p_t_dB, vmin=-40.0, vmax=0, shading="auto"
    )
    ax1.set_ylim(bottom=-50, top=10)
    ax3.set_xlabel("Time (ps)")

    ax0.set_ylabel("Power (dB)")
    ax2.set_ylabel("Propagation Distance (mm)")
    fig.tight_layout()
    fig.show()


def animate(pulse_out, model, z, a_t, a_v, plot="frq", save=False, p_ref=None):
    """
    replay the real time simulation

    Args:
        pulse_out (object):
            reference pulse instance for time and frequency grid
        model (object):
            pynlo.model.SM_UPE instance used in the simulation
        z (1D array):
            z grid returned from the call to model.simulate()
        a_t (2D array):
            time domain electric fields returned from the call to
            model.simulate()
        a_v (TYPE):
            frequency domain electric fields returned from the call to
            model.simulate()
        plot (str, optional):
            "frq", "wvl" or "time"
        save (bool, optional):
            save figures to fig/ folder, default is False (see ezgif.com)
        p_ref (pulse instance, optional):
            a reference pulse to overlay all the plots, useful if you have a
            measured spectrum to compare against to
    """
    assert np.any(
        [plot == "frq", plot == "wvl", plot == "time"]
    ), "plot must be 'frq' or 'wvl'"
    assert isinstance(pulse_out, pynlo.light.Pulse)
    assert isinstance(model, pynlo.model.SM_UPE)
    assert isinstance(p_ref, pynlo.light.Pulse) or p_ref is None
    pulse_out: pynlo.light.Pulse
    model: pynlo.model.SM_UPE
    p_ref: pynlo.light.Pulse

    fig, ax = plt.subplots(2, 1, num="Replay of Simulation", clear=True)
    ax0, ax1 = ax

    wl_grid = sc.c / pulse_out.v_grid

    p_v = abs(a_v) ** 2
    p_t = abs(a_t) ** 2
    phi_t = np.angle(a_t)
    phi_v = np.angle(a_v)

    vg_t = pulse_out.v_ref + np.gradient(
        np.unwrap(phi_t) / (2 * np.pi), pulse_out.t_grid, edge_order=2, axis=1
    )
    tg_v = pulse_out.t_ref - np.gradient(
        np.unwrap(phi_v) / (2 * np.pi), pulse_out.v_grid, edge_order=2, axis=1
    )

    for n in range(len(a_t)):
        [i.clear() for i in [ax0, ax1]]

        if plot == "time":
            ax0.semilogy(pulse_out.t_grid * 1e12, p_t[n], ".", markersize=1)
            ax1.plot(
                pulse_out.t_grid * 1e12,
                vg_t[n] * 1e-12,
                ".",
                markersize=1,
                label=f"z = {np.round(z[n] * 1e3, 3)} mm",
            )

            ax0.set_title("Instantaneous Power")
            ax0.set_ylabel("J / s")
            ax0.set_xlabel("Delay (ps)")
            ax1.set_ylabel("Frequency (THz)")
            ax1.set_xlabel("Delay (ps)")

            excess = 0.05 * (pulse_out.v_grid.max() - pulse_out.v_grid.min())
            ax0.set_ylim(top=max(p_t[n] * 1e1), bottom=max(p_t[n] * 1e-9))
            ax1.set_ylim(
                top=1e-12 * (pulse_out.v_grid.max() + excess),
                bottom=1e-12 * (pulse_out.v_grid.min() - excess),
            )

        if plot == "frq":
            ax0.semilogy(pulse_out.v_grid * 1e-12, p_v[n], ".", markersize=1)
            ax1.plot(
                pulse_out.v_grid * 1e-12,
                tg_v[n] * 1e12,
                ".",
                markersize=1,
                label=f"z = {np.round(z[n] * 1e3, 3)} mm",
            )

            if p_ref is not None:
                ax0.semilogy(p_ref.v_grid * 1e-12, p_ref.p_v, ".", markersize=1)

            ax0.set_title("Power Spectrum")
            ax0.set_ylabel("J / Hz")
            ax0.set_xlabel("Frequency (THz)")
            ax1.set_ylabel("Delay (ps)")
            ax1.set_xlabel("Frequency (THz)")

            excess = 0.05 * (pulse_out.t_grid.max() - pulse_out.t_grid.min())
            ax0.set_ylim(top=max(p_v[n] * 1e1), bottom=max(p_v[n] * 1e-9))
            ax1.set_ylim(
                top=1e12 * (pulse_out.t_grid.max() + excess),
                bottom=1e12 * (pulse_out.t_grid.min() - excess),
            )

        if plot == "wvl":
            ax0.semilogy(wl_grid * 1e6, p_v[n] * model.dv_dl, ".", markersize=1)
            ax1.plot(
                wl_grid * 1e6,
                tg_v[n] * 1e12,
                ".",
                markersize=1,
                label=f"z = {np.round(z[n] * 1e3, 3)} mm",
            )

            if p_ref is not None:
                ax0.semilogy(
                    p_ref.wl_grid * 1e6, p_ref.p_v * model.dv_dl, ".", markersize=1
                )

            ax0.set_title("Power Spectrum")
            ax0.set_ylabel("J / m")
            ax0.set_xlabel("Wavelength ($\\mathrm{\\mu m}$)")
            ax1.set_ylabel("Delay (ps)")
            ax1.set_xlabel("Wavelength ($\\mathrm{\\mu m}$)")

            excess = 0.05 * (pulse_out.t_grid.max() - pulse_out.t_grid.min())
            ax0.set_ylim(
                top=max(p_v[n] * model.dv_dl * 1e1),
                bottom=max(p_v[n] * model.dv_dl * 1e-9),
            )
            ax1.set_ylim(
                top=1e12 * (pulse_out.t_grid.max() + excess),
                bottom=1e12 * (pulse_out.t_grid.min() - excess),
            )

        ax1.legend(loc="upper center")
        if n == 0:
            fig.tight_layout()

        if save:
            plt.savefig(f"fig/{n}.png")
        else:
            plt.pause(0.05)


def package_sim_output(simulate):
    def wrapper(self, *args, **kwargs):
        pulse_out, z, a_t, a_v = simulate(self, *args, **kwargs)
        model = self

        class result:
            def __init__(self):
                self.pulse_out = light.Pulse.clone_pulse(pulse_out)
                self.z = z
                self.a_t = a_t
                self.a_v = a_v
                self.p_t = abs(a_t) ** 2
                self.p_v = abs(a_v) ** 2
                self.model = model

            def animate(self, plot, save=False, p_ref=None):
                animate(
                    self.pulse_out,
                    self.model,
                    self.z,
                    self.a_t,
                    self.a_v,
                    plot=plot,
                    save=save,
                    p_ref=p_ref,
                )

            def plot(self, plot):
                plot_results(self.pulse_out, self.z, self.a_t, self.a_v, plot=plot)

            def save(self, path, filename):
                assert path != "" and isinstance(path, str), "give a save path"
                assert filename != "" and isinstance(filename, str)

                path = path + "/" if path[-1] != "" else path
                np.save(path + filename + "_t_grid.npy", self.pulse_out.t_grid)
                np.save(path + filename + "_v_grid.npy", self.pulse_out.v_grid)
                np.save(path + filename + "_z.npy", self.z)
                np.save(path + filename + "_amp_t.npy", abs(self.pulse_out.a_t))
                np.save(path + filename + "_amp_v.npy", abs(self.pulse_out.a_v))
                np.save(path + filename + "_phi_t.npy", np.angle(self.pulse_out.a_t))
                np.save(path + filename + "_phi_v.npy", np.angle(self.pulse_out.a_v))

        return result()

    return wrapper


class SM_UPE(pynlo.model.SM_UPE):

    """
    This is the same as connor's SM_UPE but with the package_sim_output wrapper
    for the simulate call
    """

    def __init__(self, pulse, mode):
        super().__init__(pulse, mode)

    @package_sim_output
    def simulate(self, z_grid, dz=None, local_error=1e-6, n_records=None, plot=None):
        return super().simulate(
            z_grid, dz=dz, local_error=local_error, n_records=n_records, plot=plot
        )

    @property
    def dispersive_wave_dk(self):
        mode = self.mode
        pulse = self.pulse_in
        w_p = pulse.v0 * 2 * np.pi
        w = self.w_grid

        b_w = mode.beta()
        b_w_p = spi.interp1d(w, b_w, bounds_error=True)(w_p)

        b_1_w = mode.beta(m=1)
        b_1_w_p = spi.interp1d(w, b_1_w, bounds_error=True)(w_p)

        gamma = pynlo.utility.chi3.g3_to_gamma(pulse.v_grid, self.g3)
        P = pulse.p_t.max()

        return dispersive_wave_dk(w, w_p, b_w, b_w_p, b_1_w_p, gamma=gamma, P=P)
