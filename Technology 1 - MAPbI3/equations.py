import numpy as np
import scipy.constants as const


def interpolate_array(array, num_points):
    # Create an array of indices for the input array
    original_indices = np.arange(len(array))

    # Create a new array of indices for the interpolated array
    interpolated_indices = np.linspace(0, len(array) - 1, num_points)

    # Use numpy's interpolation function to interpolate the array
    interpolated_array = np.interp(interpolated_indices, original_indices, array)

    return interpolated_array


def j_g_func(
    spectrum: np.ndarray,
    wav: np.ndarray,
    eqe_from_inp: np.ndarray,
    spectrum_wav: np.ndarray,
) -> float:
    """Function calculates generation current density.
    This function has been modfied from the original version to include a factor of
    lambda / (h*c) in the integral and to do conversions step-by-step
    (see https://doi.org/10.1002/aenm.202100022)

    Args:
        spectrum: Solar spectrum data (y-axis values) in W/m^2/nm.
        wav: Wavelength values used to calculate solar spectrum in m. This should have the same length as eqe and spectrum.
        eqe_from_inp: eqe y-axis values.
        spectrum_wav: wavelength values from the solar spectrum file in m.

    Returns:
        Generation current density in A/m^2.
    """

    q = const.e
    h = const.h
    c = const.c

    # Choose an appropriate range for the spectrum values
    mask = (spectrum_wav >= min(wav)) & (spectrum_wav <= max(wav))
    spectrum = spectrum[mask]
    spectrum_wav = spectrum_wav[mask]

    spectrum = interpolate_array(spectrum, len(wav))
    spectrum_wav = interpolate_array(spectrum_wav, len(wav))

    # Convert units to A/m^2/m for integration
    wav_m = wav
    photon_flux = spectrum * wav_m / (h * c)  # units of photons/m^2/s/nm
    photon_to_amps = q * eqe_from_inp * photon_flux  # units of A/m^2/nm
    photon_to_amps *= 1 / (1e-9)  # convert from nm^-1 to m^-1

    # Cumulative trapezoid method:
    j_g = np.trapz(photon_to_amps, wav)  # Units of A/m^2
    return j_g


def j_o_func(eqe_from_inp: np.ndarray, wav: np.ndarray) -> float:
    """Calculates dark current density.

    Args:
        eqe_from_inp: eqe y-axis values.
        wav: Wavelength values to use to calculate solar spectrum in m.

    Returns:
        Dark current density in A/m^2.
    """

    h = const.h
    c = const.c
    q = const.e
    k = const.k
    pi = const.pi

    # Method from https://doi.org/10.1002/aenm.202100022
    j_o = (
        2
        * 2
        * pi
        * q
        * c
        * np.trapz(
            eqe_from_inp * 1 / wav**4 * 1 / (np.exp(h * c / (wav * k * 300)) - 1),
            wav,
            # dx=wav[1] - wav[0],
        )
    )
    return j_o


def v_oc_func(
    spectrum: np.ndarray, eqe_from_inp: np.ndarray, spectrum_wav: np.ndarray
) -> np.ndarray:
    """Function calculates V_oc for a given eqe spectrum, solar spectrum, and band gap energy.

    Args:
        spectrum: solar spectrum in W/m^2/nm
        eqe_from_inp: eqe x and y values (2D array) (in decimal on y and m on x)
        spectrum_wav: wavelength values from the solar spectrum file in m.


    Returns:
        V_oc values in volts. V_oc has the form of a 2D array. Each row is a different band gap energy. Each column is a different airmass.
    """

    # Define constants
    k = const.k
    q = const.e

    # Calculate V_oc
    v_oc = (k * 300 / q) * np.log(
        j_g_func(spectrum, eqe_from_inp[0], eqe_from_inp[1], spectrum_wav)
        * (j_o_func(eqe_from_inp[1], eqe_from_inp[0])) ** (-1)
        + 1
    )
    return v_oc
