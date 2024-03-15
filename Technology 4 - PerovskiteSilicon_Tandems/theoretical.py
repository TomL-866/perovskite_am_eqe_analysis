import os
import numpy as np
from scipy import constants as const


def EQE(E_G, wav):
    """Calculates EQE step function for a given band gap energy.

    Args:
        E_G (array): Band gap energy in Joules
        wav (array): Wavelength values to use to calculate solar spectrum in m. This should have the same length as EQE and spectrum.

    Returns:
        2D array: EQE step function.
    """

    h = const.h
    c = const.c

    EQE = np.zeros(shape=(len(E_G), len(wav)))

    hc_wav = h * c / wav

    # Vectorised operations are much faster here
    E_G = E_G.reshape(-1, 1)
    hc_wav = hc_wav.reshape(1, -1)
    EQE = np.where(hc_wav > E_G, 1, 0)

    # Slower method that accomplishes the same thing:
    # for i in range(len(E_G)):
    #     for x in range(len(wav)):
    #         if h * c / wav[x] > E_G[i]:
    #             EQE[i, x] = 1
    #         else:
    #             EQE[i, x] = 0

    return EQE


def J_G(spectrum, wav, E_G, EQE):
    # Define constants
    q = const.e
    h = const.h
    c = const.c

    wav_m = wav

    photon_flux = spectrum * wav_m / (h * c)  # units of photons/m^2/s/nm
    photon_flux *= 1 / (1e-9)  # convert from nm^-1 to m^-1

    # Integrate to get 1 J_G value for each band gap energy
    J_G = np.zeros(len(E_G))
    for i in range(len(E_G)):
        J_G[i] = np.trapz(q * photon_flux * EQE[i, :], wav)
    return J_G


def J_o(EQE, wav, E_G):

    q = const.e

    J_o = np.zeros(len(wav))

    # Old equation (BB explicity put in)
    # for i in range(len(E_G)):
    #     J_o[i] = q * integrate.simpson(EQE[i, :] * BB, wav, dx=wav[1] - wav[0])

    for i in range(len(E_G)):
        J_o[i] = (
            2
            * 2
            * const.pi
            * q
            * const.c
            * np.trapz(
                EQE[i, :]
                * 1
                / wav**4
                * 1
                / (np.exp(const.h * const.c / (wav * const.k * 300)) - 1),
                wav,
            )
        )

    return J_o


def V_oc(J_G, J_o):
    k = const.k
    q = const.e

    return (k * 300 / q) * np.log(J_G / J_o + 1)


def return_values_AM_any(airmass_num: float) -> tuple[float, float, float, float]:
    base_dir = os.path.dirname(__file__)

    AM = np.loadtxt(
        f"{base_dir}/Solar Spectra by Airmass/AM{airmass_num}.txt",
        skiprows=1,
    )

    E_G = np.linspace(1, 5, len(AM[:, 0])) * const.e
    T = 300
    wav = AM[:, 0] * 1e-9

    spectrum = AM[:, 3]  # Units of W/m^2/nm
    V_oc_value = V_oc(
        J_G(spectrum, wav, E_G, EQE(E_G, wav)),
        J_o(EQE(E_G, wav), wav, E_G),
    )
    J_G_value = J_G(spectrum, wav, E_G, EQE(E_G, wav))
    J_o_value = J_o(EQE(E_G, wav), wav, E_G)

    return (V_oc_value, E_G / const.e, J_G_value, J_o_value)
