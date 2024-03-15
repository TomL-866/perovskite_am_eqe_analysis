import typing
import csv
import os
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import equations as eqs
import theoretical as theory


def get_csv_file_list(directory: str) -> list[str]:
    return sorted([f for f in os.listdir(directory) if f.endswith(".csv")])


def get_technology_name() -> str:
    technology_name = os.path.dirname(__file__).split("/")[-1].split("-")[-1]
    technology_name = technology_name.replace(" ", "")

    return technology_name


def load_EQE_data() -> (
    list[list[np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]]]
):
    """This function loads in EQE data from .csv files
    located in the "EQE Data" folder.

    Returns:
        EQE data in the form of a list of lists.
    """

    # Use os.path.dirname(__file__) so code can be run from any directory
    dir_eqe = os.path.dirname(__file__) + "/EQE Data"
    csv_files = get_csv_file_list(dir_eqe)

    EQE_raw_data = {}
    EQE = []

    for file in csv_files:
        key = file.replace(".csv", "")

        EQE_raw_data[key] = np.genfromtxt(os.path.join(dir_eqe, file), delimiter=",")

        # Apply unit conversions before appending
        EQE.append([EQE_raw_data[key][:, 0] * 1e-9, EQE_raw_data[key][:, 1] * 1 / 100])

    for _, eqe_enum in enumerate(EQE):
        # Interpolating EQE in order to increase the amount of data points
        func = interp1d(eqe_enum[0], eqe_enum[1])
        wav_range_for_interp = np.linspace(
            np.min(eqe_enum[0]), np.max(eqe_enum[0]), 3000
        )
        eqe_interpolated = func(wav_range_for_interp)
        eqe_enum[1] = eqe_interpolated
        eqe_enum[0] = wav_range_for_interp

    return EQE


def load_spectral_data() -> (
    list[np.ndarray[typing.Any, np.dtype[np.floating[np._typing._64Bit]]]]
):
    """This function loads in spectral data from .txt files
    located in the "Solar Spectra by Airmass" folder.

    Returns:
        Solar spectral data in the form of a list of lists.
    """
    # Load in spectral data
    AM = [
        np.loadtxt(
            f"{os.path.dirname(__file__)}/Solar Spectra by Airmass/AM{1.0 + i * 0.25}.txt",
            skiprows=1,
        )
        for i in range(5)
    ]

    return AM


def generate_E_G_data(EQE: np.ndarray) -> list[typing.Any]:
    """This function generates E_G data by differentiating EQE spectra
    and finding the maximum value of the differentiated EQE spectra.

    Args:
        EQE: EQE data

    Returns:
        Band gaps for each EQE spectrum (eV)
    """
    E_G = []

    # Calculate E_G from EQE spectra

    EQE_diff = [
        np.diff(EQE[i][1]) / np.diff(const.h * const.c / EQE[i][0])
        for i in range(len(EQE))
    ]  # x axis should be in units of Joules

    for i, (eqe, eqe_diff) in enumerate(zip(EQE, EQE_diff)):
        # Find maximum index and wavelength
        max_index = np.argmax(eqe_diff)
        wavelength = eqe[0][max_index]

        # Convert this wavelength into energy using the formula E = h*c/λ
        # Convert this energy from Joules to eV
        energy = const.h * const.c / wavelength / const.e

        E_G.append(energy)

    return E_G


def calculate_V_oc(
    EQE: np.ndarray, AM: np.ndarray
) -> list[list[np.ndarray[typing.Any, typing.Any]]]:
    """This function calculates V_oc for each EQE spectrum and each AM value.

    Args:
        EQE: EQE data
        AM: Solar spectrum by airmass

    Returns:
        Open circuit voltage for each EQE spectrum and each AM value (V)
    """

    V_oc = [
        [eqs.v_oc_func(AM[i][:, 3], EQE[x], AM[i][:, 0] * 1e-9) for i in range(len(AM))]
        for x in range(len(EQE))  # A list of 5 elements is created for each AM
    ]

    return V_oc


def calculate_J_G(EQE: np.ndarray, AM: np.ndarray) -> list[float]:
    """This function calculates J_G for each EQE spectrum and each AM value.

    Args:
        EQE: EQE data
        AM: Solar spectrum by airmass

    Returns:
        Generation current density for each EQE spectrum and each AM value (A/m^2)
    """
    J_G = [
        eqs.j_g_func(AM[i][:, 3], EQE[x][0], EQE[x][1], AM[i][:, 0] * 1e-9)
        for i in range(len(AM))
        for x in range(len(EQE))
    ]  # 5 values are created for each AM

    return J_G


def calculate_J_o(EQE: np.ndarray[typing.Any, typing.Any]) -> list[float]:
    """This function calculates J_o for each EQE spectrum.

    Args:
        EQE: EQE data

    Returns:
        Dark current density for each EQE spectrum (A/m^2)
    """

    J_o = [
        eqs.j_o_func(EQE[x][1], EQE[x][0]) for x in range(len(EQE))
    ]  # Length of J_o is the same as the length of E_G. One J_o value is found for each band gap energy.

    return J_o


def get_sorted_dataframe_with_reported_E_G() -> pd.core.frame.DataFrame:
    """This function extracts the relevant data from the technology_name_data.csv file

    Returns:
        Filtered and sorted dataframe
    """

    dois = []

    dir_eqe = os.path.dirname(__file__) + "/EQE Data"
    csv_files = get_csv_file_list(dir_eqe)

    for file in csv_files:
        parts = file.split("_")
        doi_with_extension = parts[2]
        # DOIs come after the second underscore and before the file extension.
        doi = doi_with_extension.rsplit(".", 1)[0]
        dois.append(doi)

    technology_name = get_technology_name()

    df = pd.read_csv(f"{os.path.dirname(__file__)}/{technology_name}_data.csv")

    df["Ref_DOI_number"] = df["Ref_DOI_number"].str.replace("/", "")

    df = df[df["Ref_DOI_number"].isin(dois)]

    # Sort df so that the rows are in the same order as the csv files
    # in the EQE Data folder
    df["correct_file_order"] = pd.Categorical(
        df["Ref_DOI_number"], categories=dois, ordered=True
    )
    df = df.sort_values("correct_file_order")
    df = df.reset_index(drop=True)

    # Only keeping the highest reported J_sc
    df = df.drop_duplicates(subset="Ref_DOI_number", keep="first")

    return df


def get_reported_V_oc(sorted_dataframe: pd.core.frame.DataFrame):
    """This function extracts the reported V_oc values from the filtered data.

    Args:
        sorted_dataframe: Filtered data

    Returns:
        Reported open circuit voltage for each EQE spectrum (V)
    """
    df = sorted_dataframe

    V_oc_reported = df["JV_default_Voc"].to_numpy()

    return V_oc_reported


def get_reported_J_sc(sorted_dataframe):
    """This function extracts the reported J_sc (= J_G) values from the filtered data.

    Args:
        sorted_dataframe: Filtered data

    Returns:
        Reported J_sc for each EQE spectrum (A/m^2)
    """
    df = sorted_dataframe

    J_sc_reported = (
        df["JV_default_Jsc"].to_numpy() * 10
    )  # Working in SI right up until plotting

    return J_sc_reported


def save_V_oc_values(
    V_oc: np.ndarray,
    V_oc_reported: np.ndarray,
    AM: np.ndarray,
    E_G: np.ndarray,
    reported_E_G: np.ndarray,
) -> None:
    """This function saves the values used to make the V_oc plot to a .csv file."""

    technology_name = get_technology_name()

    directory_outputs = f"{os.path.dirname(__file__)}/Output"
    os.makedirs(directory_outputs, exist_ok=True)

    with open(
        f"{directory_outputs}/{technology_name}_V_oc_values.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "AM",
                "Band gap energy (eV)",
                "Reported band gap energy (eV)",
                "Simulated V_oc (V)",
                "Reported V_oc (V)",
            ]
        )
        for x, _ in enumerate(AM):
            for i, band_gap in enumerate(E_G):
                if (1 + x * 0.25) == 1.5:
                    writer.writerow(
                        [
                            1.0 + x * 0.25,
                            band_gap,
                            reported_E_G[i],
                            V_oc[i][x],
                            V_oc_reported[i],
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            1.0 + x * 0.25,
                            band_gap,
                            "N/A",
                            V_oc[i][x],
                            "N/A",
                        ]
                    )


def save_J_sc_values(
    J_G: np.ndarray,
    J_sc_reported: np.ndarray,
    AM: np.ndarray,
    E_G: np.ndarray,
    reported_E_G: np.ndarray,
) -> None:
    """This function saves the values used to make the J_sc plot to a .csv file."""

    technology_name = get_technology_name()

    directory_outputs = f"{os.path.dirname(__file__)}/Output"
    os.makedirs(directory_outputs, exist_ok=True)

    with open(
        f"{directory_outputs}/{technology_name}_J_sc_values.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "AM",
                "Band gap energy (eV)",
                "Reported band gap energy (eV)",
                "Simulated J_sc (mA/cm^2)",
                "Reported J_sc (mA/cm^2)",
            ]
        )
        for x, _ in enumerate(AM):
            for i, band_gap in enumerate(E_G):
                if (1 + x * 0.25) == 1.5:
                    writer.writerow(
                        [
                            1.0 + x * 0.25,
                            band_gap,
                            reported_E_G[i],
                            J_G[x * len(AM) + i] * 1 / 10,
                            J_sc_reported[i] * 1 / 10,
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            1.0 + x * 0.25,
                            band_gap,
                            "N/A",
                            J_G[x * len(AM) + i] * 1 / 10,
                            "N/A",
                        ]
                    )


def save_J_o_values(J_o: np.ndarray, E_G: np.ndarray, reported_E_G: np.ndarray) -> None:
    """This function saves the values used to make the J_o plot to a .csv file."""

    technology_name = get_technology_name()

    directory_outputs = f"{os.path.dirname(__file__)}/Output"
    os.makedirs(directory_outputs, exist_ok=True)

    with open(
        f"{directory_outputs}/{technology_name}_J_o_values.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Band gap energy (eV)",
                "Reported band gap energy (eV)",
                "J_o (A/m^2)",
            ]
        )
        for i, band_gap in enumerate(E_G):
            writer.writerow(
                [
                    band_gap,
                    reported_E_G[i],
                    J_o[i],
                ]
            )


def make_V_oc_plot(
    V_oc: np.ndarray,
    V_oc_reported: np.ndarray,
    AM: np.ndarray,
    E_G: np.ndarray,
) -> None:
    """This function makes a plot of V_oc against band gap energy for each AM value."""

    technology_name = get_technology_name()

    colors = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
    ]  # Store 5 colour-blind friendly colours

    markers = []
    for marker in list(mlines.Line2D.markers.keys()):
        if isinstance(marker, int):
            marker = str(marker)
        markers.append(marker)

    for x, _ in enumerate(AM):
        plt.figure()

        V_oc_theory, E_G_theory, _, _ = theory.return_values_AM_any(1.0 + x * 0.25)
        for i, band_gap in enumerate(E_G):

            if i == 0:  # Only add label for first point (they all have the same colour)
                plt.scatter(
                    band_gap,
                    V_oc[i][x],
                    color=colors[0],
                    label="Simulated value",
                    marker=markers[0],
                )
            plt.scatter(
                band_gap,
                V_oc[i][x],
                color=colors[0],
                marker=markers[i],
            )

            if (1 + x * 0.25) == 1.5:
                if (
                    i == 0
                ):  # Only add label for first point (they all have the same colour)
                    plt.scatter(
                        band_gap,
                        V_oc_reported[i],
                        color=colors[1],
                        label="Reported value",
                        marker=markers[0],
                    )

                plt.scatter(
                    band_gap,
                    V_oc_reported[i],
                    color=colors[1],
                    marker=markers[i],
                )

            if i == 0:
                plt.plot(
                    E_G_theory,
                    V_oc_theory,
                    label="Theoretical limit",
                )

        plt.xlim(min(E_G) - 0.005, max(E_G) + 0.005)
        plt.ylim(min(V_oc_reported) - 0.2, max(V_oc[i]) + 0.2)
        plt.xlabel("Simulated band gap energy (eV)")
        plt.ylabel("$V_{oc}$ (V)")
        plt.legend()
        if x == 0:
            # I've treated AM0 as AM 1.0 throughout because a lot of program logic
            # depends on AM 1.0 existing. This is a hack to make it so that the output
            # does not say "AM0" but instead "AM1.0", as AM1.0 cannot physically exist.
            plt.title(f"{technology_name}, AM0")
            directory_outputs = f"{os.path.dirname(__file__)}/Output/AM0"
            os.makedirs(directory_outputs, exist_ok=True)
            plt.savefig(f"{directory_outputs}/{technology_name}_V_oc_AM0.png")
            plt.close()
        else:
            plt.title(f"{technology_name}, AM{1.0 + x * 0.25}")
            directory_outputs = f"{os.path.dirname(__file__)}/Output/AM{1.0 + x * 0.25}"
            os.makedirs(directory_outputs, exist_ok=True)
            plt.savefig(
                f"{directory_outputs}/{technology_name}_V_oc_AM{1.0 + x * 0.25}.png"
            )
            plt.close()


def make_J_sc_plot(
    J_G: np.ndarray,
    J_sc_reported: np.ndarray,
    AM: np.ndarray,
    E_G: np.ndarray,
) -> None:
    """This function makes a plot of J_sc (= J_G) against band gap energy for each AM value."""

    technology_name = get_technology_name()

    colors = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
    ]  # Store 5 colour-blind friendly colours

    markers = []
    for marker in list(mlines.Line2D.markers.keys()):
        if isinstance(marker, int):
            marker = str(marker)
        markers.append(marker)

    for x, _ in enumerate(AM):
        plt.figure()

        _, E_G_theory, J_sc_theory, _ = theory.return_values_AM_any(1.0 + x * 0.25)
        for i, band_gap in enumerate(E_G):

            if i == 0:  # Only add label for first point (they all have the same colour)
                plt.scatter(
                    band_gap,
                    J_G[x * len(AM) + i] * 1 / 10,
                    color=colors[0],
                    label="Simulated value",
                    marker=markers[0],
                )
            plt.scatter(
                band_gap,
                J_G[x * len(AM) + i] * 1 / 10,
                color=colors[0],
                marker=markers[i],
            )

            if (1 + x * 0.25) == 1.5:

                if i == 0:
                    plt.scatter(
                        band_gap,
                        J_sc_reported[i] * 1 / 10,
                        color=colors[1],
                        label="Reported value",
                        marker=markers[0],
                    )

                plt.scatter(
                    band_gap,
                    J_sc_reported[i] * 1 / 10,
                    color=colors[1],
                    marker=markers[i],
                )

            if i == 0:
                plt.plot(
                    E_G_theory,
                    J_sc_theory * 1 / 10,
                    label="Theoretical limit",
                )

        plt.xlabel("Simulated band gap energy (eV)")
        plt.ylabel("$J_{sc}$ (mA/cm$^2$)")
        plt.legend()
        plt.xlim(min(E_G) - 0.005, max(E_G) + 0.005)
        plt.ylim(
            min(
                np.min(J_sc_reported / 10),
                np.min(np.array(J_sc_theory) / 10),
            ),
            max(
                np.max(J_sc_reported / 10),
                np.max(np.array(J_sc_theory) / 10),
            )
            + 0.1,
        )
        if x == 0:
            # I've treated AM0 as AM 1.0 throughout because a lot of program logic
            # depends on AM 1.0 existing. This is a hack to make it so that the output
            # does not say "AM0" but instead "AM1.0", as AM1.0 cannot physically exist.
            plt.title(f"{technology_name}, AM0")
            directory_outputs = f"{os.path.dirname(__file__)}/Output/AM0"
            os.makedirs(directory_outputs, exist_ok=True)
            plt.savefig(f"{directory_outputs}/{technology_name}_J_sc_AM0.png")
            plt.close()
        else:
            plt.title(f"{technology_name}, AM{1.0 + x * 0.25}")
            directory_outputs = f"{os.path.dirname(__file__)}/Output/AM{1.0 + x * 0.25}"
            os.makedirs(directory_outputs, exist_ok=True)
            plt.savefig(
                f"{directory_outputs}/{technology_name}_J_sc_AM{1.0 + x * 0.25}.png"
            )
            plt.close()
        plt.close()


def make_J_o_plot(J_o: np.ndarray, E_G: np.ndarray) -> None:
    """This function makes a plot of J_o against band gap energy."""

    technology_name = get_technology_name()

    plt.figure()
    for i, band_gap in enumerate(E_G):
        if i == 0:
            plt.scatter(band_gap, J_o[i], color="#000000", label="Simulated value")
        plt.scatter(
            band_gap,
            J_o[i],
            color="#000000",
        )

    _, E_G_theory, _, J_o_theory = theory.return_values_AM_any(
        1.0
    )  # AM doesn't matter for J_o
    plt.plot(
        E_G_theory,
        J_o_theory,
        label="Theoretical limit",
    )

    plt.ylim(0, max(J_o) + 1e-18)
    plt.xlim(min(E_G) - 0.005, max(E_G) + 0.005)
    plt.xlabel("Simulated band gap energy (eV)")
    plt.ylabel("$J_{0}$ (A/m$^2$)")
    plt.title(f"{technology_name}")
    plt.legend()
    directory_outputs = f"{os.path.dirname(__file__)}/Output"
    os.makedirs(directory_outputs, exist_ok=True)
    plt.savefig(f"{directory_outputs}/{technology_name}_J_0.png")
    plt.close()
