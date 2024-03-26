"""This script calculates the open circuit voltage for a given EQE spectrum and
solar spectrum by deriving the band gap from the EQE spectrum itself. As a byproduct,
it also calculates the short circuit current density and generation 
current density for a given EQE spectrum. 

This is done for 5 different airmass values.
Results in the form of graphs are outputted in the "Output" folder.

This script should be in a folder named:
    Technology [number] - [Technology name]
(e.g. Technology 1 - MAPbI3)

This program takes EQE data in the form of .csv files. 
Place EQE spectra in an "EQE Data" folder in the same directory as this script.
Place solar spectra in a "Solar Spectra by Airmass" folder in the same directory as this script.
The EQE data .csv files should be named in the following format:
    EQE_[number]_[DOI with slashes removed].csv
(e.g. EQE_1_10.1021acs.jpclett.0c01776.csv would be acceptable)
([number] is used so that the order of EQE spectra can be controlled at will)

This program also requires a .csv of filtered data from the Perovskite database.
This file should have the following columns:
Ref_DOI_number,
Perovskite_composition_short_form,
Perovskite_composition_long_form,
EQE_integrated_Jsc,
JV_default_Voc,
JV_default_Jsc,
Perovskite_band_gap
This file should be named in the following format:
    [Technology name]_data.csv
and placed into the same directory as this script.

EQE data .csv files have as their first column wavelength values in nm, 
and as their second column EQE values in %.

Outputs are saved in the "Output" folder. This folder is created if it does not already exist.

NOTE: this program will only work if there are at least 2 EQE data files.
NOTE: In Solar Spectra by Airmass, the airmass 1.0 file actually represents AM0. 
This is because the program logic originally assumed that an airmass of 1.0 existed.
If you want to replace the solar spectra data with your own, keep this in mind!
"""

import matplotlib.pyplot as plt
import matplotlib

import functions as f


def main() -> None:
    # PGF plot settings for exporting plots to LaTeX
    plt.rcParams.update({"font.size": 20})
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "CMU Serif"],  #  Fallback to CMU Serif
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    plt.style.use("seaborn-v0_8-bright")

    print("Loading data...")
    EQE = f.load_EQE_data()
    AM = f.load_spectral_data()
    E_G = f.generate_E_G_data(EQE)
    sorted_dataframe_with_E_G = f.get_sorted_dataframe_with_reported_E_G()
    reported_E_G = sorted_dataframe_with_E_G["Perovskite_band_gap"].to_numpy()

    print("Calculating V_oc, J_sc, and J_o...")
    V_oc = f.calculate_V_oc(EQE, AM)
    V_oc_reported = f.get_reported_V_oc(sorted_dataframe_with_E_G)
    J_G = f.calculate_J_G(EQE, AM)  # This is equivalent to J_sc under our assumptions
    J_sc_reported = f.get_reported_J_sc(sorted_dataframe_with_E_G)
    J_o = f.calculate_J_o(EQE)

    print("Saving data...")
    f.save_J_sc_values(J_G, J_sc_reported, AM, E_G, reported_E_G)
    f.save_V_oc_values(V_oc, V_oc_reported, AM, E_G, reported_E_G)
    f.save_J_o_values(J_o, E_G, reported_E_G)

    print("Making plots...")
    f.make_V_oc_plot(V_oc, V_oc_reported, AM, E_G)
    f.make_J_sc_plot(J_G, J_sc_reported, AM, E_G)
    f.make_J_o_plot(J_o, E_G)


if __name__ == "__main__":
    main()
