import os
from math import floor, log10
import uncertainties
import pandas as pd
import numpy as np
from helpers import get_base_dir


def diff_logic(tech_name, var_name):
    if tech_name == "MAPbI3":
        folder_num = 1
    elif tech_name == "FAPbI3":
        folder_num = 2
    elif tech_name == "PerovskitePerovskite_Tandems":
        folder_num = 3
    elif tech_name == "PerovskiteSilicon_Tandems":
        folder_num = 4
    else:
        raise ValueError("Invalid tech_name")

    dataframe = pd.read_csv(
        f"{get_base_dir()}/Technology {folder_num} - {tech_name}/Output/{tech_name}_{var_name}_values.csv"
    )

    am_values = [1, 1.25, 1.5, 1.75, 2]
    data = {am: dataframe[dataframe["AM"] == am].to_numpy() for am in am_values}

    var_data = {am: data[:, 3] for am, data in data.items()}

    diffs = [
        [voc1 - voc2 for voc1, voc2 in zip(var_data[am1], var_data[am2])]
        for am1, am2 in zip(am_values, am_values[1:])
    ]

    output_df = pd.DataFrame()
    for i, arr in enumerate(diffs):
        if i == 0:
            am = 0
            mean_of_diffs = np.mean(arr)
            std_of_diffs = np.std(arr)

            mean_of_diffs = uncertainties.ufloat(mean_of_diffs, std_dev=std_of_diffs)

            temp_df = pd.DataFrame(
                {
                    "AM": [f"AM{am} - AM{1.25}"],
                    "mean of diffs": [mean_of_diffs],
                }
            )
            output_df = pd.concat([output_df, temp_df])
        else:
            am = am_values[i]
            mean_of_diffs = np.mean(arr)
            std_of_diffs = np.std(arr)

            mean_of_diffs = uncertainties.ufloat(mean_of_diffs, std_dev=std_of_diffs)

            temp_df = pd.DataFrame(
                {
                    "AM": [f"AM{am} - AM{am + i*0.25}"],
                    "mean of diffs": [mean_of_diffs],
                }
            )
            output_df = pd.concat([output_df, temp_df], ignore_index=True)
    print(output_df)
    os.makedirs(f"{get_base_dir()}/Statistical Analyses/Output/am_diffs", exist_ok=True)
    output_df.to_csv(
        f"{get_base_dir()}/Statistical Analyses/Output/am_diffs/{var_name}_{tech_name}_diffs.csv"
    )


def voc_mean_calcs():
    print("\n --- Voc (V) ---")
    print("-- MAPbI3 --")
    diff_logic("MAPbI3", "V_oc")
    print("-- FAPbI3 --")
    diff_logic("FAPbI3", "V_oc")
    print("-- PerovskitePerovskite_Tandems --")
    diff_logic("PerovskitePerovskite_Tandems", "V_oc")
    print("-- PerovskiteSilicon_Tandems --")
    diff_logic("PerovskiteSilicon_Tandems", "V_oc")


def jsc_mean_calcs():
    print("\n --- Jsc (mA/cm^2) ---")
    print("-- MAPbI3 --")
    diff_logic("MAPbI3", "J_sc")
    print("-- FAPbI3 --")
    diff_logic("FAPbI3", "J_sc")
    print("-- PerovskitePerovskite_Tandems --")
    diff_logic("PerovskitePerovskite_Tandems", "J_sc")
    print("-- PerovskiteSilicon_Tandems --")
    diff_logic("PerovskiteSilicon_Tandems", "J_sc")


def main():
    voc_mean_calcs()
    jsc_mean_calcs()
