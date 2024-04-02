from math import floor, log10
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

    if var_name == "V_oc":
        units = "V"
    elif var_name == "J_sc":
        units = "mA/cm^2"

    dataframe = pd.read_csv(
        f"{get_base_dir()}/Technology {folder_num} - {tech_name}/Output/{tech_name}_{var_name}_values.csv"
    )

    # Only keep rows that have AM = 1.5
    dataframe = dataframe[dataframe["AM"] == 1.5]

    # Drop every column that's not simulated or reported values
    dataframe = dataframe[
        [f"Simulated {var_name} ({units})", f"Reported {var_name} ({units})"]
    ]

    # Calculate difference between simulated and reported values
    diffs = (
        dataframe[f"Simulated {var_name} ({units})"]
        - dataframe[f"Reported {var_name} ({units})"]
    )

    mean = np.mean(diffs)
    std = np.std(diffs)

    # Save out to csv using python writer
    with open(
        f"{get_base_dir()}/Statistical Analyses/Output/sim_rep_diffs/{tech_name}_{var_name}_simrepdiffs.csv",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"mean,std\n{mean},{std}\n")


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
