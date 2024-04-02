import os
from math import floor, log10
import pandas as pd
import numpy as np
from helpers import get_base_dir


def diff_logic(tech_name):
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
        f"{get_base_dir()}/Technology {folder_num} - {tech_name}/Output/{tech_name}_V_oc_values.csv"
    )

    dataframe = dataframe[dataframe["AM"] == 1.5]

    dataframe = dataframe[["Band gap energy (eV)", "Reported band gap energy (eV)"]]
    dataframe = dataframe.dropna()
    print(dataframe)
    diffs = (
        dataframe["Band gap energy (eV)"] - dataframe["Reported band gap energy (eV)"]
    )
    mean = np.mean(diffs)
    std = np.std(diffs)

    with open(
        f"{get_base_dir()}/Statistical Analyses/Output/eg_diffs/{tech_name}_egdiffs.csv",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"mean,std\n{mean},{std}\n")


def mean_calcs():
    print("-- MAPbI3 --")
    diff_logic("MAPbI3")
    print("-- FAPbI3 --")
    diff_logic("FAPbI3")
    print("-- PerovskitePerovskite_Tandems --")
    diff_logic("PerovskitePerovskite_Tandems")
    print("-- PerovskiteSilicon_Tandems --")
    diff_logic("PerovskiteSilicon_Tandems")


def main():
    mean_calcs()
