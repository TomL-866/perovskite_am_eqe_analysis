# Code for "Analysing Perovskite Solar Cell Performance as a Function of Airmass and EQE".

This repository contains the code used to generate the results given in the paper "Analysing Perovskite Solar Cell Performance as a Function of Airmass and EQE". 

This README file contains instructions for building the project.

## Requirements

To build this project, all you will need is a working installation of Python 3.10. You can download Python from the [official website](https://www.python.org/downloads/).

## Building the project

The following commands are for the bash shell (Linux).

In the project directory, run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Running the code

There are 4 folders in the project, corresponding to the analysis of 4 separate technologies. To run the code for a specific technology, navigate to the corresponding folder and run the following command:

```bash
python3 main.py
```

For example, to run the analysis on the MAPbI3 technology, you would run the following commands:

```bash
cd "Technology 1 - MAPbI3"
python3 main.py
```

This will run the code and generate all plots and data files used in the paper.