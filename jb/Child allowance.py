# 1. Gov replace childcare costs v flat transfer
# 2. Gov pay state-based costs of childcare v flat transfer

# TO DO: Work out states:'GESTFIPS' by geting raw data from IPUMS rather than Max's repo

# Links:
# https://github.com/MaxGhenis/datarepo/blob/master/pppub20.csv.gz
# https://github.com/UBICenter/child-allowance/blob/master/jb/simulation.ipynb
# https://github.com/UBICenter/child-allowance/blob/master/jb/data/make_decile_data.py

# Preamble and read data
import microdf as mdf
import pandas as pd
import numpy as np
import us

# Read in census data and create a copy for manipulation
raw = pd.read_csv(
    "https://github.com/MaxGhenis/datarepo/raw/master/pppub20.csv.gz"
)
df = raw.copy(deep=True)

# Define child age identifiers
df["child_6"] = df.A_AGE < 6
df["infant"] = df.A_AGE < 1
df["toddler"] = df.A_AGE.between(1, 2)
df["preschool"] = df.A_AGE.between(3, 5)
df["person"] = 1

# Aggregate to SPM level
SPMU_COLS = [
    "SPM_ID",
    "SPM_WEIGHT",
    "SPM_TOTVAL",
    "SPM_POVTHRESHOLD",
    "SPM_CHILDCAREXPNS",
]
spmu = pd.DataFrame(
    df.groupby(SPMU_COLS)[
        ["child_6", "infant", "toddler", "preschool", "person"]
    ].sum()
).reset_index()

# Calculate total cost of transfers, weight number of children
program_cost = mdf.weighted_sum(spmu, "SPM_CHILDCAREXPNS", "SPM_WEIGHT")
df["total_child_6"] = mdf.weighted_sum(spmu, "child_6", "SPM_WEIGHT")

# Create copies of the dataset in which to simulate the policies and generate dataset flags
spmu_replace_cost = df.copy(deep=True)
spmu_flat_transfer = df.copy(deep=True)

# Create simulation flag
spmu_replace_cost["replace_cost"] = True
spmu_flat_transfer["replace_cost"] = False

# Append dataframes
simdf = pd.concat([spmu_replace_cost, spmu_flat_transfer], ignore_index=True)

# Calculate transfer size to individual SPM units
simdf["flat_transfer"] = program_cost / simdf.total_child_6

# Simulate new income given 1. childcare cost subsidy to household equal to childcare expenditure
# or 2. flat allowance of equal value using replace_cost flag.

simdf.loc[simdf["replace_cost"] == True, "new_inc"] = (
    simdf.SPM_TOTVAL + simdf.SPM_CHILDCAREXPNS
)
simdf.loc[simdf["replace_cost"] == False, "new_inc"] = (
    simdf.SPM_TOTVAL + simdf.flat_transfer
)

# Create poverty flags on simulated incomes
simdf["poverty_flag"] = simdf.new_inc < simdf.SPM_POVTHRESHOLD

# Disaggregate to person level


# Calculate person-level poverty rate by multiplying through by person-level weights