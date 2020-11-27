# Roadmap:
# Get raw data from IPUMS and save to jb/data
# Use GESTFIPS state identifier (column) to identify
# State-based heterogeneity in first simulation
# Calculate Gini Coefficient using microdf
# Improve these initial estimates by averaging over 3 years
# Look at by-decile code used in child allowance simulation
# Replicate for this analysis
# Do exploratory analysis on
# Attempt 2nd analysis when data from CAP is provided

# Data to pull from IPUMS:
# Person-level - Hispanic status/race identifiers, Age, Weight
# SPM-level - CCare Exp, SPMID, SPMWEIGHT, TOTRes, POVThreshold
# See what housing cost data is available (should be geo code
# adjustments for different costs - SPM unit housing multiplier

# Links:
# https://github.com/MaxGhenis/datarepo/blob/master/pppub20.csv.gz
# https://github.com/UBICenter/child-allowance/blob/master/jb/
# simulation.ipynb
# https://github.com/UBICenter/child-allowance/blob/master/jb/
# data/make_decile_data.py

# Preamble and read data
import microdf as mdf
import pandas as pd
import numpy as np
import us

# Read in census data and specify columns for use
raw = pd.read_csv(
    "https://github.com/MaxGhenis/datarepo/raw/master/pppub20.csv.gz",
    usecols=[
        "A_AGE",
        "SPM_ID",
        "SPM_WEIGHT",
        "SPM_TOTVAL",
        "SPM_POVTHRESHOLD",
        "SPM_CHILDCAREXPNS",
        "MARSUPWT",
    ],
)

# Create a copy of the raw dataset and make column names non-capitalized
# for readability
person = raw.copy(deep=True)
person.columns = person.columns.str.lower()

# Define child age identifiers
person["child_6"] = person.a_age < 6
person["infant"] = person.a_age < 1
person["toddler"] = person.a_age.between(1, 2)
person["preschool"] = person.a_age.between(3, 5)
person["person"] = 1

# Aggregate to SPM level
spmu_cols = [
    "spm_id",
    "spm_weight",
    "spm_totval",
    "spm_povthreshold",
    "spm_childcarexpns",
]

spmu = pd.DataFrame(
    person.groupby(spmu_cols)[
        ["child_6", "infant", "toddler", "preschool", "person"]
    ].sum()
).reset_index()

# Calculate total cost of transfers, and total number of children
program_cost = mdf.weighted_sum(spmu, "spm_childcarexpns", "spm_weight")
person["total_child_6"] = mdf.weighted_sum(spmu, "child_6", "spm_weight")

# Create copies of the dataset in which to simulate the policies
spmu_replace_cost = person.copy(deep=True)
spmu_flat_transfer = person.copy(deep=True)

# Generate simulation-level flags to separate datasets,
# 0 = base case, 1 = cost replacement design, 2 = flat transfer
person["sim_flag"] = 0
spmu_replace_cost["sim_flag"] = 1
spmu_flat_transfer["sim_flag"] = 2

# Append dataframes
spmu = pd.concat(
    [person, spmu_replace_cost, spmu_flat_transfer], ignore_index=True
)

# Calculate transfer size to individual SPM units
spmu["flat_transfer"] = program_cost / spmu.total_child_6

# Simulate new income using replace_cost flag to identify base and 2 policies:
# 0. base case
spmu.loc[spmu["sim_flag"] == 0, "new_inc"] = spmu.spm_totval

# 1. childcare cost subsidy to household equal to childcare expenditure
# (cost replacement)
spmu.loc[spmu["sim_flag"] == 1, "new_inc"] = (
    spmu.spm_totval + spmu.spm_childcarexpns
)

# 2. flat allowance per child of equal total value
spmu.loc[spmu["sim_flag"] == 2, "new_inc"] = (
    spmu.spm_totval + spmu.flat_transfer
)

# Create poverty flags on simulated incomes
spmu["poverty_flag"] = spmu.new_inc < spmu.spm_povthreshold

# Construct dataframe to disaggregate poverty flag to person level
person = person.merge(
    spmu[["spm_id", "poverty_flag", "sim_flag"]], on=["spm_id"]
)

# summation across poverty_flag using person level weights for each policy
sim_flag_0 = person.loc[person.sim_flag == 0]
poverty_rate_base = mdf.weighted_mean(sim_flag_0, "poverty_flag", "marsupwt")
sim_flag_1 = person.loc[person.sim_flag == 1]
poverty_rate_replace = mdf.weighted_mean(
    sim_flag_1, "poverty_flag", "marsupwt"
)
sim_flag_2 = person.loc[person.sim_flag == 2]
poverty_rate_flat = mdf.weighted_mean(sim_flag_2, "poverty_flag", "marsupwt")

# Construct first differences and % changes
dif_poverty_rate_replace = poverty_rate_base - poverty_rate_replace
p_dif_poverty_rate_replace = dif_poverty_rate_replace / poverty_rate_base
dif_poverty_rate_flat = poverty_rate_base - poverty_rate_flat
p_dif_poverty_rate_flat = dif_poverty_rate_flat / poverty_rate_base
