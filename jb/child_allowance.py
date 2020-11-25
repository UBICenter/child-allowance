# 1. Gov replace childcare costs v flat transfer
# 2. Gov pay state-based costs of childcare v flat transfer

# TO DO: Work out states:'GESTFIPS' by geting raw
# data from IPUMS rather than Max's repo
#### EMAIL BROOKINGS
#### roadmap:
### Poverty by state
### Do other
### Inequality
### By decile inequality

### Swap to IPUMS data for the 3 years
### Hispanic status
### Age
### Weight
### SPM - SPMID, SPMWEIGHT, TOTRes, POV Threshold
### Grab childcare expenses column
### Housing costs - Geo adjust for local area 
### (SPM unit housing multiplier)


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

# Read in census data and create a copy for manipulation
raw = pd.read_csv(
    "https://github.com/MaxGhenis/datarepo/raw/master/pppub20.csv.gz"
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

# Calculate total cost of transfers, weight number of children
program_cost = mdf.weighted_sum(spmu, "spm_childcarexpns", "spm_weight")
person["total_child_6"] = mdf.weighted_sum(spmu, "child_6", "spm_weight")

# Create copies of the dataset in which to simulate the policies
spmu_replace_cost = person.copy(deep=True)
spmu_flat_transfer = person.copy(deep=True)

# Generate simulation-level flags to separate datasets
spmu_replace_cost["replace_cost"] = True
spmu_flat_transfer["replace_cost"] = False

# Append dataframes
spmu = pd.concat([spmu_replace_cost, spmu_flat_transfer], ignore_index=True)

# Calculate transfer size to individual SPM units
spmu["flat_transfer"] = program_cost / spmu.total_child_6

# Simulate new income given 1. childcare cost subsidy to household 
# equal to childcare expenditure
# or 2. flat allowance of equal value using replace_cost flag.

spmu.loc[spmu["replace_cost"] == True, "new_inc"] = (
    spmu.spm_totval + spmu.spm_childcarexpns
)
spmu.loc[spmu["replace_cost"] == False, "new_inc"] = (
    spmu.spm_totval + spmu.flat_transfer
)

# Create poverty flags on simulated incomes
spmu["poverty_flag"] = spmu.new_inc < spmu.spm_povthreshold

# Disaggregate to person level by merging on

# Construct dataframe to disaggregate poverty flag to person level
person = person.merge(spmu[["spm_id", "poverty_flag", "replace_cost"]], on=["spm_id"])

# summation across poverty_flag using person level weights for sim 1
poverty_rate_replace = person.loc[spmu["replace_cost"] == True, "pov_rt_replace"] = (
    mdf.weighted_mean(person, "poverty_flag", "marsupwt"
)

# summation across poverty_flag using person level weights for sim 2
poverty_rate_flat = person.loc[spmu["replace_cost"] == True, "pov_rt_replace"] = (
    mdf.weighted_mean(person, "poverty_flag", "marsupwt"
)

