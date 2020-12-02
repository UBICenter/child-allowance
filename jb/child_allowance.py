## Calculate largest differences in the poverty rate by simulation by state.
## Interested in child poverty as well.
## How much does child allowance compare to child care

## Interested in WHERE the poverty differential is taking place expenditure-wise
## % change in poverty/child poverty.
## In which states do childcare expenses push the most kids/adults into poverty.

# 1. Calculate largest differences in the poverty rate by simulation by state.
# 2. Do this in a second commit: Construct function to turn race into these 4 buckets:
# BLACK HISPANIC White-non-hispanic other-non-hispanic
##### CHANGES TO THE RACE DELINIATIONS?

# Preamble and read data
import microdf as mdf
import pandas as pd
import numpy as np
import us

# Read in census data and specify columns for use
raw = pd.read_csv(
    "wsl$\\Ubuntu\\home\\johnwalker\\childallowance\\jb\\datacps_00003.csv.gz",
    usecols=[
        "YEAR",
        "MONTH",
        "STATEFIP",
        "AGE",
        "SEX",
        "RACE",
        "HISPAN",
        "SPMWT",
        "SPMFTOTVAL",
        "SPMTOTRES",
        "SPMCHXPNS",
        "SPMTHRESH",
        "SPMFAMUNIT",
        "ASECWT",
    ],
)

# Create a copy of the raw dataset and make column names non-capitalized
# for readability
person_1 = raw.copy(deep=True)
person_1 = person_1[~person_1.ASECWT.isnull()]
person_1.columns = person_1.columns.str.lower()

# Define child age identifiers
person_1["child_6"] = person_1.age < 6
person_1["infant"] = person_1.age < 1
person_1["toddler"] = person_1.age.between(1, 2)
person_1["preschool"] = person_1.age.between(3, 5)
person_1["person"] = 1

# Redefine race categories
person_1.race = person_1.race.replace(
    {
        100: "White",
        200: "Black",
        300: "American First Nations",
        651: "Asian",
        651: "Hawaiian/Pacific Islander",
    }
)

# Define data collected at the SPM unit level
spmu_cols = [
    "spmfamunit",
    "spmwt",
    "spmftotval",
    "spmtotres",
    "spmchxpns",
    "spmthresh",
    "year",
]

spmu = pd.DataFrame(
    person_1.groupby(spmu_cols)[
        ["child_6", "infant", "toddler", "preschool", "person"]
    ].sum()
).reset_index()

# Calculate total cost of transfers, and total number of children
program_cost = mdf.weighted_sum(spmu, "spmchxpns", "spmwt")
spmu["total_child_6"] = mdf.weighted_sum(spmu, "child_6", "spmwt")

# Create copies of the dataset in which to simulate the policies
spmu_replace_cost = spmu.copy(deep=True)
spmu_flat_transfer = spmu.copy(deep=True)

# Generate simulation-level flags to separate datasets,
# 0 = base case, 1 = cost replacement design, 2 = flat transfer
spmu["sim_flag"] = 0
spmu_replace_cost["sim_flag"] = 1
spmu_flat_transfer["sim_flag"] = 2

# Caluclate new income by simulation
spmu_replace_cost.spmftotval += spmu_replace_cost.spmchxpns
spmu_flat_transfer.spmftotval += (
    program_cost / spmu_flat_transfer.total_child_6
)

# Append dataframes
spmu_sim = pd.concat(
    [spmu, spmu_replace_cost, spmu_flat_transfer], ignore_index=True
)

# Create poverty flags on simulated incomes
spmu_sim["poverty_flag"] = spmu_sim.spmftotval < spmu_sim.spmthresh

# Construct dataframe to disaggregate poverty flag to person level
person = person_1.merge(
    spmu_sim[["spmfamunit", "poverty_flag", "sim_flag", "year"]],
    on=["spmfamunit", "year"],
)

# Consider sex, race, state heterogeneity
poverty_rate = (
    person.groupby(["sim_flag"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)
poverty_rate_sex = (
    person.groupby(["sim_flag", "sex"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)
poverty_rate_race = (
    person.groupby(["sim_flag", "race"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)
poverty_rate_hispan = (
    person.groupby(["sim_flag", "hispan"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)
poverty_rate_state = (
    person.groupby(["sim_flag", "statefip"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)
poverty_rate.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_sex.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_race.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_hispan.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_state.rename({0: "poverty_rate"}, axis=1, inplace=True)
