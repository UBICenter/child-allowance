## Calculate largest differences in the poverty rate by simulation by state.
## Interested in child poverty as well.
## How much does child allowance compare to child care

## Interested in WHERE the poverty differential is taking place expenditure-wise
## % change in poverty/child poverty.
## In which states do childcare expenses push the most kids/adults into poverty.

# "wsl$\\Ubuntu\\home\\johnwalker\\childallowance\\jb\\datacps_00003.csv.gz",

# Preamble and read data
import microdf as mdf
import pandas as pd
import numpy as np
import us


# Read in census data and specify columns for use
raw = pd.read_csv(
    "C:\\Users\\John Walker\\Downloads\\cps_00003.csv.gz",
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
person_1["race"].mask(
    person_1.race.between(200, 999, inclusive=False), "Other", inplace=True
)
person_1["race"].mask(person_1.race == 100, "White", inplace=True)
person_1["race"].mask(person_1.race == 200, "Black", inplace=True)
person_1["race"].mask(person_1.race == 999, "Race unknown", inplace=True)

# Redefine hispanic categories
person_1["hispan"].mask(
    person_1.hispan.between(100, 612, inclusive=True), "Hispanic", inplace=True
)
person_1["hispan"].mask(person_1.hispan == 0, "Not Hispanic", inplace=True)
person_1["hispan"].mask(
    (person_1.hispan != "Hispanic") & (person_1.hispan != "Not Hispanic"),
    "Hispanic status unknown",
    inplace=True,
)

# Combine race + hispanic categories
person_1["race_hispan"] = np.nan
person_1["race_hispan"] = person_1["race_hispan"].mask(
    (person_1["race"] == "White") & (person_1["hispan"] == "Not Hispanic"),
    "White non-Hispanic",
)
person_1["race_hispan"] = person_1["race_hispan"].mask(
    (person_1["race"] == "Black") & (person_1["hispan"] == "Not Hispanic"),
    "Black",
)
person_1["race_hispan"] = person_1["race_hispan"].mask(
    (person_1["race"] == "Other") & (person_1["hispan"] == "Not Hispanic"),
    "Other non-Hispanic",
)
person_1["race_hispan"] = person_1["race_hispan"].mask(
    person_1["hispan"] == "Hispanic", "Hispanic"
)
person_1["race_hispan"] = person_1["race_hispan"].mask(
    (person_1["race"] == "Race unknown") & (["hispan"] == "Not Hispanic"),
    "Race unknown",
)
person_1["race_hispan"] = person_1["race_hispan"].mask(
    (person_1["race"] == "Race unknown")
    & (["hispan"] == "Hispanic status unknown"),
    "Race unknown",
)

# Relabel sex categories
person_1["sex"].mask(person_1["sex"] == 1, "Male", inplace=True)
person_1["sex"].mask(person_1["sex"] == 2, "Female", inplace=True)

## TO DO
# Create State categories
# person_1["statefip"] = (
#  pd.Series(state.index)
#  .apply(lambda x: us.states.lookup(str(x).zfill(2)).name)
#  .tolist()
# )

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
poverty_rate_race_hispan = (
    person.groupby(["sim_flag", "race_hispan"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)
poverty_rate_state = (
    person.groupby(["sim_flag", "statefip"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)

# Rename constructed poverty_rate
poverty_rate.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_sex.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_race_hispan.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_state.rename({0: "poverty_rate"}, axis=1, inplace=True)

# Construct first differences (pp changes)
# dif_pov_flat = poverty_rate.loc[poverty_rate["sim_flag"] == 1] -
# # poverty_rate.loc[poverty_rate["simflag"] == 0]
# dif_pov_flat_sex =
# dif_pov_replace = poverty_rate["sim_flag"] == 2 - poverty_rate["simflag"] == 0
# dif_pov_replace_sex =

# Interesting findings
# The poverty change is much larger for female-identifying people and generally
# flat transfer is much more effective, particularly for female-identifying.
# The poverty change for the flat transfer is largest for Black and \
# Hispanic populations ~3%, lower for White ~1.2%, and other non-hispanic ~1.8%.


# Sort by poverty rate to show max sizes (state)
# poverty_rate_state = poverty_rate_state.sort_values(by="", ascending=False)
# TO DO: Insert column value.