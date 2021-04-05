# Preamble and read data
import microdf as mdf
import pandas as pd
import numpy as np
import us

"""
Load this locally (should also be able to skip compression="gzip" after doing so
Explain load this locally
"""

# Read in census data and specify columns for use
person_raw = pd.read_csv(
    "https://github.com/UBICenter/child-allowance/blob/master/jb/data/cps_00003.csv.gz?raw=true",  # noqa
    compression="gzip",
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
person = person_raw.copy(deep=True)
person.columns = person.columns.str.lower()

# Asec weights are year-person units, and we average over 3 years,
# so divide by the 3 years to give per year weights.
person.asecwt /= 3

# Define child age identifiers
person["child_6"] = person.age < 6
person["infant"] = person.age < 1
person["toddler"] = person.age.between(1, 2)
person["preschool"] = person.age.between(3, 5)
person["person"] = 1

# Redefine race categories
person["race_group"] = "Other"
person.race_group.mask(person.race == 100, "White", inplace=True)
person.race_group.mask(person.race == 200, "Black", inplace=True)

# Redefine hispanic categories
person["hispanic"] = person.hispan.between(100, 612, inclusive=True)

# Combine race + hispanic categories
person["race_hispan"] = "Other non-Hispanic"
person.race_hispan.mask(
    (person.race_group == "White") & ~person.hispanic,
    "White non-Hispanic",
    inplace=True,
)
person.race_hispan.mask(
    (person.race_group == "Black") & ~person.hispanic,
    "Black non-Hispanic",
    inplace=True,
)
person.race_hispan.mask(person.hispanic, "Hispanic", inplace=True)

# Relabel sex categories
person["female"] = person.sex == 2

# Create State categories
person["state"] = (
    pd.Series(person.statefip)
    .apply(lambda x: us.states.lookup(str(x).zfill(2)).name)
    .tolist()
)

# Define data collected at the SPM unit level
SPMU_COLS = [
    "spmfamunit",
    "spmwt",
    "spmftotval",
    "spmtotres",
    "spmchxpns",
    "spmthresh",
    "year",
]

spmu = pd.DataFrame(
    person.groupby(SPMU_COLS)[
        ["child_6", "infant", "toddler", "preschool", "person"]
    ].sum()
).reset_index()

SPMU_AGG_COLS = ["child_6", "infant", "toddler", "preschool", "person"]
spmu = person.groupby(SPMU_COLS)[SPMU_AGG_COLS].sum()
spmu.columns = ["spmu_" + i for i in SPMU_AGG_COLS]
spmu.reset_index(inplace=True)

# Calculate total cost of transfers, and total number of children
program_cost = mdf.weighted_sum(spmu, "spmchxpns", "spmwt")
total_child_6 = mdf.weighted_sum(spmu, "spmu_child_6", "spmwt")
childallowance = program_cost / total_child_6

# Create copies of the dataset in which to simulate the policies
spmu_replace_cost = spmu.copy(deep=True)
spmu_flat_transfer = spmu.copy(deep=True)

# Generate scenario flags to separate datasets
spmu["sim_flag"] = "baseline"
spmu_replace_cost["sim_flag"] = "cc_replacement"
spmu_flat_transfer["sim_flag"] = "child_allowance"

# Caluclate new income by simulation
spmu_replace_cost.spmftotval += spmu_replace_cost.spmchxpns

spmu_flat_transfer["childallowance"] = (
    childallowance * spmu_flat_transfer.spmu_child_6
)
spmu_flat_transfer.spmftotval += spmu_flat_transfer.childallowance

# Append dataframes
spmu_sim = pd.concat(
    [spmu, spmu_replace_cost, spmu_flat_transfer], ignore_index=True
)

# Create poverty flags on simulated incomes
spmu_sim["poverty_flag"] = spmu_sim.spmftotval < spmu_sim.spmthresh

# Calculate per person spmftotval (resources)
spmu_sim["resources_pp"] = spmu_sim.spmftotval / spmu_sim.spmu_person

# Construct dataframe to disaggregate poverty flag to person level
person_sim = person.drop("spmftotval", axis=1).merge(
    spmu_sim[
        [
            "spmfamunit",
            "year",
            "poverty_flag",
            "sim_flag",
            "spmftotval",
            "resources_pp",
        ]
    ],
    on=["spmfamunit", "year"],
)

# Define a function to calculate poverty rates
def pov(data, group):
    return pd.DataFrame(
        mdf.weighted_mean(data, "poverty_flag", "asecwt", groupby=group)
    )


# Poverty rate and state/demographic-based heterogenous poverty rates
poverty_rate = pov(person_sim, "sim_flag")
poverty_rate_state = pov(person_sim, ["sim_flag", "state"])
poverty_rate_sex = pov(person_sim, ["sim_flag", "sex"])
poverty_rate_race_hispan = pov(person_sim, ["sim_flag", "race_hispan"])

# Child poverty rate
poverty_rate_child = pov(person_sim[person_sim.child_6], "sim_flag")

# Rename constructed poverty_rates
poverty_rates = [
    poverty_rate,
    poverty_rate_sex,
    poverty_rate_race_hispan,
    poverty_rate_state,
    poverty_rate_child,
]
for i in poverty_rates:
    i.rename({0: "poverty_rate"}, axis=1, inplace=True)

"""
The following code creates a pivot table to examine in detail
the impacts of each policy on state-based outcomes. The procedure
is replicable for any of the demographics of interest.
"""


# Define percentage change functions
def percent_change(pp_change, old):
    return 100 * pp_change / old


# Define function to generate gini coefficients
def gin(data, group):
    return pd.DataFrame(
        data.groupby(group).apply(
            lambda x: mdf.gini(x, "spmftotval", "asecwt")
        )
    )


# Gini coefficients and state/demographic-based heterogenous gini coefficients
gini = gin(person_sim, "sim_flag")
gini_state = gin(person_sim, ["sim_flag", "state"])

# Rename constructed gini coefficients
ginis = [
    gini,
    gini_state,
]
for i in ginis:
    i.rename({0: "gini_coefficient"}, axis=1, inplace=True)

# Create pivot table to interpret state-based poverty effects
state_pov = poverty_rate_state.pivot_table(
    values="poverty_rate", index="state", columns="sim_flag"
)
# Create pivot table to interpret state-based gini effects
state_gini = gini_state.pivot_table(
    values="gini_coefficient", index="state", columns="sim_flag"
)

"""
Construct percentage changes in defined metrics
"""

# Generate state-based poverty rate percentage changes
state_pov["poverty_change_cc"] = state.baseline - state.cc_replacement
state_pov["poverty_change_flat"] = state.baseline, state.child_allowance
state_pov["poverty_change_%_cc"] = state_pov.poverty_change_cc - state.baseline
state_pov["poverty_change_%_flat"] = (
    state_pov.poverty_change_flat - state.baseline
)

# Construct state-based gini coefficient percentage changes
state_gini["gini_change_cc"] = pp_change(state.baseline, state.cc_replacement)
state_gini["gini_change_flat"] = pp_change(
    state.baseline, state.child_allowance
)
state_gini["gini_change_pc_cc"] = percent_change(
    state_gini.gini_change_cc, state.baseline
)
state_gini["gini_change_pc_flat"] = percent_change(
    state_gini.gini_change_flat, state.baseline
)

# Re-arrange and present pivot tables, descending by % change
# in poverty rate
state_pov.sort_values(by="poverty_change_%_flat", ascending=True)
state_gini.sort_values(by="gini_change_%_flat", ascending=True)
