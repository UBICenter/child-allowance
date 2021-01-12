# Roadmap:
# Comparison to other potential policies
# Replicate plotly code from the child allowance for JB
# Move to second simulation

# Preamble and read data
import microdf as mdf
import pandas as pd
import numpy as np
import us

# Read in census data and specify columns for use
raw = pd.read_csv(
    "https://github.com/UBICenter/child-allowance/blob/master/jb/data/cps_00003.csv.gz?raw=true",
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
person = raw.copy(deep=True)
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

# Generate scenario flags to separate datasets,
# 0 = base case, 1 = cost replacement design, 2 = flat transfer
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
def pov(groupby, data=person_sim):
    return (
        data.groupby(groupby)
        .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
        .reset_index()
    )


# Function to be simplified with microdf update to:
# `return mdf.weighted_mean(data, "poverty_flag", "asecwt",groupby)
# .reset_index()`

# Poverty rate and demographic-based heterogenous poverty rates
poverty_rate = pov("sim_flag")
poverty_rate_sex = pov(["sim_flag", "sex"])
poverty_rate_race_hispan = pov(["sim_flag", "race_hispan"])
poverty_rate_state = pov(["sim_flag", "state"])

# Child poverty rate
poverty_rate_child = pov("sim_flag", person_sim[person_sim.child_6])

# Rename constructed poverty_rates
poverty_rates = [
    poverty_rate,
    poverty_rate_sex,
    poverty_rate_race_hispan,
    poverty_rate_race_hispan,
    poverty_rate_state,
    poverty_rate_child,
]
for i in poverty_rates:
    i.rename({0: "poverty_rate"}, axis=1, inplace=True)

# Create pivot table to interpret state-based poverty effects
state = poverty_rate_state.pivot_table(
    values="poverty_rate", index="state", columns="sim_flag"
)

# Construct poverty percentage changes
def percent_change(base, new):
    return 100 * (new - base) / new


state["poverty_change_cc"] = percent_change(
    state.baseline, state.cc_replacement
)
state["poverty_change_flat"] = percent_change(
    state.baseline, state.child_allowance
)

# Gini coefficients
mdf.gini(person_sim, "spmftotval", "asecwt")
mdf.gini(person_sim, "resources_pp", "asecwt")
person_sim.groupby("sim_flag").apply(
    lambda x: mdf.gini(x, "spmftotval", "asecwt")
)
person_sim.groupby("sim_flag").apply(
    lambda x: mdf.gini(x, "resources_pp", "asecwt")
)

# Re-arrange by poverty rate
state.sort_values(by="poverty_change_flat", ascending=False)

# Interesting findings:
# Flat transfer: Roughly 10* the impact on poverty and gini coefficient
# compared to the childcare provision equivalent policy (paying costs)
# The poverty change is much larger for female-identifying people.
# The poverty change for the flat transfer is largest for Black and
# Hispanic populations ~3%, lower for White ~1.2%, and other non-hispanic ~1.8%.
