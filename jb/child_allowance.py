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
    "https://github.com/UBICenter/child-allowance/blob/master/jb/data/cps_00003.csv.gz",
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
person = person[~person.ASECWT.isnull()]
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
person.race.mask(
    person.race.between(200, 999, inclusive=False), "Other", inplace=True
)
person.race.mask(person.race == 100, "White", inplace=True)
person.race.mask(person.race == 200, "Black", inplace=True)
person.race.mask(person.race == 999, "Race unknown", inplace=True)

# Redefine hispanic categories
person.hispan.mask(
    person.hispan.between(100, 612, inclusive=True), "Hispanic", inplace=True
)
person.hispan.mask(person.hispan == 0, "Not Hispanic", inplace=True)
person.hispan.mask(
    (person.hispan != "Hispanic") & (person.hispan != "Not Hispanic"),
    "Hispanic status unknown",
    inplace=True,
)

# Combine race + hispanic categories
person["race_hispan"] = "Other or Unknown"
person.race_hispan.mask(
    (person.race == "White") & (person.hispan == "Not Hispanic"),
    "White non-Hispanic",
    inplace=True,
)
person.race_hispan.mask(
    (person.race == "Black") & (person.hispan == "Not Hispanic"),
    "Black",
    inplace=True,
)
person.race_hispan.mask(
    (person.race == "Other") & (person.hispan == "Not Hispanic"),
    "Other non-Hispanic",
    inplace=True,
)
person.race_hispan.mask(person.hispan == "Hispanic", "Hispanic", inplace=True)

# Relabel sex categories
person["sex"].mask(person["sex"] == 1, "Male", inplace=True)
person["sex"].mask(person["sex"] == 2, "Female", inplace=True)

# Create State categories
person["state"] = (
    pd.Series(person.statefip)
    .apply(lambda x: us.states.lookup(str(x).zfill(2)).name)
    .tolist()
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
    person.groupby(spmu_cols)[
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
spmu["sim_flag"] = "baseline"
spmu_replace_cost["sim_flag"] = "cc_replacement"
spmu_flat_transfer["sim_flag"] = "child_allowance"

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
spmu_sim["poverty_flag_child"] = (spmu_sim.spmftotval < spmu_sim.spmthresh) & (
    spmu_sim.child_6 == 1
)

# Calculate per person spmftotval (resources)
spmu_sim["resources_pp"] = spmu_sim.spmftotval / spmu_sim.person

# Construct dataframe to disaggregate poverty flag to person level
person_sim = person.drop("spmftotval", axis=1).merge(
    spmu_sim[
        [
            "spmfamunit",
            "poverty_flag",
            "poverty_flag_child",
            "sim_flag",
            "year",
            "spmftotval",
            "resources_pp",
        ]
    ],
    on=["spmfamunit", "year"],
)

# Consider sex, race, state heterogeneity
poverty_rate = (
    person_sim.groupby(["sim_flag"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)
poverty_rate_child = (
    person_sim.groupby(["sim_flag"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag_child", "asecwt"))
    .reset_index()
)
poverty_rate_sex = (
    person_sim.groupby(["sim_flag", "sex"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)
poverty_rate_race_hispan = (
    person_sim.groupby(["sim_flag", "race_hispan"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)
poverty_rate_state = (
    person_sim.groupby(["sim_flag", "state"])
    .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
    .reset_index()
)

# Rename constructed poverty_rate
poverty_rate.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_sex.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_race_hispan.rename({0: "poverty_rate"}, axis=1, inplace=True)
poverty_rate_state.rename({0: "poverty_rate"}, axis=1, inplace=True)

# Construct poverty percentage changes
state = poverty_rate_state.pivot_table(
    values="poverty_rate", index="state", columns="sim_flag"
)


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
