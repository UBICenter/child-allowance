# Preamble and read data
import microdf as mdf
import pandas as pd
import numpy as np
import us
import statsmodels.api as sm

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
person["child_18"] = person.age < 18
person["infant"] = person.age < 1
person["toddler"] = person.age.between(1, 2)
person["preschool"] = person.age.between(3, 5)
person["age_6_12"] = person.age.between(6, 12)
person["person"] = 1

# Define child age categories for group-by analysis and merge
person["age_cat"] = "over_5"
person.loc[(person.age < 1), "age_cat"] = "infant"
person.loc[(person.age.between(1, 2)), "age_cat"] = "toddler"
person.loc[(person.age.between(3, 5)), "age_cat"] = "preschool"

# reg SPMU childcare XPNS ~ number of kids in each age group
# If we gave an amount to each kid what are they for each
# age group.

# Infant number is low due to low takeup?
# Disabled kids - potentially can take another stab


# Define child age units for summation by SPMU unit
person["person"] = 1
person["child_6"] = person.age < 6
person["infant"] = person.age < 1
person["toddler"] = person.age.between(1, 2)
person["preschool"] = person.age.between(3, 5)

# Create State categories
person["state"] = (
    pd.Series(person.statefip)
    .apply(lambda x: us.states.lookup(str(x).zfill(2)).name)
    .tolist()
)

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
    "spmtotres",
    "spmchxpns",
    "spmthresh",
    "year",
]

spmu = pd.DataFrame(
    person.groupby(SPMU_COLS)[
        ["child_18", "child_6", "infant", "toddler", "preschool", "person"]
    ].sum()
).reset_index()

SPMU_AGG_COLS = [
    "child_18",
    "child_6",
    "infant",
    "toddler",
    "preschool",
    "age_6_12",
    "person",
]
spmu = person.groupby(SPMU_COLS)[SPMU_AGG_COLS].sum()
spmu.columns = ["spmu_" + i for i in SPMU_AGG_COLS]
spmu.reset_index(inplace=True)

reg = sm.regression.linear_model.WLS(
    spmu.spmchxpns,
    spmu[["spmu_infant", "spmu_toddler", "spmu_preschool", "spmu_age_6_12"]],
    weights=spmu.spmwt,
)
child_allowance_amounts = reg.fit().params

# Calculate total cost of transfers, and total number of children
program_cost = mdf.weighted_sum(spmu, "spmchxpns", "spmwt")
total_child_6 = mdf.weighted_sum(spmu, "spmu_child_6", "spmwt")
childallowance = program_cost / total_child_6

### Ben - characterize distribution of spmchxpns - histogram (by number of kids)

### filter out households with children over 6.
### Recover average cost for children under 6.
### Weighting to recover average - multiply by total number of kids under age six.
### Other options - predict reg childcare expenses ~ child ages + num_kid
# Less controls may be better here - just trying to decompose the amount
# Consider different specifications

# Create copies of the dataset in which to simulate the policies
spmu_replace_cost = spmu.copy(deep=True)
spmu_flat_transfer = spmu.copy(deep=True)

# Generate scenario flags to separate datasets
spmu["sim_flag"] = "baseline"
spmu_replace_cost["sim_flag"] = "cc_replacement"
spmu_flat_transfer["sim_flag"] = "child_allowance"

# Calculate new income by simulation
### Opportunity here to put in a threshold to define the incomes
### Once we get into tax, marginal tax rates + GE effects are annoying
spmu_replace_cost.spmtotres += spmu_replace_cost.spmchxpns

spmu_flat_transfer["childallowance"] = (
    child_allowance_amounts.spmu_infant * spmu_flat_transfer.spmu_infant
    + child_allowance_amounts.spmu_toddler * spmu_flat_transfer.spmu_toddler
    + child_allowance_amounts.spmu_preschool
    * spmu_flat_transfer.spmu_preschool
    + child_allowance_amounts.spmu_age_6_12 * spmu_flat_transfer.spmu_age_6_12
)

flat_transfer_cost = mdf.weighted_sum(
    spmu_flat_transfer, "childallowance", "spmwt"
)
cost_ratio = program_cost / flat_transfer_cost
spmu_flat_transfer.childallowance *= cost_ratio

spmu_flat_transfer.spmtotres += spmu_flat_transfer.childallowance

true_child_allowance = child_allowance_amounts * cost_ratio

# Append/stack/concatenate dataframes - allows for use of groupby functions
spmu_sim = pd.concat(
    [spmu, spmu_replace_cost, spmu_flat_transfer], ignore_index=True
)

# Create poverty flags on simulated incomes
# Threshold take into account household size and local property value
spmu_sim["poverty_flag"] = spmu_sim.spmtotres < spmu_sim.spmthresh

# Calculate per person spmtotres (resources) - we are not using this but
# may be useful for gini calculation
spmu_sim["resources_pp"] = spmu_sim.spmtotres / spmu_sim.spmu_person

# Construct dataframe to disaggregate poverty flag to person level
person_sim = person.drop("spmtotres", axis=1).merge(
    spmu_sim[
        [
            "spmfamunit",
            "year",
            "poverty_flag",
            "sim_flag",
            "spmtotres",
            "resources_pp",
        ]
    ],
    on=["spmfamunit", "year"],
)


def pov(data, group):
    return pd.DataFrame(
        mdf.weighted_mean(data, "poverty_flag", "asecwt", groupby=group)
    )


poverty_rate_child = pov(
    person_sim[person_sim.child_18], "sim_flag"
)  # Child poverty rate

# Output the dataset (which is housed on Github)
compression_opts = dict(method="gzip", archive_name="person_sim.csv")
person_sim.to_csv(
    "person_sim.csv.gz", index=False, compression=compression_opts
)
