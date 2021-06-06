# Packages
import microdf as mdf
import pandas as pd
import numpy as np
import us
import statsmodels.api as sm

"""
Read in the CPS and CAP datasets
"""

# Read in CPS data and specify columns for use
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

# Read in CAP dataset
costs_raw = pd.read_csv(
    "C:\\Users\\John Walker\\Desktop\\CCare_cost.csv",
)

"""
Generate copies of the datasets, perform data cleaning.
"""

# Create a copy of the raw dataset and make column names non-capitalized
# for readability
person = person_raw.copy(deep=True)
person.columns = person.columns.str.lower()
costs = costs_raw.copy(deep=True)
costs.columns = costs.columns.str.lower()

# Asec weights are year-person units, and we average over 3 years,
# so divide by the 3 years to give per year weights.
person.asecwt /= 3

"""
Define CPS variables for analysis and merging
"""

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

"""
Generate copies of the datasets in which to perform simulations
and create a column to specify the relevant scenario/simulation.
Merge the CPS and CAP data for simulations 4-7, copy dataset.
"""

# Create 2 copies of the dataset in which to simulate the policies
person_cc_rep = person.copy()
person_cc_rep_ca = person.copy()

# Create a column to specify scenario/simulation
person["scenario"] = "baseline"
person_cc_rep["scenario"] = "cc_replacement"
person_cc_rep["ca"] = False
person_cc_rep_ca["scenario"] = "cc_replacement"
person_cc_rep_ca["ca"] = True

# Merge the CPS and CAP datasets to produce person_quality with per-child costs
# Creates two rows per person (base and high quality, different costs)
# Note: over_5s do not have a cost or childcare quality
person_quality = person.merge(
    costs[
        [
            "state",
            "high_quality",
            "age_cat",
            "cost",
        ]
    ],
    how="left",
    on=["state", "age_cat"],
)

# Assign scenario to each quality
person_quality_ca = person_quality.copy()
person_quality["scenario"] = np.where(
    person_quality.high_quality, "high_cc_full", "low_cc_full"
)
person_quality_ca["scenario"] = np.where(
    person_quality_ca.high_quality, "high_cc_full_flat", "low_cc_full_flat"
)
person_quality["ca"] = False
person_quality_ca["ca"] = True

# Append/stack/concatenate dataframes
person_sim = pd.concat(
    [
        person,
        person_replace_cost,
        person_flat_transfer,
        person_quality,
        person_quality_flat,
    ],
    ignore_index=True,
)

"""
Aggregate to SPMU level to use household childcare expenditures
"""

# Define data collected at the SPMU level
SPMU_COLS = [
    "spmfamunit",
    "spmwt",
    "spmtotres",
    "spmchxpns",
    "spmthresh",
    "year",
    "state",
    "scenario",
    "ca",
]

# Define columns to be aggregated at the SPMU level
SPMU_AGG_COLS = [
    "child_18",
    "child_6",
    "infant",
    "toddler",
    "preschool",
    "age_6_12",
    "person",
]

# Define a new SPMU-level dataframe with aggregated data
spmu_sim = person_sim.groupby(SPMU_COLS)[SPMU_AGG_COLS].sum()
spmu_sim.columns = ["spmu_" + i for i in SPMU_AGG_COLS]
spmu_sim.reset_index(inplace=True)

"""
Baseline transfer amount (0)
'baseline'
"""

spmu_sim.loc[spmu_sim.scenario == "baseline", "transfer"] = 0

"""
Conduct simulation 1 - replacing childcare expenditure
'cc_replacement'
ca = 'False'
"""

spmu_sim.loc[
    (spmu_sim.scenario == "cc_replacement") & ~spmu_sim.ca, "transfer"
] = spmu_sim.spmchxpns

"""
Conduct simulation 2 - flat child allowance of equal size
"cc_replacement"
ca = "True"
"""

# Use regression to predict actual expenditure per child of a given age
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

# Calculate new income by simulation
spmu_sim.loc[
    (spmu_sim.scenario == "cc_replacement") & spmu_sim.ca, "transfer"
] = (
    child_allowance_amounts.spmu_infant * spmu_sim.spmu_infant
    + child_allowance_amounts.spmu_toddler * spmu_sim.spmu_toddler
    + child_allowance_amounts.spmu_preschool * spmu_sim.spmu_preschool
    + child_allowance_amounts.spmu_age_6_12 * spmu_sim.spmu_age_6_12
)

flat_transfer_cost = mdf.weighted_sum(
    spmu_flat_transfer, "childallowance", "spmwt"
)

"""
Because we do not include an intercept in our regression
(to maintain per-child-category allowances - i.e. to not give families
lump sum additional amounts), the total sum of predicted costs do not sum 
to the summation in of expenditures in the dataset. We therefore
inflate the costs by the cost-ratio.
"""

cost_ratio = program_cost / flat_transfer_cost
spmu_flat_transfer.childallowance *= cost_ratio
spmu_flat_transfer.spmtotres += spmu_flat_transfer.childallowance
true_child_allowance = child_allowance_amounts * cost_ratio


"""
Conduct simulation 3 - base qual replace

We begin by merging the datasets
"""


"""
Conduct simulation 4 - base qual flat
"""

"""
Conduct simulation 5 - high qual replace
"""

"""
Conduct simulation 6 - high qual flat
"""


# Output the dataset (which is housed on Github)
compression_opts = dict(method="gzip", archive_name="person_sim.csv")
person_sim.to_csv(
    "person_sim.csv.gz", index=False, compression=compression_opts
)

"""
SPMU analysis
"""

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

# 7 rows per person single sim.
# Group by SPM ID and scenario. Merge by SPM ID and scenario.
