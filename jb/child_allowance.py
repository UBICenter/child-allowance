# To do:
# Low quality still has too few observations (likely state
# match issue)
# Logistic regression of childcare expenses on demographics.
#

# Packages
import microdf as mdf
import pandas as pd
import numpy as np
import us
import statsmodels.api as sm


# Read in the CPS (census) and CAP datasets

# Read in CPS data and specify columns for use
person_raw = pd.read_csv(
    "jb/data/cps_00003.csv.gz",
    compression="gzip",
    usecols=[
        "YEAR",
        "STATEFIP",
        "AGE",
        "SEX",
        "ASECWT",
        "SPMWT",
        "SPMFTOTVAL",
        "SPMTOTRES",
        "SPMCHXPNS",
        "SPMTHRESH",
        "SPMFAMUNIT",
        "ASECWT",
        "RACE",
        "HISPAN",
    ],
)

# Read in CAP dataset
costs_raw = pd.read_csv("jb/data/CCare_cost.csv")

# Generate copies of the datasets, perform data cleaning.

# Create a copy of the raw dataset and make column names non-capitalized
# for readability
person = person_raw.copy()
person.columns = person.columns.str.lower()
costs = costs_raw.copy()
costs.columns = costs.columns.str.lower()

# We average over 3 years so divide by 3 to give per-year weights
person.asecwt /= 3
person.spmwt /= 3

# Define CPS variables for analysis and merging

# Define child age identifiers
person["child_6"] = person.age < 6
person["child_18"] = person.age < 18
person["infant"] = person.age < 1
person["toddler"] = person.age.between(1, 2)
person["preschool"] = person.age.between(3, 5)
person["age_6_12"] = person.age.between(6, 12)
person["person"] = 1

# Define child age categories for group-by analysis and merge
person["age_cat"] = "adult"
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


# Create 2 copies of the dataset in which to simulate
# policies based on child expenditure (CPS)
person_cc_rep = person.copy()
person_cc_rep_ca = person.copy()

# Create a column to specify scenario/simulation
person["scenario"] = "baseline"
person["ca"] = False
person_cc_rep["scenario"] = "cc_replacement"
person_cc_rep["ca"] = False
person_cc_rep_ca["scenario"] = "cc_replacement"
person_cc_rep_ca["ca"] = True


# Merge the CPS and CAP datasets to produce person_quality with per-child costs
# Creates two rows per person (base and high quality, different costs)
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
person_quality["scenario"] = np.where(
    person_quality.high_quality, "high_cc_full", "low_cc_full"
)
person_quality_ca = person_quality.copy()
person_quality["ca"] = False
person_quality_ca["ca"] = True

# Append/stack/concatenate dataframes
person_sim = pd.concat(
    [
        person,
        person_cc_rep,
        person_cc_rep_ca,
        person_quality,
        person_quality_ca,
    ],
    ignore_index=True,
)


# In the following code, we begin simulating the various policies.

# Simulations 3 and 5 are conducted first and grouped together
# as the transfer amount is simply set to the amount set by the
# CAP estimate at the person level.

# Conduct simulation - 3
# base_cc_full, ca = False

person_sim.loc[
    (person_sim.scenario == "low_cc_full"),
    "transfer",
] = person_sim.cost

# Conduct simulation 5 - high quality full take-up
# high_cc_full, ca = False

person_sim.loc[
    (person_sim.scenario == "high_cc_full"),
    "transfer",
] = person_sim.cost


# Simulations 4 and 6 are similarly grouped together as
# we need to estimate per-child-age-quality costs at the
# person level.

# Conduct simulation 4 - base quality flat transfer
# base_cc_full, ca = True

# Conduct simulation 6 - high quality flat transfer
# high_cc_full, ca = True

# Define function to calculate child cost by age, qual, weighted by asecwt
def tot_cost(group):
    return mdf.weighted_sum(
        person_quality, ["cost", "person"], "asecwt", groupby=group
    ).reset_index()


# Use function to generate dataframe
qual_cost = tot_cost(["age_cat", "high_quality"])
# Add column of per-child costs
qual_cost["per_child"] = qual_cost.cost / qual_cost.person


### Easier to merge to person sim on age_cat and high_quality
ages = ["infant", "toddler", "preschool"]
for x in ages:
    person_sim.loc[
        (person_sim.scenario == "low_cc_full_flat")
        & person_sim.ca
        & (person_sim.high_quality == 0)
        & (person_sim.age_cat == x),
        "transfer",
    ] = qual_cost.loc[
        (qual_cost.age_cat == x) & (person_sim.high_quality == 0), "per_child"
    ]
    person_sim.loc[
        (person_sim.scenario == "high_cc_full_flat")
        & person_sim.ca
        & (person_sim.high_quality == 1)
        & (person_sim.age_cat == x),
        "transfer",
    ] = qual_cost.loc[
        (qual_cost.age_cat == x) & (person_sim.high_quality == 1), "per_child"
    ]

# For simulations 1 and 2, we need to aggregate at the SPMU level, so
# again we group them below and also specify the baseline dataset for
# clarity.

# Create a dummy for whether an spmu has child care expenses
person_sim["anyspmchxpns"] = person_sim.spmchxpns > 0

# Aggregate to SPMU level to use household childcare expenditures
# Define data collected at the SPMU level
SPMU_COLS = [
    "spmfamunit",
    "spmwt",
    "spmtotres",
    "spmchxpns",
    "anyspmchxpns",
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

# Conduct simulation 0 - baseline
# Transfer amount (0)
# baseline, ca = False

spmu_sim.loc[spmu_sim.scenario == "baseline", "transfer"] = 0

# Conduct simulation 1 - replacing childcare expenditure
# cc_replacement, ca = False

spmu_sim.loc[
    (spmu_sim.scenario == "cc_replacement") & ~spmu_sim.ca, "transfer"
] = spmu_sim.spmchxpns

# Conduct simulation 2 - flat child allowance of equal size
# cc_replacement, ca = True

# Use regression to predict actual expenditure per child of a given age
spmu = spmu_sim[spmu_sim.scenario == "baseline"]
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

initial_replacement_ca_cost = mdf.weighted_sum(
    spmu_sim[(spmu_sim.scenario == "cc_replacement") & spmu_sim.ca],
    "transfer",
    "spmwt",
)

# Because we do not include an intercept in our regression
# (to maintain per-child-category allowances - i.e. to not give families
# lump sum additional amounts), the total sum of predicted costs do not sum
# to the summation in of expenditures in the dataset. We therefore
# inflate the costs by the cost-ratio.

cost_ratio = program_cost / initial_replacement_ca_cost
spmu_sim.loc[
    (spmu_sim.scenario == "cc_replacement") & spmu_sim.ca, "transfer"
] *= cost_ratio
true_child_allowance = child_allowance_amounts * cost_ratio

# Add the simulated transfer amounts to totres to give the policy impact
# on household resources

# Add transfer to SPM resources
spmu_sim.spmtotres += spmu_sim.transfer

# Create poverty flags on simulated incomes
# Thresholds take into account household size and local property value
spmu_sim["poverty_flag"] = spmu_sim.spmtotres < spmu_sim.spmthresh
spmu_sim["deep_poverty_flag"] = spmu_sim.spmtotres < spmu_sim.spmthresh / 2

# Merge back to person_sim to replace spmtotres.
SPM_SIM_IDS = [
    "spmfamunit",
    "scenario",
    "ca",
    "year",
]
###### Does the inclusion of the poverty flags here work?
person_sim = person_sim.drop(columns="spmtotres", axis=1).merge(
    spmu_sim[
        SPM_SIM_IDS
        + ["spmtotres", "poverty_flag", "deep_poverty_flag", "anyspmchxpns"]
    ],
    on=SPM_SIM_IDS,
)

# Output the dataset
compression_opts = dict(method="gzip", archive_name="person_sim.csv")
person_sim.to_csv(
    "person_sim.csv.gz", index=False, compression=compression_opts
)
