"""
Roadmap:
Check what states are dropped - e.g. Puerto Rico - Use 
US average for those states.
Pull in the edited dataset
Merge on State index using person-level child age indicator
Create two rows per index (low and high quality)
Calculate state-based outcomes.
Poverty Gap analysis?
"""

# Preamble and read data
import microdf as mdf
import pandas as pd
import numpy as np
import us

# Read in census data and specify columns for use
person_raw = pd.read_csv(
    "https://github.com/UBICenter/child-allowance/blob/master/jb/data/cps_00003.csv.gz?raw=true",  # noqa
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

# Create a copy of the raw dataset and make column names non-capitalized
# for readability
person = person_raw.copy(deep=True)
person.columns = person.columns.str.lower()

# Asec weights are year-person units, and we average over 3 years,
# so divide by the 3 years to give per year weights.
person.asecwt /= 3

# Define child age categories for group-by analysis and merge
person["age_cat"] = "over_5"
person.loc[(person.age < 1), "age_cat"] = "infant"
person.loc[(person.age.between(1, 2)), "age_cat"] = "toddler"
person.loc[(person.age.between(3, 5)), "age_cat"] = "preschool"
person["child_6"] = person.age < 6

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

costs = pd.read_csv(
    "C:\\Users\\John Walker\\Desktop\\CCare_cost.csv",
)

# Merge datasets to calculate per-child cost
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

# Set over_5 cost of childcare to 0
person_quality.loc[(person_quality.age_cat == "over_5"), "cost"] = 0

# Calculate total cost and number of children by age_cat 
def tot_num_cost(group):
    return mdf.weighted_sum(person_quality,["cost", "person"],"asecwt",groupby=group).reset_index()

#  Calculate cost for all children
num_cost = tot_num_cost(["age_cat", "high_quality"])

# Calculate cost per child
num_cost["cost_per_child"] = num_cost.cost/num_cost.person
person_quality = person_quality.merge(num_cost[["age_cat","high_quality","cost_per_child"]], on=["age_cat", "high_quality"])

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

# Define columns to be aggregated at SPMU level
SPMU_AGG_COLS = ["child_6", "infant", "toddler", "preschool", "person", "cost", "cost_per_child"]

# Aggregate at SPMU level by high_quality (cost)
spmu_quality = person_quality.groupby(SPMU_COLS + ["high_quality", "state"])[
    SPMU_AGG_COLS
].sum()
spmu_quality.columns = ["spmu_" + i for i in SPMU_AGG_COLS]
spmu_quality.reset_index(inplace=True)

"""
We are assuming 100% takeup rate.
Here, we simulate two policy extremes twice, using both a) base quality 
and b) high quality to determine total program cost:
1) a flat transfer by state equal to the average state-based cost of childcare
to a given household. i.e. a US-wide transfer that varies by state cost and 
household number of children; and
2) a flat transfer equal to the average USA cost of childcare for the average
household. 
"""

# Create copies of the dataset in which to simulate the policies
spmu_quality_state = spmu_quality.copy(deep=True)
spmu_quality_us = spmu_quality.copy(deep=True)

# Generate scenario flags to separate datasets
spmu_quality_state["sim_flag"] = "state"
spmu_quality_us["sim_flag"] = "US"

# Calculate cost of the policies
"""
TO DO: EDIT FROM HERE

Does the following weighting make sense to sum by age group?
"""

tot_cost = mdf.weighted_sum(spmu_quality,"spmu_cost_per_child",["spmwt","spmu_toddler","spmu_infant","spmu_preschool"],groupby="high_quality")

# Need total cost by infants v toddlers etc.
spmu_quality_state.spmftotval += spmu_quality_state.spmu_cost_per_child
spmu_quality_state.spmftotval += tot_cost

# Calculate number of children per household 

# Define function to calculate state-based program cost
def state_cost(age, qual):
    return (
        mdf.weighted_sum(spmu_quality[(spmu_quality.age_cat == "age") & (spmu_quality.high_quality == qual)],
        "cost",
        "spmwt",
        groupby="state",   
)

# Calculate total program cost by state by age_cat
program_cost_high_state_inf = state_cost(infant, 1)
program_cost_high_state_tod = state_cost(toddler, 1)
program_cost_high_state_pre = state_cost(preschool, 1)
program_cost_low_state_inf = state_cost(infant, 0)
program_cost_low_state_tod = state_cost(toddler, 0)
program_cost_low_state_pre = state_cost(preschool, 0)


##### GROUP BY AGE CAT AND STATE FOR STATE LEVEL
### FOR US SET state == "United States"

# Calculate total number of children
total_child_6 = mdf.weighted_sum(
    spmu_quality[spmu_quality.high_quality], "spmu_child_6", "spmwt"
)

total_child_6 = mdf.weighted_sum(
    spmu_quality[spmu_quality.high_quality], "spmu_child_6", "spmwt"
)

# Generate a flag for SPMU units living in poverty
spmu_sim["poverty_flag"] = spmu_sim.spmftotval < spmu_sim.spmthresh


def pov(groupby, data=person_sim):
    return (
        data.groupby(groupby)
        .apply(lambda x: mdf.weighted_mean(x, "poverty_flag", "asecwt"))
        .reset_index()
    )


# Define various poverty rates
poverty_rate = pov("sim_flag")
poverty_rate_sex = pov(["sim_flag", "sex"])
poverty_rate_race_hispan = pov(["sim_flag", "race_hispan"])
poverty_rate_state = pov(["sim_flag", "state"])
poverty_rate_child = pov("sim_flag", person_sim[person_sim.child_6])


# Program costs also group by age category (sum inf + todd + preschool)
mdf.weighted_sum.groupby(
    spmu_quality[spmu_quality.high_quality], "age_cat", "cost", "spmwt"
)
mdf.weighted_sum.groupby(
    spmu_quality[~spmu_quality.high_quality], "age_cat", "cost", "spmwt"
)

# Groupby age_cat and highquality to get the weighted sums we are
# interested in.


# Output the dataset for analysis 
compression_opts = dict(method="gzip", archive_name="person_costs_sim.csv")
person_sim.to_csv(
    "person_costs_sim.csv.gz", index=False, compression=compression_opts
)

