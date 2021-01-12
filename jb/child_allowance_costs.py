# Roadmap:
# Pull in the edited dataset
# Merge on State index using person-level child age indicator
# Create two rows per index (low and high quality)
# Calculate state-based outcomes.


# Preamble and read data
import microdf as mdf
import pandas as pd
import numpy as np
import us

# Read in census data and specify columns for use
person_raw = pd.read_csv(
    "https://github.com/UBICenter/child-allowance/blob/master/jb/data/cps_00003.csv.gz?raw=true",
    compression="gzip",
    usecols=[
        "YEAR",
        "STATEFIP",
        "AGE",
        "SEX",
        "SPMWT",
        "SPMFTOTVAL",
        "SPMTOTRES",
        "SPMCHXPNS",
        "SPMTHRESH",
        "SPMFAMUNIT",
        "ASECWT",
    ],
)
person = person_raw.copy(deep=True)

# Define child age identifiers
person["person"] = 1
person["child_6"] = person.age < 6
person["infant"] = person.age < 1
person["toddler"] = person.age.between(1, 2)
person["preschool"] = person.age.between(3, 5)

# Age categories for merge
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

# Read in cost data
costs = pd.read_csv(
    "https://github.com/UBICenter/child-allowance/blob/master/jb/data/CCare_cost.csv"
)

# Merge datasets to calculate per-child cost
# Creates two rows per person (one base_quality
# and one high_quality with different costs)
person_costs = person.merge(
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
person_costs.loc[(person_costs.age_cat == "over_5"), "cost"] = 0

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

SPMU_AGG_COLS = ["child_6", "infant", "toddler", "preschool", "person", "cost"]
spmu_quality = person_costs.groupby(SPMU_COLS + ["high_quality"])[
    SPMU_AGG_COLS
].sum()
spmu_quality.columns = ["spmu_" + i for i in SPMU_AGG_COLS]
spmu_quality.reset_index(inplace=True)

# Calculate total cost of transfers, and total number of children
program_cost_high = mdf.weighted_sum(
    spmu_quality[spmu_quality.high_quality], "cost", "spmwt"
)
program_cost_high = mdf.weighted_sum(
    spmu_quality[~spmu_quality.high_quality], "cost", "spmwt"
)

# Program costs also group by age category (sum inf + todd + preschool)
mdf.weighted_sum.groupby

# New microdf groupby argument - have one program cost dataframe - won't need to filter just groupby

# Groupby age_cat and highquality to get the weighted sums we are interested in.

# Get total cost for Max.