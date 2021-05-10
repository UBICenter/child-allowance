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

# Calculate new income by simulation
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

compression_opts = dict(method="gzip", archive_name="person_sim.csv")
person_sim.to_csv(
    "person_sim.csv.gz", index=False, compression=compression_opts
)
