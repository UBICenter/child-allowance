import numpy as np
import pandas as pd


run "../../covid/jb/fpuc/convert_asec_taxcalc.py"
run "../../covid/jb/fpuc/max_tax_units.py"

person = pd.read_csv('asec_2016_2018_for_taxcalc.csv.gz')

# Filter to DC.
person = person[person.fips == 11]

# Set columns to lowercase and to 0 or null as appropriate.
prep_ipum(person)
# Add taxid and related fields.
tax_unit_id(person)
# Add other person-level columns in taxcalc form.
person = convert_asec_person_taxcalc(person)

# Create policy reform.


# Run baseline tax calculation.

# Run reform tax calculation.

# Calculate delta(tax) by tax unit.




# Export.

def get_taxes(tu):
    """ Calculates taxes by running taxcalc on a tax unit DataFrame.
    Args:
        tu: Tax unit DataFrame.
    Returns:
        Series with tax liability for each tax unit.
    """
    return mdf.calc_df(records=tc.Records(tu, weights=None, gfactors=None),
                       # year doesn't matter without weights or gfactors.
                       year=2020).tax.values
