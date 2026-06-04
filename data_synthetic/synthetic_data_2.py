import pandas as pd
import numpy as np
import random

# Seed synthetic data generation for reproducible demo outputs.
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------
# 1. GENERATE SYNTHETIC 'WBS_Funding_Splits.csv' (synthetic funding reference data)
# ---------------------------------------------------------
wbs_codes = ['WBS-1.10.01', 'WBS-1.10.02', 'WBS-2.05.01', 'WBS-2.05.02', 'WBS-3.01.01', 'WBS-3.02.04', 'WBS-4.04.01', 'WBS-4.04.02']
wbs_titles = [f"Programmatic Mission Category {code}" for code in wbs_codes]
fed_pct = [random.choice([0, 25, 50, 75, 100]) for _ in wbs_codes]
state_pct = [random.choice([0, 25, 50, 75, 100]) for _ in wbs_codes]
discretionary_pct = [random.choice([0, 25, 50, 75, 100]) for _ in wbs_codes]

df_doe_wbs = pd.DataFrame({
    'WBS_Code': wbs_codes,
    'Program_Title': wbs_titles,
    'Program_Description': [f"NNSA/DOE Strategic funding description for {code}" for code in wbs_codes],
    'Fed_Type_A': fed_pct,
    'Fed_Type_B': state_pct,
    'Lab_Type_A': discretionary_pct
})

# ---------------------------------------------------------
# 2. GENERATE SYNTHETIC 'FIMS_Inventory.csv' (synthetic facility inventory data)
# ---------------------------------------------------------
num_facilities = 35
# FIMS Property IDs are unique 7-digit integers
fims_property_ids = sorted(random.sample(range(1000000, 9999999), num_facilities))
cities = ['Oak Ridge', 'Los Alamos', 'Livermore', 'Richland', 'Argonne', 'Sandia-Albuquerque']
states = ['Tennessee', 'New Mexico', 'California', 'Washington', 'Illinois']
site_names = ['Y-12 National Security Complex', 'Los Alamos National Lab', 'Lawrence Livermore National Lab', 'Hanford Site', 'Argonne National Lab', 'Sandia National Lab']

df_doe_inventory = pd.DataFrame({
    'FIMS_Record_ID': [random.randint(10000, 99999) for _ in fims_property_ids],
    'City': [random.choice(cities) for _ in fims_property_ids],
    'State': [random.choice(states) for _ in fims_property_ids],
    'Property_ID': fims_property_ids,
    'Asset_Name': [f"Research Facility {i}" for i in range(1, num_facilities + 1)],
    'Asset_Description': [f"Synthetic experimental test asset {i}" for i in range(1, num_facilities + 1)],
    'Site_Name': [random.choice(site_names) for _ in fims_property_ids],
    'Site_Code': [f"DOE-{random.randint(100, 999)}" for _ in fims_property_ids],
    'Facility_Class_Code': [f"FC{random.randint(1000, 9999)}" for _ in fims_property_ids],
    'Asset_Type': ['Building'] * num_facilities,
    'Interest_Type': ['FEE'] * num_facilities,
    'Operational_Status': ['Active'] * num_facilities,
    'FIMS_Status': ['Reportable'] * num_facilities,
    'Acquisition_Date': ['1-Jan-80'] * num_facilities,
    'Acquisition_Method': ['Constructed'] * num_facilities,
    'RPV': np.random.randint(5000000, 150000000, size=num_facilities), # Replacement Plant Value (synthetic replacement value)
    'Construction_Type': ['PERM'] * num_facilities,
    'WBS_Element': [random.choice(wbs_codes) for _ in fims_property_ids],
    'Space_ID': ['001'] * num_facilities,
    'Space_Name': ['Lab Suite 01'] * num_facilities,
    'Category_Code': [random.randint(110000, 999000) for _ in fims_property_ids],
    'FAC_Code': [random.randint(1100, 9999) for _ in fims_property_ids],
    'Primary_UOM': ['GSF'] * num_facilities, # Gross Square Feet
    'Primary_Qty': np.random.randint(5000, 250000, size=num_facilities),
    'Secondary_UOM': ['PN'] * num_facilities,
    'Secondary_Qty': [0] * num_facilities,
    'Build_Date': ['1-Jan-80'] * num_facilities,
    'Placed_in_Service_Date': ['1-Jan-80'] * num_facilities,
    'Sustainment_Fund_Code': [f"DOE-NNSA-ST{random.randint(1, 9)}" for _ in fims_property_ids],
    'Sustainment_Org': ['NNSA-NA50'] * num_facilities,
    'Shared_Use_Code': ['P'] * num_facilities,
    'User_Org_Code': ['NNSA'] * num_facilities,
    'City_and_State': ['FakeCity, FakeState'] * num_facilities,
    'Form_Label': ['Building'] * num_facilities
})

# ---------------------------------------------------------
# 3. GENERATE SYNTHETIC 'CAIS_Deficiencies.csv' (synthetic condition deficiency data)
# ---------------------------------------------------------
cais_categories = ['Foundations & Substructure', 'Superstructure & Walls', 'Roofing Systems', 'Interior Finishes', 'Plumbing Systems', 'HVAC & Mechanical', 'Electrical Systems']
cais_subtypes = ['Reinforced Concrete Repair', 'High-Performance Glazing', 'EPDM Membrane Roof', 'Epoxy Floor Coating', 'Chilled Water Loop', 'Air Handling Unit', 'High-Voltage Switchgear']
work_types = ['Repair Deficiency', 'Replace System', 'Sustainment Paint']

fake_deficiencies = []
for pid in fims_property_ids:
    num_deficiencies = random.randint(3, 10)
    for _ in range(num_deficiencies):
        cat = random.choice(cais_categories)
        sub = random.choice(cais_subtypes)
        w_type = random.choice(work_types)
        cost = float(np.random.randint(2500, 750000)) # DOE scales are often larger
        fake_deficiencies.append({
            'Asset_Name': f"Research Facility {pid}",
            'Property_ID': pid,
            'Uniformat_Category': cat,
            'Component_Type': sub,
            'Correction_Work_Type': w_type,
            'Fiscal_Year': random.choice([2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]),
            'Correction_Cost': cost,
            'API': random.randint(1, 100), # Asset Priority Index
            'Current_FCI': round(np.random.uniform(0.3, 0.98), 4), # Facility Condition Index (decimal)
            'Projected_FCI': round(np.random.uniform(0.3, 0.98), 4),
            'Current_RSL': random.randint(-5, 30),
            'Projected_RSL': random.randint(-5, 30),
            'Quantity': random.randint(1, 10000),
            'UoM': random.choice(['SF', 'EA', 'LF', 'KV']),
            'Assessment_Method': 'Direct Visual',
            'Last_Inspected_Rating': random.randint(60, 100),
            'Assessor_Comments': 'Synthetic DOE Assessor Comment',
            'CAIS_Asset_ID': f"CAIS-{random.randint(100, 999)}",
            'Section_ID': f"SEC-{random.randint(1, 5)}"
        })

df_doe_cais = pd.DataFrame(fake_deficiencies)

# ---------------------------------------------------------
# 4. GENERATE MASTER 'FIMS_Master_Scenario_Cost.csv' (synthetic scenario cost data)
# ---------------------------------------------------------
cais_merged = df_doe_cais.copy()
fims_inv = df_doe_inventory.copy()
wbs_splits = df_doe_wbs.copy()

cais_merged = cais_merged.merge(fims_inv[['Property_ID', 'WBS_Element']], on='Property_ID', how='left')
cais_merged = cais_merged.merge(wbs_splits[['WBS_Code', 'Fed_Type_A', 'Fed_Type_B', 'Lab_Type_A']], left_on='WBS_Element', right_on='WBS_Code', how='left')
# Group by WBS and Property_ID to get 10-year master cost table
doe_bldr = cais_merged.groupby(['WBS_Element', 'Property_ID'])['Correction_Cost'].sum().reset_index().rename(columns={'Correction_Cost': 'Correction_Cost_10YR'})

# ---------------------------------------------------------
# 5. WRITE OUT THE SCRUBBED REPLICAS
# ---------------------------------------------------------
df_doe_wbs.to_csv("WBS_Funding_Splits.csv", index=False)
df_doe_inventory.to_csv("FIMS_Inventory.csv", index=False)
df_doe_cais.to_csv("CAIS_Deficiencies.csv", index=False)
doe_bldr.to_csv("FIMS_Master_Scenario_Cost.csv", index=False)

print("SUCCESS: 4 synthetic demonstration files written to active directory!")
