import json

def make_nb(cells):
    return {"nbformat":4,"nbformat_minor":0,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":cells}

def cc(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.strip().split("\n")]}

def mc(src):
    return {"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.strip().split("\n")]}

nb1 = make_nb([
    mc("# Notebook 01 - Data Extraction\nloading the raw csv and checking what we are working with"),
    cc("""import pandas as pd
import numpy as np
import json"""),
    cc("""df = pd.read_csv('../data/raw/nyc_parking_uncleaned.csv', dtype=str)
df.columns = df.columns.str.replace(' ', '_')
print("rows:", df.shape[0])
print("columns:", df.shape[1])"""),
    mc("checking how many nulls each column has"),
    cc("""null_report = pd.DataFrame({
    'null_count': df.isnull().sum(),
    'null_pct': (df.isnull().sum() / len(df) * 100).round(2)
})
print(null_report[null_report.null_count > 0])"""),
    mc("looking at unique values in each column to understand the data better"),
    cc("""for col in df.columns:
    print(col, "->", df[col].nunique(), "unique values")
    print("   sample:", list(df[col].dropna().unique()[:4]))
    print()"""),
    mc("saving this audit so we can reference it later in the report"),
    cc("""audit = {
    "total_rows": int(df.shape[0]),
    "total_cols": int(df.shape[1]),
    "columns": list(df.columns),
    "null_counts": df.isnull().sum().to_dict()
}
with open('../docs/cleaning_audit.json', 'w') as f:
    json.dump(audit, f, indent=2)
print("audit saved")"""),
])

nb2 = make_nb([
    mc("# Notebook 02 - Data Cleaning Pipeline\ncleaning the raw dataset step by step"),
    cc("""import pandas as pd
import numpy as np
import json"""),
    mc("loading raw data"),
    cc("""df = pd.read_csv('../data/raw/nyc_parking_uncleaned.csv', dtype=str)
df.columns = df.columns.str.replace(' ', '_')
print("starting with", len(df), "rows and", len(df.columns), "columns")"""),
    mc("step 1 - parsing dates and creating time based columns"),
    cc("""df['Issue_Date'] = pd.to_datetime(df['Issue_Date'], format='%m/%d/%Y', errors='coerce')
df['Month'] = df['Issue_Date'].dt.month
df['Month_Name'] = df['Issue_Date'].dt.month_name()
df['Quarter'] = df['Issue_Date'].dt.quarter
df['Day_of_Week'] = df['Issue_Date'].dt.day_name()
df['Day_Num'] = df['Issue_Date'].dt.dayofweek
df['Week_Num'] = df['Issue_Date'].dt.isocalendar().week
df['Is_Weekend'] = df['Day_Num'].isin([5, 6]).astype(int)
print("dates parsed, missing dates:", df['Issue_Date'].isnull().sum())"""),
    mc("step 2 - fixing fine amounts. negatives and zeros dont make sense for parking fines"),
    cc("""df['Fine_Amount'] = pd.to_numeric(df['Fine_Amount'], errors='coerce')
df.loc[df['Fine_Amount'] <= 0, 'Fine_Amount'] = np.nan
median_by_violation = df.groupby('Violation_Description')['Fine_Amount'].transform('median')
df['Fine_Amount'] = df['Fine_Amount'].fillna(median_by_violation)
df['Fine_Category'] = pd.cut(df['Fine_Amount'], bins=[0, 50, 100, float('inf')], labels=['Low', 'Medium', 'High'])
print("fines fixed, remaining nulls:", df['Fine_Amount'].isnull().sum())"""),
    mc("step 3 - vehicle year validation. removing impossible years like 9999"),
    cc("""df['Vehicle_Year'] = pd.to_numeric(df['Vehicle_Year'], errors='coerce')
df.loc[(df['Vehicle_Year'] == 9999) | (df['Vehicle_Year'] > 2014) | (df['Vehicle_Year'] < 1980), 'Vehicle_Year'] = np.nan
df['Vehicle_Age'] = 2014 - df['Vehicle_Year']
df['Vehicle_Age_Group'] = pd.cut(df['Vehicle_Age'], bins=[-1, 5, 10, 15, float('inf')], labels=['0-5', '6-10', '11-15', '16+'])"""),
    mc("step 4 - standardizing vehicle colors"),
    cc("""color_map = {'GY': 'GRAY', 'BK': 'BLACK', 'WH': 'WHITE'}
df['Vehicle_Color'] = df['Vehicle_Color'].str.upper().replace(color_map)"""),
    mc("step 5 - fixing borough names"),
    cc("""df['Violation_County'] = df['Violation_County'].str.title()"""),
    mc("step 6 - consolidating vehicle makes. keeping top 15 and grouping rest as OTHER"),
    cc("""top_makes = df['Vehicle_Make'].value_counts().nlargest(15).index
df['Vehicle_Make'] = np.where(df['Vehicle_Make'].isin(top_makes), df['Vehicle_Make'], 'OTHER')"""),
    mc("step 7 - grouping body types into broader categories"),
    cc("""passenger = ['SUBN', '4DSD', '2DSD']
commercial = ['VAN', 'DELV', 'PICK']
df['Vehicle_Type_Group'] = 'OTHER'
df.loc[df['Vehicle_Body_Type'].isin(passenger), 'Vehicle_Type_Group'] = 'Passenger'
df.loc[df['Vehicle_Body_Type'].isin(commercial), 'Vehicle_Type_Group'] = 'Commercial'"""),
    mc("step 8 - validating registration states and flagging out of state plates"),
    cc("""valid_states = ['NY', 'NJ', 'PA', 'CT', 'FL', 'MA', 'TX', 'VA', 'MD', 'NC']
df.loc[~df['Registration_State'].isin(valid_states), 'Registration_State'] = 'UNKNOWN'
df['Is_OutOfState'] = (df['Registration_State'] != 'NY').astype(int)
df['State_Group'] = np.where(df['Registration_State'] == 'NY', 'NY', np.where(df['Registration_State'].isin(['NJ', 'CT']), 'Tri-State', 'Other'))"""),
    mc("step 9 - plate category"),
    cc("""df['Plate_Category'] = np.where(df['Plate_Type'] == 'PAS', 'Passenger', 'Other')"""),
    mc("step 10 - cleaning street names and extracting street type"),
    cc("""df['Street_Name'] = df['Street_Name'].str.upper().str.strip()
df['Street_Type'] = df['Street_Name'].str.split().str[-1]"""),
    mc("adding street level aggregations"),
    cc("""street_stats = df.groupby('Street_Name').agg(
    Street_Total_Revenue=('Fine_Amount', 'sum'),
    Street_Ticket_Count=('Summons_Number', 'count')
).reset_index()
df = df.merge(street_stats, on='Street_Name', how='left')"""),
    mc("step 11 - identifying repeat offenders based on plate frequency"),
    cc("""plate_counts = df['Plate_ID'].value_counts()
df['Ticket_Count_Per_Plate'] = df['Plate_ID'].map(plate_counts).fillna(0)
df['Is_Repeat_Offender'] = (df['Ticket_Count_Per_Plate'] >= 2).astype(int)
df['Offender_Tier'] = pd.cut(df['Ticket_Count_Per_Plate'], bins=[0, 2, 5, float('inf')], labels=['1-2', '3-5', '6+'])"""),
    mc("adding violation severity and avoidability flags"),
    cc("""desc_lower = df['Violation_Description'].str.lower()
safety_keywords = ['fire hydrant', 'double parking', 'crosswalk']
df['Violation_Severity'] = np.where(desc_lower.isin(safety_keywords), 'Safety-Critical', 'Standard')
avoidable_keywords = ['no parking', 'street cleaning', 'expired', 'no standing']
df['Is_Avoidable'] = desc_lower.str.contains('|'.join(avoidable_keywords), na=False).astype(int)"""),
    mc("data completeness scoring"),
    cc("""df['Data_Completeness_Score'] = df.notnull().sum(axis=1)
df['Is_Complete_Record'] = (df['Data_Completeness_Score'] > 20).astype(int)"""),
    mc("step 12 - removing duplicates"),
    cc("""df = df.drop_duplicates(subset=['Summons_Number'])
print("after dedup:", len(df), "rows")"""),
    mc("saving transformation log"),
    cc("""log = {
    "step_01": "parsed dates, created month/quarter/weekend columns",
    "step_02": "fixed negative fines, imputed with median per violation",
    "step_03": "removed impossible vehicle years",
    "step_04": "standardized color abbreviations",
    "step_05": "title cased borough names",
    "step_06": "consolidated vehicle makes to top 15",
    "step_07": "grouped body types",
    "step_08": "validated states, added out of state flag",
    "step_09": "cleaned streets, extracted suffix",
    "step_10": "flagged repeat offenders with tiers",
    "step_11": "added severity and avoidability",
    "step_12": "removed duplicate summons numbers"
}
with open('../docs/etl_transformation_log.json', 'w') as f:
    json.dump(log, f, indent=2)"""),
    mc("saving cleaned dataset"),
    cc("""df.to_csv('../data/processed/cleaned.csv', index=False)
print("final shape:", df.shape[0], "rows,", df.shape[1], "columns")
print("done")"""),
])

with open("notebooks/01_extraction.ipynb", "w") as f:
    json.dump(nb1, f, indent=1)
with open("notebooks/02_cleaning.ipynb", "w") as f:
    json.dump(nb2, f, indent=1)
print("01 and 02 written")
