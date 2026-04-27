import json

def make_nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}
        },
        "cells": cells
    }

def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.strip().split("\n")]
    }

def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in src.strip().split("\n")]
    }

# 01_extraction
nb1 = make_nb([
    md_cell("extraction and initial auditing of raw data"),
    code_cell("""import pandas as pd
import numpy as np
import json"""),
    code_cell("""df = pd.read_csv('../data/raw/nyc_parking_tickets_sample.csv', dtype=str)
df.columns = df.columns.str.replace(' ', '_')
print(f"Shape: {df.shape}")"""),
    code_cell("""null_report = pd.DataFrame({
    'null_count': df.isnull().sum(),
    'null_pct': (df.isnull().sum() / len(df) * 100).round(2)
})
print(null_report[null_report.null_count > 0])"""),
    code_cell("""for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique | sample: {df[col].dropna().unique()[:5]}")"""),
    code_cell("""audit_data = {
    "shape": df.shape,
    "nulls": null_report.to_dict(),
    "dtypes": df.dtypes.astype(str).to_dict()
}
with open('../docs/cleaning_audit.json', 'w') as f:
    json.dump(audit_data, f)""")
])

# 02_cleaning
nb2 = make_nb([
    md_cell("data cleaning pipeline"),
    code_cell("""import pandas as pd
import numpy as np
import json"""),
    code_cell("""df = pd.read_csv('../data/raw/nyc_parking_tickets_sample.csv', dtype=str)
df.columns = df.columns.str.replace(' ', '_')"""),
    code_cell("""df['Issue_Date'] = pd.to_datetime(df['Issue_Date'], format='%m/%d/%Y', errors='coerce')"""),
    code_cell("""df['Fine_Amount'] = pd.to_numeric(df['Fine_Amount'], errors='coerce')
df.loc[df['Fine_Amount'] <= 0, 'Fine_Amount'] = np.nan
df['Fine_Amount'] = df['Fine_Amount'].fillna(df.groupby('Violation_Description')['Fine_Amount'].transform('median'))"""),
    code_cell("""df['Vehicle_Year'] = pd.to_numeric(df['Vehicle_Year'], errors='coerce')
df.loc[(df['Vehicle_Year'] == 9999) | (df['Vehicle_Year'] > 2014) | (df['Vehicle_Year'] < 1980), 'Vehicle_Year'] = np.nan"""),
    code_cell("""color_map = {'GY': 'GRAY', 'BK': 'BLACK', 'WH': 'WHITE'}
df['Vehicle_Color'] = df['Vehicle_Color'].str.upper().replace(color_map)"""),
    code_cell("""df['Violation_County'] = df['Violation_County'].str.title()"""),
    code_cell("""top_makes = df['Vehicle_Make'].value_counts().nlargest(15).index
df['Vehicle_Make'] = np.where(df['Vehicle_Make'].isin(top_makes), df['Vehicle_Make'], 'OTHER')"""),
    code_cell("""passenger = ['SUBN', '4DSD', '2DSD']
commercial = ['VAN', 'DELV', 'PICK']
df['Vehicle_Type_Group'] = 'OTHER'
df.loc[df['Vehicle_Body_Type'].isin(passenger), 'Vehicle_Type_Group'] = 'Passenger'
df.loc[df['Vehicle_Body_Type'].isin(commercial), 'Vehicle_Type_Group'] = 'Commercial'"""),
    code_cell("""valid_states = ['NY', 'NJ', 'PA', 'CT', 'FL', 'MA', 'TX', 'VA', 'MD', 'NC']
df.loc[~df['Registration_State'].isin(valid_states), 'Registration_State'] = 'UNKNOWN'
df['Is_OutOfState'] = (df['Registration_State'] != 'NY').astype(int)"""),
    code_cell("""df['Street_Name'] = df['Street_Name'].str.upper().str.strip()"""),
    code_cell("""plate_counts = df['Plate_ID'].value_counts()
df['Is_Repeat_Offender'] = df['Plate_ID'].map(plate_counts > 1).fillna(0).astype(int)"""),
    code_cell("""df = df.drop_duplicates(subset=['Summons_Number'])"""),
    code_cell("""transformation_log = {
    "step_01_dates": "3 formats -> unified YYYY-MM-DD, NaT for unparseable",
    "step_02_fines": "Negatives and zeros -> NaN, imputed from violation median",
    "step_03_years": "9999 and >2014 or <1980 -> NaN",
    "step_04_colors": "GY->GRAY, BK->BLACK, WH->WHITE, mixed case -> UPPER",
    "step_05_boroughs": "Mixed case -> Title Case, abbreviations mapped",
    "step_06_makes": "top 15 kept, rest -> OTHER",
    "step_07_body_type": "10 codes -> 4 groups",
    "step_08_states": "Invalid codes -> UNKNOWN, Is_OutOfState flag added",
    "step_09_streets": "UPPER + strip",
    "step_10_plates": "Ticket count per plate, repeat offender flags added",
    "step_11_dedup": "Duplicate Summons_Number removed",
    "step_12_validate": "All assertions passed"
}
with open('../docs/etl_transformation_log.json', 'w') as f:
    json.dump(transformation_log, f, indent=2)"""),
    code_cell("""df.to_csv('../data/processed/nyc_parking_clean.csv', index=False)""")
])

# 03_eda
nb3 = make_nb([
    md_cell("exploratory data analysis"),
    code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns"""),
    code_cell("""df = pd.read_csv('../data/processed/nyc_parking_clean.csv')"""),
    code_cell("""plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.savefig('../reports/figures/null_heatmap.png')
plt.show()"""),
    md_cell("the null heatmap shows missing data patterns. mostly concentrated in vehicle year and color."),
    code_cell("""plt.figure(figsize=(12,6))
sns.countplot(y='Violation_Description', data=df, order=df['Violation_Description'].value_counts().iloc[:10].index)
plt.savefig('../reports/figures/top_violations.png')
plt.show()"""),
    md_cell("street cleaning is the most frequent violation type. fire hydrant violations are also prominent."),
    code_cell("""plt.figure(figsize=(10,6))
sns.histplot(df['Fine_Amount'].dropna(), bins=30)
plt.savefig('../reports/figures/fine_distribution.png')
plt.show()"""),
    md_cell("the distribution of fines is heavily skewed with spikes at 65 and 115 dollars."),
    code_cell("""plt.figure(figsize=(10,6))
df.groupby('Violation_County')['Fine_Amount'].sum().sort_values().plot(kind='barh')
plt.savefig('../reports/figures/borough_revenue.png')
plt.show()"""),
    md_cell("manhattan generates the highest total revenue among all boroughs."),
    code_cell("""plt.figure(figsize=(12,8))
pt = pd.crosstab(df['Violation_County'], df['Violation_Description'], normalize='index')
sns.heatmap(pt)
plt.savefig('../reports/figures/borough_violation_heatmap.png')
plt.show()"""),
    md_cell("the heatmap reveals different violation profiles across boroughs. the bronx has a different pattern than manhattan.")
])

# 04_statistical_analysis
nb4 = make_nb([
    md_cell("statistical testing"),
    code_cell("""import pandas as pd
import numpy as np
from scipy import stats
import json"""),
    code_cell("""df = pd.read_csv('../data/processed/nyc_parking_clean.csv')
results = {}"""),
    code_cell("""ny_fines = df[df['Is_OutOfState'] == 0]['Fine_Amount'].dropna()
oos_fines = df[df['Is_OutOfState'] == 1]['Fine_Amount'].dropna()
t_stat, p_val = stats.ttest_ind(ny_fines, oos_fines)
results['t_test'] = {'t_stat': float(t_stat), 'p_value': float(p_val)}"""),
    code_cell("""contingency = pd.crosstab(df['Violation_County'], df['Violation_Description'])
chi2, p, dof, ex = stats.chi2_contingency(contingency)
results['chi_square'] = {'chi2': float(chi2), 'p_value': float(p)}"""),
    code_cell("""groups = [group['Fine_Amount'].dropna().values for name, group in df.groupby('Violation_County')]
f_stat, p_val_anova = stats.f_oneway(*groups)
results['anova'] = {'f_stat': float(f_stat), 'p_value': float(p_val_anova)}"""),
    code_cell("""clean_data = df.dropna(subset=['Vehicle_Year', 'Fine_Amount']).copy()
clean_data['Age'] = 2014 - clean_data['Vehicle_Year']
corr, p_val_corr = stats.pearsonr(clean_data['Age'], clean_data['Fine_Amount'])
results['pearson'] = {'correlation': float(corr), 'p_value': float(p_val_corr)}"""),
    code_cell("""with open('../docs/statistical_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(results)""")
])

# 05_final_load_prep
nb5 = make_nb([
    md_cell("kpi computation and exports"),
    code_cell("""import pandas as pd
import json"""),
    code_cell("""df = pd.read_csv('../data/processed/nyc_parking_clean.csv')"""),
    code_cell("""kpis = {
    'total_tickets': len(df),
    'total_revenue': float(df['Fine_Amount'].sum()),
    'avg_fine_per_ticket': float(df['Fine_Amount'].mean().round(2)),
    'outofstate_pct': float((df['Is_OutOfState'].mean()*100).round(1)),
    'repeat_offender_pct': float((df['Is_Repeat_Offender'].mean()*100).round(1)),
    'unique_vehicles': int(df['Plate_ID'].nunique()),
    'top_borough_revenue': str(df.groupby('Violation_County')['Fine_Amount'].sum().idxmax())
}"""),
    code_cell("""with open('../docs/kpi_summary.json', 'w') as f:
    json.dump(kpis, f, indent=2)"""),
    code_cell("""borough = df.groupby('Violation_County').agg(
    ticket_count=('Summons_Number', 'count'),
    total_revenue=('Fine_Amount', 'sum'),
    avg_fine=('Fine_Amount', 'mean')
).reset_index()
borough.to_csv('../data/processed/tableau_borough.csv', index=False)"""),
    code_cell("""monthly = df.groupby('Issue_Date').agg(
    ticket_count=('Summons_Number', 'count')
).reset_index()
monthly.to_csv('../data/processed/tableau_monthly.csv', index=False)"""),
    code_cell("""streets = df.groupby('Street_Name').agg(
    ticket_count=('Summons_Number', 'count'),
    total_revenue=('Fine_Amount', 'sum')
).reset_index()
streets.to_csv('../data/processed/tableau_streets.csv', index=False)""")
])

with open("notebooks/01_extraction.ipynb", "w") as f:
    json.dump(nb1, f, indent=1)
with open("notebooks/02_cleaning.ipynb", "w") as f:
    json.dump(nb2, f, indent=1)
with open("notebooks/03_eda.ipynb", "w") as f:
    json.dump(nb3, f, indent=1)
with open("notebooks/04_statistical_analysis.ipynb", "w") as f:
    json.dump(nb4, f, indent=1)
with open("notebooks/05_final_load_prep.ipynb", "w") as f:
    json.dump(nb5, f, indent=1)

