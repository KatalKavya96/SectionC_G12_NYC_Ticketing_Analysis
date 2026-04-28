import json

def make_nb(cells):
    return {"nbformat":4,"nbformat_minor":0,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":cells}

def cc(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.strip().split("\n")]}

def mc(src):
    return {"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.strip().split("\n")]}

nb4 = make_nb([
    mc("# Notebook 04 - Statistical Analysis\nrunning hypothesis tests on the cleaned data to back up our findings with numbers"),
    cc("""import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.seasonal import seasonal_decompose
import json

df = pd.read_csv('../data/processed/cleaned.csv')
results = {}
print("loaded", len(df), "rows")"""),

    mc("### test 1 - independent t-test\nare out of state drivers getting fined differently than ny drivers?\n\nh0: average fine for ny plates = average fine for out of state plates\n\nh1: they are different"),
    cc("""ny_fines = df[df['Is_OutOfState'] == 0]['Fine_Amount'].dropna()
oos_fines = df[df['Is_OutOfState'] == 1]['Fine_Amount'].dropna()
t_stat, p_val = stats.ttest_ind(ny_fines, oos_fines)
pooled_std = np.sqrt((ny_fines.std()**2 + oos_fines.std()**2) / 2)
cohens_d = (oos_fines.mean() - ny_fines.mean()) / pooled_std
results['t_test'] = {'t_stat': round(float(t_stat), 4), 'p_value': round(float(p_val), 6), 'cohens_d': round(float(cohens_d), 4)}
print("t-stat:", round(t_stat, 4), "p-value:", round(p_val, 6))
print("ny avg:", round(ny_fines.mean(), 2), "oos avg:", round(oos_fines.mean(), 2))"""),
    mc("if p value is less than 0.05 we reject the null hypothesis. this tells us whether the city charges out of state drivers differently or if they just happen to commit different types of violations."),

    mc("### test 2 - chi square test\nis the type of violation independent of which borough it happens in?\n\nh0: violation type and borough are independent\n\nh1: they are dependent"),
    cc("""contingency = pd.crosstab(df['Violation_County'], df['Violation_Description'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
n = contingency.sum().sum()
k = min(contingency.shape) - 1
cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0
results['chi_square'] = {'chi2': round(float(chi2), 2), 'p_value': float(p), 'cramers_v': round(float(cramers_v), 4), 'dof': int(dof)}
print("chi2:", round(chi2, 2), "p-value:", p, "cramers v:", round(cramers_v, 4))"""),
    mc("a significant result here means different boroughs have fundamentally different violation profiles. this is important because it means the city cannot use a one size fits all enforcement strategy."),

    mc("### test 3 - one way anova with tukey hsd\ndo boroughs differ in their average fine amounts?\n\nh0: all boroughs have the same mean fine\n\nh1: at least one borough is different"),
    cc("""borough_fines = df[['Violation_County', 'Fine_Amount']].dropna()
groups = [g['Fine_Amount'].values for _, g in borough_fines.groupby('Violation_County')]
f_stat, p_anova = stats.f_oneway(*groups)
results['anova'] = {'f_stat': round(float(f_stat), 4), 'p_value': float(p_anova)}
print("f-stat:", round(f_stat, 4), "p-value:", p_anova)

tukey = pairwise_tukeyhsd(borough_fines['Fine_Amount'], borough_fines['Violation_County'], alpha=0.05)
results['tukey_significant_pairs'] = int(sum(tukey.reject))
print("significant pairwise differences:", sum(tukey.reject))"""),
    mc("anova tells us if there is any difference between groups. tukey hsd then tells us exactly which boroughs differ from each other. this helps us identify which boroughs might need adjusted fine structures."),

    mc("### test 4 - pearson correlation\nis there a relationship between how old a car is and how many tickets it gets?\n\nh0: no linear relationship\n\nh1: there is a linear relationship"),
    cc("""vehicle_data = df.groupby('Plate_ID').agg(
    ticket_count=('Summons_Number', 'count'),
    avg_age=('Vehicle_Age', 'first')
).dropna()
corr, p_corr = stats.pearsonr(vehicle_data['avg_age'], vehicle_data['ticket_count'])
results['pearson'] = {'correlation': round(float(corr), 4), 'p_value': round(float(p_corr), 6)}
print("correlation:", round(corr, 4), "p-value:", round(p_corr, 6))"""),
    mc("if the correlation is close to zero it means vehicle age has nothing to do with how many tickets someone gets. repeat offending is a behavioral issue not a demographic one."),

    mc("### test 5 - time series decomposition\ndoes ticket volume follow a weekly seasonal pattern?\n\nh0: no seasonal pattern exists\n\nh1: there is a seasonal component"),
    cc("""daily = df.groupby('Issue_Date')['Summons_Number'].count().sort_index()
daily.index = pd.to_datetime(daily.index)
daily = daily.resample('D').sum().fillna(0)
if len(daily) >= 14:
    decomp = seasonal_decompose(daily, model='additive', period=7)
    seasonal_var = float(np.var(decomp.seasonal.dropna()))
else:
    seasonal_var = 0.0
results['seasonal'] = {'seasonal_variance': round(seasonal_var, 4)}
print("seasonal variance:", round(seasonal_var, 4))"""),
    mc("a high seasonal variance confirms that ticketing follows a predictable weekly cycle. enforcement drops on weekends which creates a gap that could be exploited by habitual violators."),

    mc("### test 6 - linear regression (ols)\ncan we predict how much a fine will be based on vehicle characteristics?\n\nh0: vehicle age, state, and repeat status do not predict fine amount\n\nh1: at least one predictor is significant"),
    cc("""reg_df = df.dropna(subset=['Fine_Amount', 'Vehicle_Age', 'Is_OutOfState', 'Is_Repeat_Offender']).copy()
X = reg_df[['Vehicle_Age', 'Is_OutOfState', 'Is_Repeat_Offender']]
X = sm.add_constant(X)
y = reg_df['Fine_Amount']
model = sm.OLS(y, X).fit()
results['regression'] = {'r_squared': round(float(model.rsquared), 4), 'f_pvalue': float(model.f_pvalue)}
print("r-squared:", round(model.rsquared, 4))
print("f p-value:", model.f_pvalue)"""),
    mc("a low r-squared means these variables alone cannot explain the variation in fines. this makes sense because the fine is mostly determined by the violation code itself not by who committed it. the penalty structure is rigid."),

    mc("saving all results"),
    cc("""with open('../docs/statistical_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("all 6 tests completed and saved")"""),
])

nb5 = make_nb([
    mc("# Notebook 05 - Final Load and Tableau Prep\ncomputing kpis and exporting aggregated csvs for the dashboard"),
    cc("""import pandas as pd
import json
import os

df = pd.read_csv('../data/processed/cleaned.csv')
print("loaded", len(df), "rows,", len(df.columns), "columns")"""),

    mc("### computing kpis\nthese are the headline numbers that go at the top of our tableau dashboard"),
    cc("""total_tickets = len(df)
total_revenue = float(df['Fine_Amount'].sum())
avg_fine = float(df['Fine_Amount'].mean().round(2))
oos_pct = float((df['Is_OutOfState'].mean() * 100).round(1))
repeat_pct = float((df['Is_Repeat_Offender'].mean() * 100).round(1))
safety_pct = float((df['Violation_Severity'] == 'Safety-Critical').mean() * 100)
avoidable_pct = float((df['Is_Avoidable'].mean() * 100).round(1))
unique_plates = int(df['Plate_ID'].nunique())
peak_month = str(df.groupby('Month_Name')['Summons_Number'].count().idxmax())
top_borough = str(df.groupby('Violation_County')['Fine_Amount'].sum().idxmax())
commercial_pct = float((df['Vehicle_Type_Group'] == 'Commercial').mean() * 100)
weekend_pct = float((df['Is_Weekend'].mean() * 100).round(1))
avg_car_age = float(df['Vehicle_Age'].mean().round(1))
top_violation = str(df['Violation_Description'].mode()[0])
oos_revenue = float(df[df['Is_OutOfState'] == 1]['Fine_Amount'].sum())

kpis = {
    'total_tickets': total_tickets,
    'total_revenue': total_revenue,
    'avg_fine_per_ticket': avg_fine,
    'outofstate_pct': oos_pct,
    'repeat_offender_pct': repeat_pct,
    'safety_violation_pct': round(safety_pct, 1),
    'avoidable_pct': avoidable_pct,
    'unique_vehicles': unique_plates,
    'peak_month': peak_month,
    'top_borough_revenue': top_borough,
    'commercial_vehicle_pct': round(commercial_pct, 1),
    'weekend_ticket_pct': weekend_pct,
    'avg_vehicle_age': avg_car_age,
    'most_frequent_violation': top_violation,
    'out_of_state_revenue': oos_revenue
}

with open('../docs/kpi_summary.json', 'w') as f:
    json.dump(kpis, f, indent=2)

print("15 kpis computed")
for k, v in kpis.items():
    print(f"  {k}: {v}")"""),

    mc("### exporting 6 csv files for tableau\nwe pre-aggregate in python so tableau doesnt have to process 12000 rows every time"),
    cc("""export_dir = '../tableau/exports'
os.makedirs(export_dir, exist_ok=True)

borough = df.groupby('Violation_County').agg(
    ticket_count=('Summons_Number', 'count'),
    total_revenue=('Fine_Amount', 'sum'),
    avg_fine=('Fine_Amount', 'mean')
).reset_index()
borough['avg_fine'] = borough['avg_fine'].round(2)
borough.to_csv(f'{export_dir}/borough_summary.csv', index=False)
print("1. borough_summary.csv ->", len(borough), "rows")"""),

    cc("""month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
monthly = df.dropna(subset=['Month_Name']).groupby('Month_Name').agg(
    ticket_count=('Summons_Number', 'count'),
    total_revenue=('Fine_Amount', 'sum')
).reindex(month_order).reset_index()
monthly.to_csv(f'{export_dir}/monthly_trend.csv', index=False)
print("2. monthly_trend.csv ->", len(monthly), "rows")"""),

    cc("""streets = df.groupby('Street_Name').agg(
    ticket_count=('Summons_Number', 'count'),
    total_revenue=('Fine_Amount', 'sum')
).reset_index().sort_values('ticket_count', ascending=False)
streets.to_csv(f'{export_dir}/streets_summary.csv', index=False)
print("3. streets_summary.csv ->", len(streets), "rows")"""),

    cc("""vehicle = df.groupby('Vehicle_Make').agg(
    ticket_count=('Summons_Number', 'count')
).reset_index().sort_values('ticket_count', ascending=False)
vehicle.to_csv(f'{export_dir}/vehicle_make_summary.csv', index=False)
print("4. vehicle_make_summary.csv ->", len(vehicle), "rows")"""),

    cc("""offender = df.dropna(subset=['Offender_Tier']).groupby('Offender_Tier').agg(
    ticket_count=('Summons_Number', 'count'),
    total_revenue=('Fine_Amount', 'sum')
).reset_index()
offender.to_csv(f'{export_dir}/offender_tier_summary.csv', index=False)
print("5. offender_tier_summary.csv ->", len(offender), "rows")"""),

    cc("""state_group = df.groupby(['State_Group', 'Violation_County']).agg(
    ticket_count=('Summons_Number', 'count'),
    total_revenue=('Fine_Amount', 'sum')
).reset_index()
state_group.to_csv(f'{export_dir}/state_group_summary.csv', index=False)
print("6. state_group_summary.csv ->", len(state_group), "rows")"""),

    mc("### validation checks\nmaking sure everything looks correct before we move to tableau"),
    cc("""assert len(df) >= 5000, "not enough rows"
assert df['Fine_Amount'].isnull().sum() == 0, "fine amount still has nulls"
assert len(kpis) == 15, "missing kpis"

before_after = {
    "rows_before": 12000,
    "rows_after": len(df),
    "cols_before": 14,
    "cols_after": len(df.columns),
    "engineered_features": len(df.columns) - 14
}
with open('../docs/before_after_summary.json', 'w') as f:
    json.dump(before_after, f, indent=2)

print("all validations passed")
print(f"started with 12000 rows 14 cols -> ended with {len(df)} rows {len(df.columns)} cols")
print(f"engineered {len(df.columns) - 14} new features")
print("ready for tableau")"""),
])

with open("notebooks/04_statistical_analysis.ipynb", "w") as f:
    json.dump(nb4, f, indent=1)
with open("notebooks/05_final_load_prep.ipynb", "w") as f:
    json.dump(nb5, f, indent=1)
print("04 and 05 written")
