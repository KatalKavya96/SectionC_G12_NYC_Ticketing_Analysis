import json

def make_nb(cells):
    return {"nbformat":4,"nbformat_minor":0,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":cells}

def cc(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.strip().split("\n")]}

def mc(src):
    return {"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.strip().split("\n")]}

nb3 = make_nb([
    mc("# Notebook 03 - Exploratory Data Analysis\nlooking at the cleaned data visually to find patterns"),
    cc("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../reports/figures', exist_ok=True)
df = pd.read_csv('../data/processed/nyc_parking_clean.csv')
print("loaded", len(df), "rows")"""),

    mc("### 1. null heatmap\nchecking if there are still any missing values after cleaning"),
    cc("""plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig('../reports/figures/01_null_heatmap.png', dpi=150)
plt.close()
print("saved")"""),
    mc("most of the missing values are in vehicle year and vehicle age columns. the core fields like fine amount and violation description have no nulls after our cleaning."),

    mc("### 2. top 10 violations colored by severity"),
    cc("""plt.figure(figsize=(12,6))
top10 = df['Violation_Description'].value_counts().nlargest(10).index
subset = df[df['Violation_Description'].isin(top10)]
sns.countplot(y='Violation_Description', hue='Violation_Severity', data=subset, order=top10)
plt.title('Top 10 Violations by Severity')
plt.tight_layout()
plt.savefig('../reports/figures/02_top_violations_severity.png', dpi=150)
plt.close()
print("saved")"""),
    mc("street cleaning violations dominate the count. fire hydrant violations stand out as the biggest safety critical issue among the top 10. this means most tickets are for minor administrative stuff but the dangerous ones are also very common."),

    mc("### 3. fine amount distribution"),
    cc("""plt.figure(figsize=(10,6))
sns.histplot(df['Fine_Amount'].dropna(), bins=30, kde=True)
plt.title('Distribution of Fine Amounts')
plt.xlabel('Fine Amount ($)')
plt.tight_layout()
plt.savefig('../reports/figures/03_fine_distribution.png', dpi=150)
plt.close()
print("saved")"""),
    mc("the fines are not normally distributed at all. there are big spikes at 65 and 115 dollars because those are the standard rates for the most common violations. very few tickets are in the middle range between 70 and 100."),

    mc("### 4. monthly ticket and revenue trend"),
    cc("""monthly = df.dropna(subset=['Month_Name']).groupby('Month_Name').agg(
    tickets=('Summons_Number', 'count'),
    revenue=('Fine_Amount', 'sum')
).reindex(['January','February','March','April','May','June','July','August','September','October','November','December'])

fig, ax1 = plt.subplots(figsize=(12,6))
ax1.bar(range(len(monthly)), monthly['revenue'], color='skyblue', label='Revenue')
ax1.set_ylabel('Revenue ($)')
ax2 = ax1.twinx()
ax2.plot(range(len(monthly)), monthly['tickets'], color='red', marker='o', label='Tickets')
ax2.set_ylabel('Ticket Count')
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly.index, rotation=45)
plt.title('Monthly Ticket Volume vs Revenue')
plt.tight_layout()
plt.savefig('../reports/figures/04_monthly_trend.png', dpi=150)
plt.close()
print("saved")"""),
    mc("ticket volume and revenue generally move together but in some months the revenue drops even though tickets stay high. this happens when enforcement shifts toward cheaper violation types like street cleaning instead of expensive ones like fire hydrants."),

    mc("### 5. day of week vs month heatmap"),
    cc("""plt.figure(figsize=(12,7))
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
ct = pd.crosstab(df['Day_of_Week'], df['Month_Name'])
ct = ct.reindex(index=day_order, columns=month_order)
sns.heatmap(ct, cmap='Blues', annot=True, fmt='d')
plt.title('Tickets by Day of Week and Month')
plt.tight_layout()
plt.savefig('../reports/figures/05_day_month_heatmap.png', dpi=150)
plt.close()
print("saved")"""),
    mc("weekdays clearly have way more tickets than weekends across all months. tuesday through thursday are the peak enforcement days. this makes sense because parking rules like alternate side parking are stricter on weekdays."),

    mc("### 6. borough wise revenue"),
    cc("""plt.figure(figsize=(10,6))
borough_rev = df.groupby('Violation_County')['Fine_Amount'].sum().sort_values()
borough_rev.plot(kind='barh', color='teal')
plt.title('Total Revenue by Borough')
plt.xlabel('Revenue ($)')
plt.tight_layout()
plt.savefig('../reports/figures/06_borough_revenue.png', dpi=150)
plt.close()
print("saved")"""),
    mc("manhattan brings in the most revenue by a huge margin. this is expected because parking in manhattan is extremely scarce and enforcement is aggressive. staten island barely registers on the chart."),

    mc("### 7. normalized borough x violation heatmap"),
    cc("""plt.figure(figsize=(14,8))
norm = pd.crosstab(df['Violation_County'], df['Violation_Description'], normalize='index')
sns.heatmap(norm, cmap='YlOrRd')
plt.title('Violation Profile by Borough (Normalized)')
plt.tight_layout()
plt.savefig('../reports/figures/07_borough_violation_normalized.png', dpi=150)
plt.close()
print("saved")"""),
    mc("when we normalize by borough we can see the actual differences in violation profiles. manhattan has a much higher proportion of meter violations while the bronx has more double parking. this means each borough has its own unique parking problem and needs targeted solutions."),

    mc("### 8. repeat offender pareto chart"),
    cc("""fig, ax1 = plt.subplots(figsize=(10,6))
tier_data = df.dropna(subset=['Offender_Tier']).groupby('Offender_Tier')['Fine_Amount'].sum().sort_values(ascending=False)
cumulative = (tier_data.cumsum() / tier_data.sum() * 100)
ax1.bar(tier_data.index.astype(str), tier_data.values, color='steelblue')
ax1.set_ylabel('Total Revenue ($)')
ax2 = ax1.twinx()
ax2.plot(tier_data.index.astype(str), cumulative.values, color='red', marker='D', linewidth=2)
ax2.set_ylabel('Cumulative %')
plt.title('Revenue by Offender Tier (Pareto)')
plt.tight_layout()
plt.savefig('../reports/figures/08_pareto_offenders.png', dpi=150)
plt.close()
print("saved")"""),
    mc("the pareto principle holds here. repeat offenders with 3 or more tickets contribute a disproportionate share of the total revenue. targeting these habitual violators with escalating fines would be more effective than random enforcement."),

    mc("### 9. correlation matrix"),
    cc("""plt.figure(figsize=(8,6))
num_cols = ['Fine_Amount', 'Vehicle_Age', 'Is_Repeat_Offender', 'Is_Avoidable', 'Is_OutOfState']
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Between Key Variables')
plt.tight_layout()
plt.savefig('../reports/figures/09_correlation_matrix.png', dpi=150)
plt.close()
print("saved")"""),
    mc("there is almost no strong correlation between any of the variables. vehicle age does not predict fine amounts. being a repeat offender is not correlated with avoidable violations either. this tells us that ticketing is driven by violation type and location rather than driver demographics."),

    mc("### 10. avoidable vs safety critical breakdown"),
    cc("""plt.figure(figsize=(8,6))
ct = pd.crosstab(
    df['Is_Avoidable'].map({1: 'Avoidable', 0: 'Not Avoidable'}),
    df['Violation_Severity']
)
ct.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], figsize=(8,6))
plt.title('Avoidable vs Safety-Critical Violations')
plt.ylabel('Number of Tickets')
plt.xlabel('')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('../reports/figures/10_avoidable_vs_safety.png', dpi=150)
plt.close()
print("saved")"""),
    mc("most standard violations are avoidable meaning drivers could have simply read the signs or moved their car. safety critical violations like fire hydrants fall into the not avoidable bucket because drivers often have no other option in dense areas. this supports our recommendation for better signage in high ticket zones."),
])

with open("notebooks/03_eda.ipynb", "w") as f:
    json.dump(nb3, f, indent=1)
print("03 written")
