# NYC_Parking_Ticket_Violations_Analysis

## Executive Overview
DVA is a data analytics capstone focused on city parking violation intelligence. It transforms raw summons data into executive dashboards, operational insights, offender intelligence, and policy recommendations.

## Objectives
- Quantify ticket volume, revenue, repeat offenders, and safety violations
- Optimize patrol deployment by borough/day/season
- Identify repeat offender concentration
- Improve collections from out-of-state plates
- Support infrastructure decisions with street-level evidence

## Dashboard Suite

### DB1 — The City Picture
For directors and executives.

- KPI tiles: Total Tickets, Revenue, Avg Fine, Repeat Offender %, Safety %, Out-of-State %
- Monthly trend (tickets vs revenue)
- Severity donut
- Borough revenue ranking
- Borough normalized heatmap

### DB2 — When and What
For operations managers.

- Violation treemap
- Day-of-week volume analysis
- Borough × Violation heatmap
- Avoidable vs Safety stacked analysis

### DB3 — Who Is Being Ticketed
For planners and analysts.

- Top streets stacked bar
- Vehicle age histogram
- Vehicle make bubble chart
- Body type donut
- State vs Borough table

### DB4 — The Repeat Problem
For policy and compliance teams.

- Repeat offender KPI tiles
- Pareto revenue chart
- Repeat offender by borough

## Key Statistical Methods
- Independent T-Test
- Chi-Square Test
- One-Way ANOVA + Tukey HSD
- Pearson Correlation
- Time Series Decomposition
- Linear Regression

## Top Recommendations

### 1. Violation-Calibrated Patrol Scheduling
Deploy officers using borough × day patterns for 15–20% higher yield per officer-hour.

### 2. Repeat Offender Escalation Program
Warnings, compliance appointments, and booting thresholds for chronic violators.

### 3. Dynamic Digital Parking Signs
Install on top avoidable-violation streets to reduce unnecessary citations.

### 4. Expand Loading Zones
Reduce commercial double-parking and safety-critical violations.

### 5. Interstate DMV Recovery Program
Improve collection rates from NJ/CT violators.

## Tech Stack
- Tableau
- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Jupyter Notebook
- Excel / CSV

## Business Impact
This project demonstrates how analytics can convert enforcement data into measurable policy, staffing, and revenue outcomes.

## Team
Kavya Katal
Ayush Tiwari
Vaibhav Rajput
Abuzar Haideri
Deepak Pandey
Abhay Mallik
