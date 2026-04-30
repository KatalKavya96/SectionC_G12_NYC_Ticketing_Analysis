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

### DB1 — Enforcement Overview (City-Level Insights)
For directors and executives.

- KPI tiles: Total Tickets, Total Revenue, Avg Fine per Ticket, Out-of-State %, Safety Violations %, Data Completeness  
- Monthly trend (tickets vs revenue)  
- Violation severity donut  
- Borough revenue comparison  
- Tickets by day-of-week (weekday vs weekend)  
- State group distribution  

<img width="1302" height="848" alt="Screenshot 2026-04-29 at 11 09 11 PM" src="https://github.com/user-attachments/assets/2e9e19d5-03b7-4981-b080-6bc484292654" />

### DB2 — Violation & Location Analysis
For operations managers.

- Violation treemap  
- Borough × Violation heatmap  
- Top violations by revenue  
- Top streets by fine amount  
- Vehicle age histogram  
- KPI tiles: Avoidable Violations %, Unique Vehicles, Top Violation, Top Revenue Violation, Weekend Enforcement %, Top Street Revenue  

<img width="1299" height="847" alt="Screenshot 2026-04-29 at 11 09 23 PM" src="https://github.com/user-attachments/assets/c9773a60-0ab6-419a-94ba-1afa4c95ebad" />

### DB3 — Vehicle & Offender Insights
For planners and analysts.

- KPI tiles: Revenue per Vehicle, Commercial Vehicle %, Repeat Offender %, Peak Month  
- Top vehicle makes (bubble chart: Avg Fine vs Ticket Count vs Revenue)  
- Vehicle count by make (bar chart)  
- State group × Borough revenue  
- Vehicle type distribution (donut)  
- Offender tier revenue (Pareto analysis)

<img width="1301" height="848" alt="Screenshot 2026-04-29 at 11 09 37 PM" src="https://github.com/user-attachments/assets/fa37a6c6-616a-4ec7-83fd-f262a4fe8692" />

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
Abuzar Haideri
Ayush Tiwari
Vaibhav Rajput
Deepak Pandey
Abhay Mallik
