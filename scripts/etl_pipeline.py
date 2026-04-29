# etl_pipeline.py

import os
import json
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


RAW_PATH = "data/raw/nyc_parking_tickets_sample.csv"
PROCESSED_PATH = "data/processed/nyc_parking_clean.csv"
TABLEAU_DIR = "data/processed/tableau_exports"
DOCS_DIR = "docs"


os.makedirs("data/processed", exist_ok=True)
os.makedirs(TABLEAU_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_col(col: str) -> str:
    col = col.strip()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def safe_col(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------
# 01 Extraction + Audit
# ---------------------------------------------------------

def extract_and_audit(raw_path: str) -> pd.DataFrame:
    print_section("01 EXTRACTION + RAW AUDIT")

    df = pd.read_csv(raw_path, dtype=str)
    df.columns = [normalize_col(c) for c in df.columns]

    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    print(df.dtypes)

    null_report = pd.DataFrame({
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2)
    })

    duplicate_count = df.duplicated().sum()

    column_profile = {}

    for col in df.columns:
        sample_values = df[col].dropna().astype(str).unique()[:5].tolist()
        unique_count = int(df[col].nunique(dropna=True))

        column_profile[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": float((df[col].isnull().sum() / len(df) * 100).round(2)),
            "unique_count": unique_count,
            "sample_values": sample_values
        }

        print(f"{col}: {unique_count} unique | sample: {sample_values}")

    audit = {
        "source_file": raw_path,
        "audit_timestamp": datetime.now().isoformat(),
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "duplicate_rows": int(duplicate_count),
        "columns": column_profile,
        "null_report": null_report.reset_index()
        .rename(columns={"index": "column"})
        .to_dict(orient="records")
    }

    save_json(audit, os.path.join(DOCS_DIR, "cleaning_audit.json"))

    print("\nSaved raw audit to docs/cleaning_audit.json")
    return df


# ---------------------------------------------------------
# 02 Cleaning Pipeline
# ---------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print_section("02 12-STEP CLEANING PIPELINE")

    transformation_log = {}

    df = df.copy()

    summons_col = safe_col(df, ["Summons_Number", "SummonsNumber", "summons_number"])
    plate_col = safe_col(df, ["Plate_ID", "Plate_Id", "plate_id"])
    state_col = safe_col(df, ["Registration_State", "State", "registration_state"])
    date_col = safe_col(df, ["Issue_Date", "Violation_Date", "issue_date"])
    fine_col = safe_col(df, ["Fine_Amount", "fine_amount"])
    year_col = safe_col(df, ["Vehicle_Year", "vehicle_year"])
    color_col = safe_col(df, ["Vehicle_Color", "vehicle_color"])
    county_col = safe_col(df, ["Violation_County", "Violation_Borough", "violation_county"])
    make_col = safe_col(df, ["Vehicle_Make", "vehicle_make"])
    body_col = safe_col(df, ["Vehicle_Body_Type", "Body_Type", "vehicle_body_type"])
    street_col = safe_col(df, ["Street_Name", "Violation_Street_Name", "street_name"])
    violation_col = safe_col(df, ["Violation", "Violation_Description", "violation"])
    violation_code_col = safe_col(df, ["Violation_Code", "violation_code"])

    # Step 01: Dates
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        df["Issue_Date_Clean"] = df[date_col].dt.strftime("%Y-%m-%d")
        df["Issue_Date_Clean"] = pd.to_datetime(df["Issue_Date_Clean"], errors="coerce")
        df["Year"] = df["Issue_Date_Clean"].dt.year
        df["Month"] = df["Issue_Date_Clean"].dt.month
        df["Month_Name"] = df["Issue_Date_Clean"].dt.month_name()
        df["Day"] = df["Issue_Date_Clean"].dt.day
        df["Weekday"] = df["Issue_Date_Clean"].dt.day_name()

    transformation_log["step_01_dates"] = "Multi-format dates parsed into unified YYYY-MM-DD format; unparseable values set to NaT."

    # Step 02: Fine Amount
    if fine_col:
        df[fine_col] = pd.to_numeric(df[fine_col], errors="coerce")
        df.loc[df[fine_col] <= 0, fine_col] = np.nan

        if violation_code_col:
            df[fine_col] = df.groupby(violation_code_col)[fine_col].transform(
                lambda x: x.fillna(x.median())
            )

        df[fine_col] = df[fine_col].fillna(df[fine_col].median())
        df.rename(columns={fine_col: "Fine_Amount"}, inplace=True)
    else:
        df["Fine_Amount"] = 0

    transformation_log["step_02_fines"] = "Negative and zero fines converted to NaN, then imputed using violation median or global median."

    # Step 03: Vehicle Year
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df.loc[(df[year_col] < 1980) | (df[year_col] > 2014), year_col] = np.nan
        df[year_col] = df[year_col].fillna(df[year_col].median())
        df.rename(columns={year_col: "Vehicle_Year"}, inplace=True)
    else:
        df["Vehicle_Year"] = 2000

    df["Vehicle_Age"] = 2014 - df["Vehicle_Year"]
    transformation_log["step_03_years"] = "Invalid vehicle years such as 9999, >2014, or <1980 replaced with median year."

    # Step 04: Color Mapping
    color_map = {
        "GY": "GRAY", "GREY": "GRAY", "GRAY": "GRAY",
        "BK": "BLACK", "BLK": "BLACK", "BLACK": "BLACK",
        "WH": "WHITE", "WHT": "WHITE", "WHITE": "WHITE",
        "RD": "RED", "RED": "RED",
        "BL": "BLUE", "BLUE": "BLUE",
        "GR": "GREEN", "GREEN": "GREEN",
        "TN": "TAN", "TAN": "TAN",
        "BR": "BROWN", "BROWN": "BROWN",
        "SIL": "SILVER", "SL": "SILVER",
        "GL": "GOLD", "GOLD": "GOLD"
    }

    if color_col:
        df[color_col] = df[color_col].astype(str).str.upper().str.strip()
        df[color_col] = df[color_col].map(color_map).fillna(df[color_col])
        df.rename(columns={color_col: "Vehicle_Color"}, inplace=True)
    else:
        df["Vehicle_Color"] = "UNKNOWN"

    transformation_log["step_04_colors"] = "Vehicle color abbreviations mapped to full names and standardized to uppercase."

    # Step 05: Borough Standardization
    borough_map = {
        "K": "Brooklyn", "BK": "Brooklyn", "BROOKLYN": "Brooklyn",
        "M": "Manhattan", "MN": "Manhattan", "MANHATTAN": "Manhattan",
        "Q": "Queens", "QN": "Queens", "QUEENS": "Queens",
        "BX": "Bronx", "BRONX": "Bronx",
        "R": "Staten Island", "SI": "Staten Island", "STATEN ISLAND": "Staten Island"
    }

    if county_col:
        df[county_col] = df[county_col].astype(str).str.upper().str.strip()
        df[county_col] = df[county_col].map(borough_map).fillna(df[county_col].str.title())
        df.rename(columns={county_col: "Violation_County"}, inplace=True)
    else:
        df["Violation_County"] = "Unknown"

    transformation_log["step_05_boroughs"] = "Borough abbreviations and mixed casing standardized into full borough names."

    # Step 06: Make Consolidation
    make_map = {
        "TOYOT": "TOYOTA",
        "HONDA": "HONDA",
        "FORD": "FORD",
        "NISSA": "NISSAN",
        "CHEVR": "CHEVROLET",
        "CHEVY": "CHEVROLET",
        "BMW": "BMW",
        "ME/BE": "MERCEDES",
        "MERC": "MERCEDES",
        "HYUND": "HYUNDAI"
    }

    if make_col:
        df[make_col] = df[make_col].astype(str).str.upper().str.strip()
        df[make_col] = df[make_col].replace(make_map)

        top_makes = df[make_col].value_counts().head(15).index
        df[make_col] = np.where(df[make_col].isin(top_makes), df[make_col], "OTHER")
        df.rename(columns={make_col: "Vehicle_Make"}, inplace=True)
    else:
        df["Vehicle_Make"] = "UNKNOWN"

    transformation_log["step_06_makes"] = "Vehicle make names corrected; top 15 makes retained and all remaining makes grouped as OTHER."

    # Step 07: Body Type Grouping
    passenger_codes = ["SUBN", "4DSD", "2DSD", "SDN", "CONV"]
    commercial_codes = ["VAN", "TRUCK", "PICK", "DELV", "REFG"]
    taxi_codes = ["TAXI", "LIM"]
    transit_codes = ["BUS", "OMN"]

    def map_body_type(x):
        x = str(x).upper().strip()
        if x in passenger_codes:
            return "Passenger"
        if x in commercial_codes:
            return "Commercial"
        if x in taxi_codes:
            return "Taxi"
        if x in transit_codes:
            return "Transit"
        return "Other"

    if body_col:
        df["Body_Type_Group"] = df[body_col].apply(map_body_type)
        df.rename(columns={body_col: "Vehicle_Body_Type"}, inplace=True)
    else:
        df["Vehicle_Body_Type"] = "UNKNOWN"
        df["Body_Type_Group"] = "Other"

    transformation_log["step_07_body_type"] = "Vehicle body codes grouped into Passenger, Commercial, Taxi, Transit, and Other."

    # Step 08: State Validation
    valid_states = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC"
    }

    if state_col:
        df[state_col] = df[state_col].astype(str).str.upper().str.strip()
        df[state_col] = np.where(df[state_col].isin(valid_states), df[state_col], "UNKNOWN")
        df.rename(columns={state_col: "Registration_State"}, inplace=True)
    else:
        df["Registration_State"] = "UNKNOWN"

    df["Is_OutOfState"] = df["Registration_State"].ne("NY")
    transformation_log["step_08_states"] = "Invalid registration state codes replaced with UNKNOWN; out-of-state flag added."

    # Step 09: Street Type Extraction
    if street_col:
        df[street_col] = df[street_col].astype(str).str.upper().str.strip()
        df["Street_Type"] = df[street_col].str.extract(r"\b(ST|AVE|AVENUE|RD|ROAD|BLVD|DR|PL|LN|WAY|CT)\b", expand=False)
        df["Street_Type"] = df["Street_Type"].fillna("UNKNOWN")
        df.rename(columns={street_col: "Street_Name"}, inplace=True)
    else:
        df["Street_Name"] = "UNKNOWN"
        df["Street_Type"] = "UNKNOWN"

    transformation_log["step_09_streets"] = "Street names standardized to uppercase and street suffix/type extracted."

    # Step 10: Plate Repeat Detection
    if plate_col:
        df[plate_col] = df[plate_col].astype(str).str.upper().str.strip()
        df.rename(columns={plate_col: "Plate_ID"}, inplace=True)
    else:
        df["Plate_ID"] = "UNKNOWN"

    df["Ticket_Count_Per_Plate"] = df.groupby("Plate_ID")["Plate_ID"].transform("count")
    df["Is_Repeat_Offender"] = df["Ticket_Count_Per_Plate"] > 1

    transformation_log["step_10_plates"] = "Ticket count per plate calculated and repeat offender flag added."

    # Step 11: Duplicate Removal
    before_dedup = len(df)

    if summons_col:
        df.rename(columns={summons_col: "Summons_Number"}, inplace=True)
        df = df.drop_duplicates(subset=["Summons_Number"])
    else:
        df["Summons_Number"] = np.arange(1, len(df) + 1)
        df = df.drop_duplicates()

    after_dedup = len(df)

    transformation_log["step_11_dedup"] = f"Duplicate Summons_Number records removed. Rows before: {before_dedup}, after: {after_dedup}."

    # Additional Business Fields
    if violation_col:
        df.rename(columns={violation_col: "Violation_Description"}, inplace=True)
    else:
        df["Violation_Description"] = "Unknown Violation"

    if violation_code_col and violation_code_col in df.columns:
        df.rename(columns={violation_code_col: "Violation_Code"}, inplace=True)
    elif "Violation_Code" not in df.columns:
        df["Violation_Code"] = "UNKNOWN"

    safety_keywords = [
        "FIRE", "HYDRANT", "BUS STOP", "CROSSWALK", "NO STANDING",
        "NO STOPPING", "SIDEWALK", "DOUBLE PARKING"
    ]

    avoidable_keywords = [
        "METER", "EXPIRED", "REGISTRATION", "INSPECTION",
        "PARKING", "STREET CLEANING"
    ]

    df["Violation_Severity"] = np.where(
        df["Violation_Description"].astype(str).str.upper().str.contains("|".join(safety_keywords), na=False),
        "Safety-Critical",
        "General"
    )

    df["Is_Avoidable"] = df["Violation_Description"].astype(str).str.upper().str.contains(
        "|".join(avoidable_keywords),
        na=False
    )

    # Fill remaining nulls
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("UNKNOWN")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].fillna(False)
        else:
            df[col] = df[col].fillna("UNKNOWN")

    # Step 12: Validation Assertions
    assert df["Fine_Amount"].isnull().sum() == 0
    assert (df["Fine_Amount"] >= 0).all()
    assert df["Plate_ID"].isnull().sum() == 0
    assert df["Summons_Number"].isnull().sum() == 0
    assert df.duplicated(subset=["Summons_Number"]).sum() == 0

    transformation_log["step_12_validate"] = "All validation assertions passed successfully."

    save_json(transformation_log, os.path.join(DOCS_DIR, "etl_transformation_log.json"))

    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Saved cleaned dataset to {PROCESSED_PATH}")
    print(f"Final shape: {df.shape}")
    print("Saved transformation log to docs/etl_transformation_log.json")

    return df


# ---------------------------------------------------------
# 05 KPI + Tableau Exports
# ---------------------------------------------------------

def generate_kpis_and_exports(df: pd.DataFrame):
    print_section("05 KPI COMPUTATION + TABLEAU EXPORTS")

    kpis = {
        "total_tickets": int(len(df)),
        "total_revenue": float(df["Fine_Amount"].sum().round(2)),
        "avg_fine_per_ticket": float(df["Fine_Amount"].mean().round(2)),
        "median_fine": float(df["Fine_Amount"].median().round(2)),
        "max_fine": float(df["Fine_Amount"].max().round(2)),
        "outofstate_pct": float((df["Is_OutOfState"].mean() * 100).round(1)),
        "repeat_offender_pct": float((df["Is_Repeat_Offender"].mean() * 100).round(1)),
        "safety_violation_pct": float((df["Violation_Severity"].eq("Safety-Critical").mean() * 100).round(1)),
        "avoidable_pct": float((df["Is_Avoidable"].mean() * 100).round(1)),
        "unique_vehicles": int(df["Plate_ID"].nunique()),
        "unique_states": int(df["Registration_State"].nunique()),
        "unique_boroughs": int(df["Violation_County"].nunique()),
        "peak_month": str(df.groupby("Month_Name")["Summons_Number"].count().idxmax()),
        "top_borough_revenue": str(df.groupby("Violation_County")["Fine_Amount"].sum().idxmax()),
        "top_violation": str(df.groupby("Violation_Description")["Summons_Number"].count().idxmax())
    }

    save_json(kpis, os.path.join(DOCS_DIR, "kpi_summary.json"))

    # Export 1: KPI table
    pd.DataFrame([kpis]).to_csv(
        os.path.join(TABLEAU_DIR, "01_kpi_summary.csv"),
        index=False
    )

    # Export 2: Monthly trend
    monthly = df.groupby(["Year", "Month", "Month_Name"], as_index=False).agg(
        total_tickets=("Summons_Number", "count"),
        total_revenue=("Fine_Amount", "sum"),
        avg_fine=("Fine_Amount", "mean")
    )
    monthly.to_csv(os.path.join(TABLEAU_DIR, "02_monthly_trend.csv"), index=False)

    # Export 3: Borough revenue
    borough = df.groupby("Violation_County", as_index=False).agg(
        total_tickets=("Summons_Number", "count"),
        total_revenue=("Fine_Amount", "sum"),
        avg_fine=("Fine_Amount", "mean"),
        unique_vehicles=("Plate_ID", "nunique")
    )
    borough.to_csv(os.path.join(TABLEAU_DIR, "03_borough_revenue.csv"), index=False)

    # Export 4: Violation summary
    violation = df.groupby(["Violation_Code", "Violation_Description", "Violation_Severity"], as_index=False).agg(
        total_tickets=("Summons_Number", "count"),
        total_revenue=("Fine_Amount", "sum"),
        avg_fine=("Fine_Amount", "mean")
    ).sort_values("total_tickets", ascending=False)
    violation.to_csv(os.path.join(TABLEAU_DIR, "04_violation_summary.csv"), index=False)

    # Export 5: Borough x violation heatmap
    borough_violation = df.groupby(["Violation_County", "Violation_Description"], as_index=False).agg(
        total_tickets=("Summons_Number", "count"),
        total_revenue=("Fine_Amount", "sum")
    )

    borough_totals = borough_violation.groupby("Violation_County")["total_tickets"].transform("sum")
    borough_violation["normalized_ticket_share_pct"] = (
        borough_violation["total_tickets"] / borough_totals * 100
    ).round(2)

    borough_violation.to_csv(
        os.path.join(TABLEAU_DIR, "05_borough_violation_heatmap.csv"),
        index=False
    )

    # Export 6: Repeat offender Pareto
    repeat = df.groupby("Plate_ID", as_index=False).agg(
        ticket_count=("Summons_Number", "count"),
        total_fine=("Fine_Amount", "sum"),
        state=("Registration_State", "first"),
        is_outofstate=("Is_OutOfState", "first")
    ).sort_values("ticket_count", ascending=False)

    repeat["cumulative_tickets"] = repeat["ticket_count"].cumsum()
    repeat["cumulative_ticket_pct"] = (
        repeat["cumulative_tickets"] / repeat["ticket_count"].sum() * 100
    ).round(2)

    repeat.to_csv(os.path.join(TABLEAU_DIR, "06_repeat_offenders.csv"), index=False)

    # Export 7: Day x Month heatmap
    day_month = df.groupby(["Month_Name", "Day"], as_index=False).agg(
        total_tickets=("Summons_Number", "count"),
        total_revenue=("Fine_Amount", "sum")
    )
    day_month.to_csv(os.path.join(TABLEAU_DIR, "07_day_month_heatmap.csv"), index=False)

    # Export 8: Avoidable vs Safety
    avoidable_safety = df.groupby(["Violation_Severity", "Is_Avoidable"], as_index=False).agg(
        total_tickets=("Summons_Number", "count"),
        total_revenue=("Fine_Amount", "sum")
    )
    avoidable_safety.to_csv(os.path.join(TABLEAU_DIR, "08_avoidable_safety.csv"), index=False)

    # Before/After Summary
    summary = {
        "final_rows": int(len(df)),
        "final_columns": int(df.shape[1]),
        "final_null_values": int(df.isnull().sum().sum()),
        "duplicate_summons": int(df.duplicated(subset=["Summons_Number"]).sum()),
        "total_revenue": float(df["Fine_Amount"].sum().round(2)),
        "validation_status": "PASSED"
    }

    save_json(summary, os.path.join(DOCS_DIR, "final_validation_summary.json"))

    print("Saved KPI summary to docs/kpi_summary.json")
    print(f"Saved Tableau exports to {TABLEAU_DIR}")


# ---------------------------------------------------------
# Main Runner
# ---------------------------------------------------------

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw file not found at {RAW_PATH}. "
            "Place nyc_parking_tickets_sample.csv inside data/raw/"
        )

    raw_df = extract_and_audit(RAW_PATH)
    clean_df = clean_data(raw_df)
    generate_kpis_and_exports(clean_df)

    print_section("ETL PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()