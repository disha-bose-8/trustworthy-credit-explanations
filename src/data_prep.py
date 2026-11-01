"""Load the raw Home Credit data, fix known quirks, and build readable features."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/application_train.csv")
CLEAN_PATH = Path("data/processed/clean.csv")

# Model features (about 26). Some are derived below, e.g. AGE_YEARS.
NUMERIC = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME",
    "AGE_YEARS", "YEARS_EMPLOYED", "NO_EMPLOYMENT_RECORD",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]
CATEGORICAL = [
    "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
]
FEATURES = NUMERIC + CATEGORICAL

# Plain-English names, used later in prompts, SHAP charts and the dashboard.
READABLE = {
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Loan annuity (repayment per period)",
    "AMT_GOODS_PRICE": "Price of goods being financed",
    "CREDIT_TO_INCOME": "Loan-to-income ratio",
    "ANNUITY_TO_INCOME": "Repayment-to-income ratio",
    "AGE_YEARS": "Applicant age (years)",
    "YEARS_EMPLOYED": "Years in current job",
    "NO_EMPLOYMENT_RECORD": "No employment record (flag)",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Family size",
    "REGION_RATING_CLIENT": "Rating of applicant's region",
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Social circle: observations 30 days past due",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Social circle: defaults 30 days past due",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit bureau enquiries (last year)",
    "NAME_CONTRACT_TYPE": "Loan type",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_HOUSING_TYPE": "Housing type",
    "OCCUPATION_TYPE": "Occupation",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns property",
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Download it from Kaggle into data/raw/ first.")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Known quirk: 365243 in DAYS_EMPLOYED is a placeholder, not a real value.
    df["NO_EMPLOYMENT_RECORD"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # The dataset stores ages and job length as NEGATIVE DAYS. Convert to years.
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365

    # Ratios an explanation can talk about in plain words.
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    return df.replace([np.inf, -np.inf], np.nan)


def select(df: pd.DataFrame) -> pd.DataFrame:
    return df[["SK_ID_CURR", "TARGET"] + FEATURES]


def main() -> None:
    raw = load_raw()
    df = select(clean(raw))
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Raw data:     {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print(f"Clean data:   {df.shape[0]:,} rows x {df.shape[1]} columns -> {CLEAN_PATH}")
    print(f"Default rate: {df['TARGET'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()