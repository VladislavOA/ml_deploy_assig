import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates domain-specific features for customer churn modeling:
    - tenure_years: tenure converted to years
    - tenure_bucket: binned tenure categories (e.g. '0-12m', '13-24m', etc.)
    - tenure_x_contract_type: tenure weighted by contract length in months (interaction)
    - tenure_x_monthly_charges: tenure multiplied by monthly charges
    - months_since_last_contract_change: tenure modulo contract cycle length
    """
    df = df.copy()

    # 1. tenure_years
    df["tenure_years"] = (df["tenure"] / 12.0).round(2)

    # 2. tenure_bucket
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-60m", "61-72m"],
    ).astype(str)

    # Contract duration mapping in months
    contract_duration_map = {
        "Month-to-month": 1,
        "One year": 12,
        "Two year": 24,
    }
    contract_months = df["Contract"].map(contract_duration_map).fillna(1).astype(int)

    # 3. tenure × contract_type
    df["tenure_x_contract_type"] = df["tenure"] * contract_months

    # 4. tenure × monthly_charges
    df["tenure_x_monthly_charges"] = df["tenure"] * df["MonthlyCharges"]

    # 5. months_since_last_contract_change
    df["months_since_last_contract_change"] = df["tenure"] % contract_months

    return df
