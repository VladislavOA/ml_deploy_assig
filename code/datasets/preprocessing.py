import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads raw data, fixes broken columns, imputes missing values,
    encodes target (Churn) to 0/1, and splits into stratified train/test sets.
    Feature engineering is deferred to Stage 2 (Model Engineering).
    """
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")

    # 1. Drop identifier column if present
    if "customerID" in df.columns:
        print("Dropping non-predictive identifier: customerID")
        df = df.drop(columns=["customerID"])

    # 2. Fix broken columns (TotalCharges contains whitespace ' ')
    if "TotalCharges" in df.columns:
        print("Converting TotalCharges to numeric (coercing invalid values to NaN)...")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. Impute missing values
    # For TotalCharges: NaN values occur where tenure == 0 (new customers) -> impute 0.0
    if "TotalCharges" in df.columns and df["TotalCharges"].isna().any():
        nan_count = df["TotalCharges"].isna().sum()
        print(f"Imputing {nan_count} missing values in TotalCharges with 0.0...")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # General imputation for any other numerical columns (median) and categorical columns (mode)
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            print(f"Imputing missing values in numeric column '{col}' with median ({median_val})")
            df[col] = df[col].fillna(median_val)

    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        if col != "Churn" and df[col].isna().any():
            mode_val = df[col].mode()[0]
            print(f"Imputing missing values in categorical column '{col}' with mode ('{mode_val}')")
            df[col] = df[col].fillna(mode_val)

    # 4. Encode target as 0 or 1
    if "Churn" in df.columns:
        print("Encoding target 'Churn' to binary (0/1)...")
        if df["Churn"].dtype == object or str(df["Churn"].dtype) in ["string", "str"]:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
        df["Churn"] = df["Churn"].astype(int)
        print("Target distribution:")
        print(df["Churn"].value_counts(normalize=True))
    else:
        raise ValueError("Target column 'Churn' not found in dataset!")


    # 5. Stratified train/test split grouped by target
    print(f"Splitting data (test_size={test_size}, random_state={random_state}, stratify=Churn)...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Churn"]
    )

    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    # 6. Save processed datasets
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved training data to: {train_path}")
    print(f"Saved testing data to: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Stage")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/data_raw.csv",
        help="Path to raw data CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save train.csv and test.csv"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state seed for reproducibility"
    )
    args = parser.parse_args()

    preprocess_data(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )