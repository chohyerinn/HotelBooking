from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "hotel_bookings.csv",
    PROJECT_ROOT / "data" / "raw" / "hotel_bookings.csv",
]
LEAKAGE_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
]


def load_raw_data():
    for data_path in DATA_CANDIDATES:
        if data_path.exists():
            return pd.read_csv(data_path)
    raise FileNotFoundError("hotel_bookings.csv was not found in the data directory.")


def clean_booking_data(df):
    cleaned_df = df.copy()
    cleaned_df["children"] = cleaned_df["children"].fillna(0).astype(int)
    cleaned_df["country"] = cleaned_df["country"].fillna("Unknown")
    cleaned_df["agent"] = (
        cleaned_df["agent"].astype("Int64").astype("string").fillna("No agent")
    )
    cleaned_df["company"] = (
        cleaned_df["company"].astype("Int64").astype("string").fillna("No company")
    )

    cleaned_df["total_guests"] = (
        cleaned_df["adults"] + cleaned_df["children"] + cleaned_df["babies"]
    )
    cleaned_df["total_stays"] = (
        cleaned_df["stays_in_weekend_nights"] + cleaned_df["stays_in_week_nights"]
    )
    cleaned_df["is_family"] = (
        (cleaned_df["children"] > 0) | (cleaned_df["babies"] > 0)
    ).astype(int)

    cleaned_df = cleaned_df[
        (cleaned_df["total_guests"] > 0) & (cleaned_df["adr"] >= 0)
    ].copy()
    return cleaned_df.drop(columns=LEAKAGE_COLUMNS)


def load_cleaned_data():
    return clean_booking_data(load_raw_data())


def take_stratified_sample(df, sample_size, random_state=42):
    if sample_size is None or sample_size >= len(df):
        return df.copy()
    sample, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df["is_canceled"],
        random_state=random_state,
    )
    return sample.copy()
