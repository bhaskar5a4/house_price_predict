"""
preprocessing.py — Data cleaning, feature engineering, and scaling pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

LOCATIONS = [
    "Banjara Hills", "Jubilee Hills", "Gachibowli", "Hitech City", "Kondapur",
    "Madhapur", "Kukatpally", "Miyapur", "Secunderabad", "Begumpet",
    "Ameerpet", "Dilsukhnagar", "LB Nagar", "Uppal", "Kompally",
    "Manikonda", "Narsingi", "Kokapet", "Financial District", "Shamshabad"
]

LOCATION_TIERS = {
    "Banjara Hills": "Premium", "Jubilee Hills": "Premium", "Gachibowli": "Premium",
    "Hitech City": "Premium", "Madhapur": "Premium", "Financial District": "Premium",
    "Kondapur": "High", "Kukatpally": "High", "Secunderabad": "High",
    "Begumpet": "High", "Kokapet": "High", "Narsingi": "High",
    "Miyapur": "Mid", "Ameerpet": "Mid", "Manikonda": "Mid",
    "Kompally": "Mid", "Dilsukhnagar": "Mid",
    "LB Nagar": "Standard", "Uppal": "Standard", "Shamshabad": "Standard"
}

LOCATION_BASE_MULTIPLIER = {
    "Premium": 1.60, "High": 1.25, "Mid": 1.00, "Standard": 0.78
}

def generate_synthetic_dataset(n=2000, seed=42):
    """Generate a realistic housing dataset for Hyderabad."""
    np.random.seed(seed)
    records = []
    for _ in range(n):
        loc = np.random.choice(LOCATIONS)
        tier = LOCATION_TIERS[loc]
        mult = LOCATION_BASE_MULTIPLIER[tier]

        bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.30, 0.40, 0.20, 0.05])
        bathrooms = min(bedrooms + np.random.choice([0, 1], p=[0.6, 0.4]), 6)
        area = int(np.random.normal(loc=bedrooms * 550, scale=200))
        area = max(300, min(area, 8000))

        age = np.random.randint(0, 30)
        floor = np.random.randint(0, 25)
        parking = np.random.choice([0, 1, 2], p=[0.10, 0.65, 0.25])
        furnished = np.random.choice(["Unfurnished", "Semi-Furnished", "Furnished"],
                                     p=[0.30, 0.45, 0.25])
        amenities_score = np.random.randint(1, 11)

        base_price = area * 5200 * mult
        base_price += bedrooms * 150000
        base_price += bathrooms * 80000
        base_price -= age * 25000
        base_price += floor * 10000
        base_price += parking * 120000
        if furnished == "Semi-Furnished": base_price += 200000
        elif furnished == "Furnished": base_price += 450000
        base_price += amenities_score * 30000
        noise = np.random.normal(0, base_price * 0.06)
        price = max(500000, base_price + noise)

        records.append({
            "Location": loc, "Area": area, "Bedrooms": bedrooms,
            "Bathrooms": bathrooms, "Age": age, "Floor": floor,
            "Parking": parking, "Furnished": furnished,
            "AmenitiesScore": amenities_score, "Price": int(price)
        })
    return pd.DataFrame(records)


def load_or_generate_data(csv_path="house_data_predict.csv"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_synthetic_dataset()
        df.to_csv(csv_path, index=False)
    return df


def preprocess(df: pd.DataFrame):
    """Clean, engineer features, and encode categoricals."""
    df = df.copy()

    # Drop duplicates and nulls
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Clip outliers
    q_low = df["Price"].quantile(0.01)
    q_high = df["Price"].quantile(0.99)
    df = df[(df["Price"] >= q_low) & (df["Price"] <= q_high)]

    # Feature engineering
    df["PricePerSqFt"] = df["Price"] / df["Area"]
    df["BedBathRatio"] = df["Bedrooms"] / df["Bathrooms"].replace(0, 1)
    df["LocationTier"] = df["Location"].map(LOCATION_TIERS)
    df["TierMultiplier"] = df["LocationTier"].map(LOCATION_BASE_MULTIPLIER)

    # Encode
    le_loc = LabelEncoder()
    le_fur = LabelEncoder()
    le_tier = LabelEncoder()

    df["Location_enc"] = le_loc.fit_transform(df["Location"])
    df["Furnished_enc"] = le_fur.fit_transform(df["Furnished"])
    df["Tier_enc"] = le_tier.fit_transform(df["LocationTier"])

    # Feature columns for modelling
    feature_cols = [
        "Area", "Bedrooms", "Bathrooms", "Age", "Floor",
        "Parking", "AmenitiesScore", "Location_enc",
        "Furnished_enc", "Tier_enc", "TierMultiplier",
        "BedBathRatio"
    ]

    X = df[feature_cols]
    y = df["Price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save encoders and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(le_loc, "models/le_location.pkl")
    joblib.dump(le_fur, "models/le_furnished.pkl")
    joblib.dump(le_tier, "models/le_tier.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    return X_scaled, y.values, df, feature_cols, {
        "le_loc": le_loc, "le_fur": le_fur, "le_tier": le_tier, "scaler": scaler
    }


if __name__ == "__main__":
    df = load_or_generate_data()
    X, y, df_processed, cols, encoders = preprocess(df)
    print(f"Dataset: {len(df_processed)} rows | Features: {len(cols)}")
    print("Preprocessing complete. Artifacts saved to models/")
