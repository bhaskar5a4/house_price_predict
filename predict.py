"""
predict.py — Real-time prediction engine with confidence intervals and insights
"""

import numpy as np
import pandas as pd
import joblib
import os

from preprocessing import LOCATION_TIERS, LOCATION_BASE_MULTIPLIER, LOCATIONS


def _load_artifacts():
    base = "models"
    return {
        "rf":          joblib.load(f"{base}/random_forest.pkl"),
        "lr":          joblib.load(f"{base}/linear_regression.pkl"),
        "scaler":      joblib.load(f"{base}/scaler.pkl"),
        "le_loc":      joblib.load(f"{base}/le_location.pkl"),
        "le_fur":      joblib.load(f"{base}/le_furnished.pkl"),
        "le_tier":     joblib.load(f"{base}/le_tier.pkl"),
        "feat_cols":   joblib.load(f"{base}/feature_cols.pkl"),
        "importances": joblib.load(f"{base}/feature_importances.pkl"),
    }


_ARTIFACTS = None

def get_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = _load_artifacts()
    return _ARTIFACTS


def build_input_vector(area, bedrooms, bathrooms, location, age, floor,
                       parking, furnished, amenities_score):
    arts = get_artifacts()
    tier = LOCATION_TIERS.get(location, "Mid")
    tier_mult = LOCATION_BASE_MULTIPLIER[tier]
    bed_bath = bedrooms / max(bathrooms, 1)

    loc_enc  = arts["le_loc"].transform([location])[0]
    fur_enc  = arts["le_fur"].transform([furnished])[0]
    tier_enc = arts["le_tier"].transform([tier])[0]

    row = {
        "Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
        "Age": age, "Floor": floor, "Parking": parking,
        "AmenitiesScore": amenities_score,
        "Location_enc": loc_enc, "Furnished_enc": fur_enc,
        "Tier_enc": tier_enc, "TierMultiplier": tier_mult,
        "BedBathRatio": bed_bath
    }

    feat_cols = arts["feat_cols"]
    X = np.array([[row[c] for c in feat_cols]])
    X_scaled = arts["scaler"].transform(X)
    return X_scaled


def predict_price(area, bedrooms, bathrooms, location, age=5, floor=2,
                  parking=1, furnished="Semi-Furnished", amenities_score=6):
    """
    Returns a dict with:
      - rf_price: Random Forest prediction
      - lr_price: Linear Regression prediction
      - confidence: Confidence score (0-100)
      - price_range: (low, high) tuple
      - tier: location tier
      - market_status: Underpriced / Fair / Overpriced
      - investment_tag: Good Investment / Moderate / Risky
      - price_per_sqft: price per sq ft
    """
    arts = get_artifacts()
    X_scaled = build_input_vector(area, bedrooms, bathrooms, location, age, floor,
                                  parking, furnished, amenities_score)

    rf_pred = arts["rf"].predict(X_scaled)[0]
    lr_pred = arts["lr"].predict(X_scaled)[0]

    # Confidence via ensemble agreement
    divergence = abs(rf_pred - lr_pred) / rf_pred
    confidence = max(55, min(97, int(100 - divergence * 120)))

    # Price range (±8% for RF ± model uncertainty)
    margin = rf_pred * 0.08
    price_range = (int(rf_pred - margin), int(rf_pred + margin))

    # Market status — compare vs area avg per sqft
    tier = LOCATION_TIERS.get(location, "Mid")
    avg_psf = {
        "Premium": 8200, "High": 6500, "Mid": 5200, "Standard": 4000
    }[tier]
    estimated_psf = rf_pred / area
    ratio = estimated_psf / avg_psf

    if ratio < 0.90:
        market_status = "Underpriced"
        status_color = "green"
    elif ratio > 1.10:
        market_status = "Overpriced"
        status_color = "red"
    else:
        market_status = "Fair"
        status_color = "blue"

    # Investment tag
    if market_status == "Underpriced" and tier in ("Premium", "High"):
        investment_tag = "🟢 Strong Buy"
    elif market_status == "Fair" and tier in ("Premium", "High", "Mid"):
        investment_tag = "🔵 Good Investment"
    elif market_status == "Overpriced":
        investment_tag = "🔴 Risky"
    else:
        investment_tag = "🟡 Moderate"

    return {
        "rf_price": int(rf_pred),
        "lr_price": int(lr_pred),
        "confidence": confidence,
        "price_range": price_range,
        "tier": tier,
        "market_status": market_status,
        "status_color": status_color,
        "investment_tag": investment_tag,
        "price_per_sqft": int(rf_pred / area),
        "avg_psf_area": avg_psf,
    }


def get_market_trend_data():
    """Simulate 24-month price trend for all location tiers."""
    np.random.seed(7)
    months = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="ME")
    trend = {}
    for tier, base in [("Premium", 8200), ("High", 6500), ("Mid", 5200), ("Standard", 4000)]:
        growth = np.cumprod(1 + np.random.normal(0.008, 0.012, 24))
        trend[tier] = (base * growth).astype(int).tolist()
    return months.strftime("%b %Y").tolist(), trend


def get_location_avg_psf():
    """Return average price-per-sqft by location."""
    arts = get_artifacts()
    base = {"Premium": 8200, "High": 6500, "Mid": 5200, "Standard": 4000}
    return {
        loc: base[LOCATION_TIERS[loc]] + np.random.randint(-300, 300)
        for loc in LOCATIONS
    }


if __name__ == "__main__":
    result = predict_price(1500, 3, 2, "Gachibowli", age=3, floor=5,
                           parking=1, furnished="Semi-Furnished", amenities_score=8)
    print(result)
