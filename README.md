# 🏙️ PropIQ — Real-Time House Price Intelligence Platform

> A production-grade AI-powered real estate analytics dashboard built with Python, Streamlit, and Machine Learning.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models (first run only — auto-runs on app start)

```bash
python train_model.py
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
house_price_platform/
│
├── app.py                  # Main Streamlit dashboard (4 pages)
├── preprocessing.py        # Data pipeline: load, clean, encode, scale
├── train_model.py          # Train Linear Regression + Random Forest
├── predict.py              # Real-time prediction engine + insights
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── house_data_predict.csv  # Auto-generated on first run (2000 records)
│
└── models/                 # Auto-created after training
    ├── random_forest.pkl
    ├── linear_regression.pkl
    ├── scaler.pkl
    ├── le_location.pkl
    ├── le_furnished.pkl
    ├── le_tier.pkl
    ├── feature_cols.pkl
    ├── feature_importances.pkl
    └── eval_data.pkl
```

---

## 🧠 ML Architecture

| Model | Role | Typical R² |
|---|---|---|
| Linear Regression | Baseline / explainability | ~0.82 |
| Random Forest (200 trees) | Main prediction engine | ~0.95+ |

### Features Used

| Feature | Type |
|---|---|
| Area (sq ft) | Numeric |
| Bedrooms | Ordinal |
| Bathrooms | Ordinal |
| Building Age | Numeric |
| Floor Level | Numeric |
| Parking Slots | Ordinal |
| Amenities Score (1–10) | Numeric |
| Location (encoded) | Categorical |
| Furnishing (encoded) | Categorical |
| Location Tier (encoded) | Categorical |
| Tier Multiplier | Derived |
| Bed/Bath Ratio | Engineered |

---

## 🎯 Dashboard Pages

### 🏠 Prediction Dashboard
- Large price display card
- Confidence gauge (0–100%)
- Market status badge (Underpriced / Fair / Overpriced)
- Investment signal (Strong Buy → Risky)
- Model comparison (RF vs Linear Regression)
- Price range spread visualisation

### 📊 Analytics Dashboard
- Area vs Price scatter with tier colour-coding + trendline
- Bedrooms vs Average Price bar chart
- Price distribution histogram
- Avg Price heatmap (Tier × Bedrooms)
- Price per sq ft by all 20 locations

### 📈 Market Trends
- 24-month simulated price trend (all 4 tiers)
- 24-month growth summary cards
- 12-month forward price forecast with confidence band

### 📌 Property Insights
- Random Forest feature importance chart
- Smart natural-language insights for your property
- Side-by-side property vs area average comparison table

### 🗂️ Prediction History
- Session-scoped log of all predictions
- Multi-property comparison chart
- Clear history button

---

## ⚙️ Configuration

Dataset is auto-generated for Hyderabad with 20 real localities:

**Premium Tier:** Banjara Hills, Jubilee Hills, Gachibowli, Hitech City, Madhapur, Financial District  
**High Tier:** Kondapur, Kukatpally, Secunderabad, Begumpet, Kokapet, Narsingi  
**Mid Tier:** Miyapur, Ameerpet, Manikonda, Kompally, Dilsukhnagar  
**Standard Tier:** LB Nagar, Uppal, Shamshabad

To use your own CSV, place `house_data_predict.csv` in the project folder with columns:  
`Location, Area, Bedrooms, Bathrooms, Age, Floor, Parking, Furnished, AmenitiesScore, Price`

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11+ |
| RAM | 4 GB | 8 GB |
| Storage | 200 MB | 500 MB |

---

## 📦 Tech Stack

- **Python 3.x** — Core language
- **Streamlit** — Web dashboard framework
- **scikit-learn** — ML models (LinearRegression, RandomForestRegressor)
- **Plotly** — Interactive charts & gauges
- **Pandas / NumPy** — Data processing
- **joblib** — Model serialisation

---

*Built for B.Tech CSE Final Year Project — Hyderabad Real Estate Intelligence*
