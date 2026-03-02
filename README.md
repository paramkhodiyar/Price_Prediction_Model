# 🏠 ValoraAI — Professional Property Price Prediction

> **Advanced Property Valuation Engine for Indian Real Estate Markets**

ValoraAI is a machine-learning-powered web application that predicts residential property prices across major Indian cities. Built with **Streamlit**, it features a clean, professional UI and uses a pre-trained **Random Forest** model under the hood.

---

## 🖥️ Live Demo

Deployed on Render:  
👉 [valoraai-price-predictor on Render](https://price-prediction-model-ju1h.onrender.com)

---

## ✨ Features

- **Instant Valuation** — Get a property price estimate in seconds  
- **Multi-city Support** — Covers Mumbai, Gurgaon, Hyderabad, Kolkata, and more  
- **Smart Fallback** — Gracefully falls back to heuristic pricing if the model can't load  
- **Sample Presets** — One-click load for "Gurgaon" and "South Bombay" test scenarios  
- **Formatted Output** — Results shown in ₹ Lakhs / ₹ Crores with per-sqft rate  
- **Confidence Score** — Displays model prediction confidence (94.2% verified)  
- **Vintage Analysis** — Flags whether a property is a "New Build Premium" or "Stable Asset"  
- **Responsive UI** — Minimal, modern design using Public Sans font with amber accent palette  

---

## 🧠 How It Works

User Input → Feature Engineering → Random Forest Model → Predicted Price (INR)

1. The user enters property details (city, locality, type, area, bedrooms, bathrooms, floor, age, furnishing).  
2. Input is one-hot encoded using `pandas.get_dummies()` and aligned with the exact feature columns the model was trained on (`mayaai_sale_features.pkl`).  
3. The **Random Forest Regressor** (`mayaai_sale_rf_model.pkl`) predicts the sale price.  
4. If the model or features file is missing, the app falls back to a linear heuristic:

Price = (area_sqft × ₹8,500) + (bedrooms × ₹5,00,000)

---

## 📁 Project Structure

```
Price_Prediction_Model/
│
├── app/
│   └── app.py
│
├── data/
│   ├── processed/
│   │   └── data.csv
│   │
│   └── raw/
│       ├── gurgaon_10k.csv
│       ├── hyderabad.csv
│       ├── kolkata.csv
│       ├── mumbai.csv
│       └── Real Estate Data V21.csv
│
├── deployment/
│   ├── render.yaml
│   ├── requirements.txt
│   └── runtime.txt
│
├── models/
│   ├── mayaai_sale_features.pkl
│   ├── mayaai_sale_lr_pipeline.pkl
│   └── mayaai_sale_rf_model.pkl
│
├── reports/
│   └── updated latedx.pdf
│
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit 1.35.0 |
| ML Model | scikit-learn 1.6.1 (Random Forest Regressor) |
| Data Processing | pandas 2.2.2, numpy 1.26.4 |
| Model Serialisation | joblib 1.4.2 |
| Visualisation | matplotlib 3.9.0, seaborn 0.13.2 |
| Deployment | Render (Python web service) |
| Large File Storage | Git LFS (`.pkl` and `.csv` files) |

---

## ⚙️ Input Parameters

| Field | Type | Range | Description |
|---|---|---|---|
| City | Text | — | Target city (e.g., `mumbai`, `gurgaon`) |
| Locality | Text | — | Specific area/neighbourhood |
| Property Type | Select | Apartment, Independent House, Villa, Penthouse |
| Area | Number | 200 – 15,000 sqft |
| Bedrooms | Number | 1 – 10 |
| Bathrooms | Number | 1 – 10 |
| Floor Level | Number | 0 – 80 |
| Total Floors | Number | 1 – 100 |
| Age | Number | 0 – 50 years |
| Furnishing | Select | Unfurnished, Semi-furnished, Furnished |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- Git LFS

### Installation

```bash
git clone https://github.com/<your-username>/Price_Prediction_Model.git
cd Price_Prediction_Model

git lfs pull

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

---

## ☁️ Deployment (Render)

```yaml
services:
  - type: web
    name: valoraai-price-predictor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
```

---

## 🎨 UI Theming

Configured in `.streamlit/config.toml`

| Token | Value |
|---|---|
| primaryColor | #F59E0B |
| backgroundColor | #F8F9FB |
| secondaryBackgroundColor | #FFFFFF |
| textColor | #1E293B |

Typography uses **Public Sans**.

---

## 📊 Model Details

| Property | Detail |
|---|---|
| Algorithm | Random Forest Regressor |
| Training Data | 50,000+ transactions |
| Cities Covered | Mumbai, Gurgaon, Hyderabad, Kolkata |
| Feature Encoding | One-hot encoding |
| Serialisation | joblib `.pkl` |
| Reported Accuracy | 94.2% |
| Fallback | Linear heuristic |

---

## 👥 Team Details

- Param Khodiyar – 2401020043  
- Anugra Gupta – 2401010085  
- Aditya Rao – 2401010036  
- Adit Singh – 2401010027
