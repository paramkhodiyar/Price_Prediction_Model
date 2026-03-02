# 🏠 ValoraAI — Professional Property Price Prediction

> **Advanced Property Valuation Engine for Indian Real Estate Markets**

ValoraAI is a machine-learning-powered web application that predicts residential property prices across major Indian cities. Built with **Streamlit**, it features a clean, professional UI and uses a pre-trained **Random Forest** model under the hood.

---

## 🖥️ Live Demo

Deployed on Render:  
👉 **[valoraai-price-predictor on Render](https://render.com)** *(update with your actual URL)*

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

```
User Input → Feature Engineering → Random Forest Model → Predicted Price (INR)
```

1. The user enters property details (city, locality, type, area, bedrooms, bathrooms, floor, age, furnishing).
2. Input is one-hot encoded using `pandas.get_dummies()` and aligned with the exact feature columns the model was trained on (`mayaai_sale_features.pkl`).
3. The **Random Forest Regressor** (`mayaai_sale_rf_model.pkl`) predicts the sale price.
4. If the model or features file is missing, the app falls back to a linear heuristic:
   - `Price = (area_sqft × ₹8,500) + (bedrooms × ₹5,00,000)`

---

## 📁 Project Structure

```
Price_Prediction_Model/
│
├── app.py                        # Main Streamlit application
│
├── mayaai_sale_rf_model.pkl      # Trained Random Forest model (Git LFS)
├── mayaai_sale_features.pkl      # Feature column names used during training (Git LFS)
├── mayaai_sale_lr_pipeline.pkl   # Linear Regression pipeline (Git LFS, auxiliary)
│
├── data/                         # Training/reference datasets (Git LFS)
│   ├── Real Estate Data V21.csv
│   ├── data.csv
│   ├── gurgaon_10k.csv
│   ├── hyderabad.csv
│   ├── kolkata.csv
│   └── mumbai.csv
│
├── .streamlit/
│   └── config.toml               # Streamlit theme configuration (amber/light)
│
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version for Render
├── render.yaml                   # Render deployment configuration
└── .gitattributes                # Git LFS tracking for .pkl and .csv files
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit 1.35.0 |
| **ML Model** | scikit-learn 1.6.1 (Random Forest Regressor) |
| **Data Processing** | pandas 2.2.2, numpy 1.26.4 |
| **Model Serialisation** | joblib 1.4.2 |
| **Visualisation** | matplotlib 3.9.0, seaborn 0.13.2 |
| **Deployment** | Render (Python web service) |
| **Large File Storage** | Git LFS (`.pkl` and `.csv` files) |

---

## ⚙️ Input Parameters

| Field | Type | Range | Description |
|---|---|---|---|
| City | Text | — | Target city (e.g., `mumbai`, `gurgaon`) |
| Locality | Text | — | Specific area/neighbourhood |
| Property Type | Select | Apartment, Independent House, Villa, Penthouse | Type of dwelling |
| Area | Number | 200 – 15,000 sqft | Carpet/built-up area |
| Bedrooms | Number | 1 – 10 | Number of bedrooms |
| Bathrooms | Number | 1 – 10 | Number of bathrooms |
| Floor Level | Number | 0 – 80 | Floor the unit is on |
| Total Floors | Number | 1 – 100 | Height of the building |
| Age | Number | 0 – 50 years | Age of property |
| Furnishing | Select | Unfurnished, Semi-furnished, Furnished | Furnishing status |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- [Git LFS](https://git-lfs.github.com/) (required to pull `.pkl` / `.csv` files)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Price_Prediction_Model.git
cd Price_Prediction_Model

# 2. Pull large files via Git LFS
git lfs pull

# 3. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

The app will be available at **http://localhost:8501**

---

## ☁️ Deployment (Render)

The project includes a `render.yaml` for one-click Render deployment:

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

> ⚠️ **Important:** Make sure `.pkl` model files are committed via Git LFS and are accessible on the Render build machine. If using Git LFS, Render must have LFS support enabled or files should be stored as regular binary files in the repo.

---

## 🎨 UI Theming

The Streamlit theme is configured in `.streamlit/config.toml`:

| Token | Value | Usage |
|---|---|---|
| `primaryColor` | `#F59E0B` (Amber) | Buttons, highlights, borders |
| `backgroundColor` | `#F8F9FB` | Page background |
| `secondaryBackgroundColor` | `#FFFFFF` | Cards / panels |
| `textColor` | `#1E293B` | Body text |

Typography uses **Public Sans** from Google Fonts for a clean, professional look.

---

## 📊 Model Details

| Property | Detail |
|---|---|
| Algorithm | Random Forest Regressor |
| Training Data | 50,000+ real estate transactions |
| Cities Covered | Mumbai, Gurgaon, Hyderabad, Kolkata (expandable) |
| Feature Encoding | One-hot encoding via `pd.get_dummies` |
| Serialisation | joblib `.pkl` format |
| Reported Accuracy | 94.2% |
| Fallback | Linear heuristic (`area × ₹8,500 + beds × ₹5L`) |

---

## 📝 License

This project is proprietary to **MayaAI / ValoraAI**. All rights reserved © 2026.

---

*Built with ❤️ using Streamlit and scikit-learn — Powering smarter property decisions across India.*
