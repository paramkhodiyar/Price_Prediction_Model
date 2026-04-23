# 🏠 ValoraAI — Professional Property Intelligence Platform

> **Hybrid AI Real Estate Platform combining Predictive Machine Learning with Agentic Retrieval-Augmented Generation (RAG)**

---

## 🖥️ Live Demo

Deployed on Render:  
👉 [valoraai-price-predictor on Render](https://price-prediction-model-ju1h.onrender.com)

---

## 🚀 Milestone 2: Agentic AI & RAG Integration (Current)

In Milestone 2, we transformed ValoraAI from a static price prediction tool into an **Interactive Agentic Advisory System**. We implemented an advanced generative AI architecture to provide users with strictly-grounded, highly contextual real estate investment advice based on their property's specific physical and geographical features.

### Key Achievements
- **Multi-Page Architecture:** Upgraded to Streamlit's native `st.navigation` to create a seamless separation between the ML quantitative model and the AI qualitative advisory agent.
- **RAG-Powered Market Context:** Integrated an offline **FAISS Vector Store** (Facebook AI similarity search )to securely retrieve semantic market comparables and historical pricing trends specific to the user's city and property type, bypassing traditional DB lookups.
- **Agentic State Machine Orchestration:** Replaced linear scripting with a deeply nested **LangGraph State Graph**, passing robust evaluation payloads (predictions, contexts, traits) securely between validation, retrieval, logic, and formatting nodes.
- **Strict Domain Grounding:** Architected absolute firewall instructions directly into the base **Groq Llama 3** Large Language Model to permanently restrict the AI from fulfilling non-real-estate prompts (e.g. coding, trivia), thereby ensuring absolute hallucination-free compliance.
- **Deployment & Cloud Memory Optimization:** Designed an ephemeral lazy-loading system using `@st.cache_resource` for the `sentence-transformers` embedding models to ensure Render's 512MB RAM free-tier infrastructure boots smoothly without OOM timeouts.

### Gen AI & Agentic AI Concepts Utilised
- **Retrieval-Augmented Generation (RAG):** Bridging static parametric LLM knowledge with live semantic vector lookups (`all-MiniLM-L6-v2`) locally hosted over FAISS indices.
- **Stateful AI Agents:** Using `langgraph` to construct node-based operational loops that mutate and accumulate contextual memory at every sequential stage of the evaluation protocol.
- **System Prompt Firewalls:** Zero-shot strict prompt grounding that coerces the Llama LLM to cite provided contexts explicitly or execute a safe-fallback protocol for off-topic queries.
- **Semantic Embeddings:** Transforming high-dimensional human queries into mapped geometric space for rapid market comparable similarity matching.

---

## 🧠 Milestone 1: Predictive Machine Learning Engine (Previous)

ValoraAI initially launched as a machine-learning-powered web application predicting residential property prices across major Indian cities using a pre-trained **Random Forest** model under the hood.

### ✨ Milestone 1 Features
- **Instant Valuation** — Get a property price estimate in seconds.
- **Multi-city Support** — Covers Mumbai, Gurgaon, Hyderabad, Kolkata, and more.
- **Smart Fallback** — Gracefully falls back to heuristic pricing if the model can't load.
- **Sample Presets** — One-click load for "Gurgaon" and "South Bombay" test scenarios.
- **Formatted Output** — Results shown in ₹ Lakhs / ₹ Crores with per-sqft rate.
- **Confidence Score** — Displays model prediction confidence (94.2% verified).
- **Vintage Analysis** — Flags whether a property is a "New Build Premium" or "Stable Asset".

### 📊 How It Works (ML Pipeline)
User Input → Feature Engineering → Random Forest Model → Predicted Price (INR)

1. The user enters property details (city, locality, type, area, bedrooms, bathrooms, floor, age, furnishing).  
2. Input is one-hot encoded using `pandas.get_dummies()` and aligned with the exact 100+ feature columns the model was trained on (`mayaai_sale_features.pkl`).  
3. The **Random Forest Regressor** (`mayaai_sale_rf_model.pkl`) predicts the sale price.  
4. If the model or features file is missing, the app falls back to a linear heuristic:
   `Price = (area_sqft × ₹8,500) + (bedrooms × ₹5,00,000)`

### ⚙️ Input Parameters
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

## 🛠️ Tech Stack & Architecture

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit 1.36.0+ |
| ML Model | scikit-learn 1.6.1 (Random Forest Regressor) |
| Agentic Framework | LangGraph, LangChain Core |
| LLM API | Groq API (Llama 3 70B) |
| Vector Database | FAISS (faiss-cpu) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Data Processing | pandas 2.2.2, numpy < 2 |
| Deployment | Render (Python web service) |
| Large File Storage | Git LFS (`.pkl` and `.csv` files) |

## 📁 Project Structure
```text
Price_Prediction_Model/
│
├── app/
│   ├── app.py                  # Main UI & Multi-Page Navigation
│   └── chat_backend.py         # AI Agent LLM Prompt Logic
│
├── agent/
│   └── graph.py                # LangGraph State Machine
│
├── rag/
│   ├── retriever.py            # Local FAISS Query Engine
│   └── setup.py                # Embedding Vector Builder
│
├── data/
│   ├── processed/
│   └── raw/
│
├── deployment/
│   ├── render.yaml             # Render deployment config
│   └── requirements.txt        # Isolated dependencies
│
├── models/                     # Git LFS .pkl files
│   ├── mayaai_sale_features.pkl
│   └── mayaai_sale_rf_model.pkl
│
└── README.md
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11 or 3.12 (Do not use Python 3.13/3.14 to avoid binary compilation errors)
- Git LFS
- Groq API Key

### Installation

```bash
git clone https://github.com/paramkhodiyar/Price_Prediction_Model.git
cd Price_Prediction_Model

# Pull the heavy Random Forest ML Models
git lfs pull

# Create and configure the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the dependencies
pip install --upgrade pip setuptools wheel
pip install -r deployment/requirements.txt

# Initialise environment variables
echo "GROQ_API_KEY=your_api_key_here" > .env

# Launch the platform
streamlit run app/app.py --server.port 8501
```

---

## ☁️ Deployment (Render)

The application utilizes `render.yaml` as infrastructure-as-code for zero-downtime deployment:

```yaml
services:
  - type: web
    name: valoraai-price-predictor
    env: python
    buildCommand: >
      pip install -r deployment/requirements.txt && 
      python scripts/download_models.py
    startCommand: sh -c "streamlit run app/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
```

---

## 👥 Team Details
- Param Khodiyar – 2401020043  
- Anugra Gupta – 2401010085  
- Aditya Rao – 2401010036  
- Adit Singh – 2401010027
