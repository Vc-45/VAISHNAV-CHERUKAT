# 🛡️ IoT Threat Intelligence Dashboard

**MSc Research Platform — AI-Powered IoT Threat Intelligence Automation Using LLMs**

A professional, interactive Dash/Plotly dashboard for the UNSW-NB15 and IoT Network datasets.

---

## 📁 Project Structure

```
iot_threat_dashboard/
├── app.py                  # Main dashboard application
├── config.py               # Central configuration
├── requirements.txt        # Python dependencies
├── README.md
├── data/                   ← Place your CSV files here
│   ├── UNSW_NB15_training-set.csv
│   └── train_test_network.csv
├── models/                 # Saved model files (auto-created)
├── outputs/                # Exported artefacts (auto-created)
└── utils/
    ├── data_loader.py      # Dataset loading & preprocessing
    ├── ml_engine.py        # Model training & evaluation
    ├── ioc_engine.py       # IoC extraction & CTI generation
    └── charts.py           # Plotly figure factories
```

---

## 🚀 Quick Start (VS Code)

### 1. Prerequisites
- Python 3.10 or 3.11 (recommended)
- VS Code with the Python extension installed

### 2. Create & activate a virtual environment
```bash
# In VS Code terminal (Ctrl + `)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your datasets
Copy your CSV files into the `data/` folder:
```
data/UNSW_NB15_training-set.csv
data/train_test_network.csv
```

> If you only have one dataset, the dashboard will train on whichever is present and skip the other.

### 5. Run the dashboard
```bash
python app.py
```

Then open **http://127.0.0.1:8050** in your browser.

### 6. Train the models
- In the dashboard, click **⚡ Train All Models** in the left sidebar.
- Training runs in the background — watch the status indicator.
- Once complete, all pages populate automatically.

---

## 🗂️ Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, performance summary, MITRE & severity charts |
| 🧠 UNSW-NB15 Model | Gauge metrics, confusion matrix, feature importance |
| 📡 IoT Model | Same as above for the IoT network dataset |
| 🔍 CTI Records | Enriched threat records with MITRE ATT&CK mapping, filterable table |
| 🧩 IoC Extractor | Paste any threat report and extract IPs, CVEs, hashes, domains |
| 📊 Data Explorer | Interactive histogram for any numeric feature |
| 📋 Logs | Live training log output |

---

## ⚙️ Configuration

Edit `config.py` to change:
- `MAX_ROWS` — rows to load from each dataset (default 50,000)
- `N_ESTIMATORS` — trees in the RandomForest (default 150)
- `APP_PORT` — dashboard port (default 8050)
- `OPENAI_API_KEY` — set as environment variable for optional LLM enrichment

### Optional: LLM enrichment
```bash
# Windows
set OPENAI_API_KEY=sk-...

# macOS / Linux
export OPENAI_API_KEY=sk-...
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| dash + dash-bootstrap-components | Web dashboard framework |
| plotly | Interactive charts |
| scikit-learn | ML pipeline |
| pandas / numpy | Data processing |
| transformers | NLP summarisation |
| openai | Optional LLM enrichment |
| joblib | Model persistence |

---

## 🎓 MSc Context

This project demonstrates:
1. **Binary classification** on UNSW-NB15 (normal vs attack)
2. **Multiclass classification** on IoT network traffic
3. **IoC extraction** via regex from threat reports
4. **Enriched CTI record generation** with MITRE ATT&CK mapping
5. **Optional LLM enrichment** via OpenAI API
