# ============================================================
#  config.py — Central configuration for the dashboard
# ============================================================

import os

# ── Paths ──────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")

UNSW_CSV        = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
IOT_CSV         = os.path.join(DATA_DIR, "train_test_network.csv")

# ── Model settings ─────────────────────────────────────────
MAX_ROWS        = 50_000
RANDOM_STATE    = 42
TEST_SIZE       = 0.30
N_ESTIMATORS    = 150

# ── LLM settings ───────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
SUMMARIZER_MODEL = "facebook/bart-large-cnn"

# ── Dashboard settings ─────────────────────────────────────
APP_TITLE       = "IoT Threat Intelligence Dashboard"
APP_HOST        = "127.0.0.1"
APP_PORT        = 8050
DEBUG           = True

# ── Colour palette ─────────────────────────────────────────
PALETTE = {
    "bg_dark":     "#0d1117",
    "bg_card":     "#161b22",
    "border":      "#30363d",
    "accent":      "#58a6ff",
    "accent2":     "#3fb950",
    "danger":      "#f85149",
    "warning":     "#d29922",
    "text":        "#e6edf3",
    "text_muted":  "#8b949e",
}

PLOTLY_TEMPLATE = "plotly_dark"
