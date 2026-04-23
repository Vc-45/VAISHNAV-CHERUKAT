# ============================================================
# app.py — IoT Threat Intelligence Dashboard (VS Code ready)
# ============================================================

import os
import json
import warnings
import threading
import traceback

import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc

from config import (
    UNSW_CSV, IOT_CSV, MAX_ROWS, RANDOM_STATE,
    TEST_SIZE, N_ESTIMATORS, APP_TITLE, APP_HOST, APP_PORT, DEBUG
)
from utils.data_loader import load_unsw, prepare_unsw, load_iot, prepare_iot
from utils.ml_engine import (
    train_pipeline,
    get_confusion_matrix,
    get_feature_importances,
    save_model,
)
from utils.ioc_engine import extract_iocs, build_cti_record, generate_cti_records, cti_to_dataframe
from utils.charts import (
    metric_gauge, confusion_heatmap, feature_importance_bar,
    class_distribution, feature_histogram, severity_pie,
    mitre_tactic_bar, confidence_histogram, metrics_bar,
    attack_type_donut, ioc_table,
)

warnings.filterwarnings("ignore")

# ── Sample threat report ────────────────────────────────────
SAMPLE_REPORT = """
A suspicious campaign targeting IoT cameras was observed.
Infected devices contacted 185.220.101.45 and https://malicious-update.example.net/payload.
Researchers linked the activity to CVE-2021-35395.
The malware also attempted to communicate with botnet-control.example.org.
A suspicious SHA256 hash was identified:
44d88612fea8a8f36de82e1278abb02f4f7f7e4e8f1234567890abcdef123456.
Contact the SOC team at soc@example-corp.com for further escalation.
Security teams should isolate the devices, block the domains, and patch affected firmware.
""".strip()

# ── Colour tokens ────────────────────────────────────────────
CLR = {
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "yellow": "#d29922",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "active": "#21262d",
}

# ── Shared state (populated on train) ──────────────────────
STATE = {
    "unsw_metrics": None,
    "unsw_cm": None,
    "unsw_labels": None,
    "unsw_imp": None,
    "unsw_y_test": None,
    "unsw_y_pred": None,
    "unsw_df": None,
    "iot_metrics": None,
    "iot_cm": None,
    "iot_labels": None,
    "iot_imp": None,
    "iot_df": None,
    "cti_df": None,
    "iocs": None,
    "training_log": [],
    "training_done": False,
    "training_error": None,
}

# ──────────────────────────────────────────────────────────────
# LAYOUT HELPERS
# ──────────────────────────────────────────────────────────────

def card(children, className="", style=None):
    base = {
        "background": CLR["card"],
        "border": f"1px solid {CLR['border']}",
        "borderRadius": "10px",
        "padding": "18px",
        "marginBottom": "16px",
    }
    if style:
        base.update(style)
    return html.Div(children, className=className, style=base)


def badge(text, color=None):
    color = color or CLR["accent"]
    return html.Span(
        text,
        style={
            "background": color + "22",
            "color": color,
            "border": f"1px solid {color}55",
            "borderRadius": "20px",
            "padding": "2px 10px",
            "fontSize": "11px",
            "fontWeight": "600",
            "marginRight": "6px",
        },
    )


def section_title(text):
    return html.H6(
        text,
        style={
            "color": CLR["muted"],
            "textTransform": "uppercase",
            "letterSpacing": "1px",
            "fontSize": "11px",
            "marginBottom": "12px",
        },
    )


def kpi_card(label, value, color=None):
    color = color or CLR["accent"]
    return card(
        html.Div(
            [
                html.Div(
                    value,
                    style={"fontSize": "28px", "fontWeight": "700", "color": color},
                ),
                html.Div(
                    label,
                    style={
                        "fontSize": "12px",
                        "color": CLR["muted"],
                        "marginTop": "4px",
                    },
                ),
            ]
        )
    )


def empty_fig(msg="Run training to see results"):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=msg,
                showarrow=False,
                font=dict(color="#8b949e", size=13),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
        ],
    )
    return fig


def nav_button_style(active=False):
    style = {
        "display": "block",
        "width": "100%",
        "background": "transparent",
        "border": "none",
        "color": CLR["text"],
        "padding": "10px 16px",
        "textAlign": "left",
        "cursor": "pointer",
        "fontSize": "13px",
        "borderRadius": "6px",
        "marginBottom": "2px",
        "transition": "background 0.15s",
    }
    if active:
        style.update(
            {
                "background": CLR["active"],
                "color": CLR["accent"],
                "fontWeight": "600",
            }
        )
    return style


# ──────────────────────────────────────────────────────────────
# APP INIT
# ──────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title=APP_TITLE,
)
server = app.server

# ──────────────────────────────────────────────────────────────
# SIDEBAR NAV
# ──────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("🏠", "Overview", "tab-overview"),
    ("🧠", "UNSW-NB15 Model", "tab-unsw"),
    ("📡", "IoT Model", "tab-iot"),
    ("🔍", "CTI Records", "tab-cti"),
    ("🧩", "IoC Extractor", "tab-ioc"),
    ("📊", "Data Explorer", "tab-data"),
    ("📋", "Logs", "tab-logs"),
]

sidebar = html.Div(
    [
        html.Div(
            [
                html.Div("🛡️", style={"fontSize": "28px"}),
                html.Div(
                    [
                        html.Div(
                            "IoT ThreatIQ",
                            style={
                                "fontWeight": "700",
                                "fontSize": "15px",
                                "color": CLR["text"],
                                "lineHeight": "1.2",
                            },
                        ),
                        html.Div(
                            "MSc Research Platform",
                            style={"fontSize": "10px", "color": CLR["muted"]},
                        ),
                    ]
                ),
            ],
            style={
                "display": "flex",
                "gap": "10px",
                "alignItems": "center",
                "padding": "20px 16px 16px",
                "borderBottom": f"1px solid {CLR['border']}",
            },
        ),
        html.Div(
            [
                html.Button(
                    [html.Span(icon, style={"marginRight": "8px"}), html.Span(label)],
                    id=tab_id,
                    n_clicks=0,
                    type="button",
                    className="nav-btn",
                    style=nav_button_style(tab_id == "tab-overview"),
                )
                for icon, label, tab_id in NAV_ITEMS
            ],
            style={"padding": "12px 8px"},
        ),
        html.Div(
            [
                html.Button(
                    "⚡  Train All Models",
                    id="btn-train",
                    n_clicks=0,
                    type="button",
                    style={
                        "width": "100%",
                        "background": CLR["accent"],
                        "color": "#fff",
                        "border": "none",
                        "borderRadius": "8px",
                        "padding": "11px",
                        "fontWeight": "600",
                        "fontSize": "13px",
                        "cursor": "pointer",
                    },
                ),
                html.Div(
                    id="train-status",
                    style={
                        "fontSize": "11px",
                        "color": CLR["muted"],
                        "marginTop": "8px",
                        "textAlign": "center",
                    },
                ),
            ],
            style={
                "padding": "12px 16px",
                "borderTop": f"1px solid {CLR['border']}",
                "marginTop": "auto",
            },
        ),
    ],
    style={
        "width": "220px",
        "minHeight": "100vh",
        "background": CLR["card"],
        "borderRight": f"1px solid {CLR['border']}",
        "display": "flex",
        "flexDirection": "column",
        "flexShrink": "0",
    },
)

# ──────────────────────────────────────────────────────────────
# PAGE CONTENT AREA
# ──────────────────────────────────────────────────────────────

content = html.Div(
    id="page-content",
    style={"flex": "1", "padding": "24px", "overflowY": "auto"},
)

# ──────────────────────────────────────────────────────────────
# ROOT LAYOUT
# ──────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        dcc.Store(id="active-tab", data="tab-overview"),
        dcc.Store(id="train-trigger", data=0),
        dcc.Interval(id="poll-interval", interval=1500, n_intervals=0, disabled=True),
        html.Div(
            [sidebar, content],
            style={
                "display": "flex",
                "minHeight": "100vh",
                "fontFamily": "Inter, Segoe UI, sans-serif",
                "background": CLR["bg"],
                "color": CLR["text"],
            },
        ),
    ],
    style={"background": CLR["bg"], "margin": "0"},
)

# ──────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ──────────────────────────────────────────────────────────────

def page_overview():
    unsw_ok = STATE["unsw_metrics"] is not None
    iot_ok = STATE["iot_metrics"] is not None
    cti_ok = STATE["cti_df"] is not None

    unsw_m = STATE["unsw_metrics"] or {}
    iot_m = STATE["iot_metrics"] or {}
    cti_df = STATE["cti_df"]

    n_threats = (
        int(cti_df["attack_detected"].sum())
        if cti_ok and "attack_detected" in cti_df.columns
        else "—"
    )
    n_total = len(cti_df) if cti_ok else "—"
    n_high = (
        int((cti_df["severity"] == "high").sum())
        if cti_ok and "severity" in cti_df.columns
        else "—"
    )

    return html.Div(
        [
            html.H4("Dashboard Overview", style={"marginBottom": "4px"}),
            html.P(
                "AI-Powered IoT Threat Intelligence — MSc Research Platform",
                style={
                    "color": CLR["muted"],
                    "marginBottom": "20px",
                    "fontSize": "13px",
                },
            ),
            card(
                html.Div(
                    [
                        badge(
                            "UNSW-NB15 " + ("✓ Trained" if unsw_ok else "Not trained"),
                            CLR["green"] if unsw_ok else CLR["yellow"],
                        ),
                        badge(
                            "IoT Model " + ("✓ Trained" if iot_ok else "Not trained"),
                            CLR["green"] if iot_ok else CLR["yellow"],
                        ),
                        badge(
                            "CTI Records " + ("✓ Generated" if cti_ok else "Pending"),
                            CLR["green"] if cti_ok else CLR["muted"],
                        ),
                    ],
                    style={"display": "flex", "flexWrap": "wrap", "gap": "6px"},
                )
            ),
            html.Div(
                [
                    html.Div(
                        kpi_card(
                            "UNSW Accuracy",
                            f"{unsw_m.get('accuracy', 0) * 100:.1f}%" if unsw_ok else "—",
                            CLR["accent"],
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        kpi_card(
                            "UNSW F1-Score",
                            f"{unsw_m.get('f1_attack', 0) * 100:.1f}%" if unsw_ok else "—",
                            CLR["green"],
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        kpi_card(
                            "IoT Accuracy",
                            f"{iot_m.get('accuracy', 0) * 100:.1f}%" if iot_ok else "—",
                            CLR["accent"],
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        kpi_card("Threats Detected", str(n_threats), CLR["red"]),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        kpi_card("High Severity", str(n_high), CLR["yellow"]),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        kpi_card("Total Events", str(n_total), CLR["muted"]),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            card(
                                [
                                    section_title("UNSW-NB15 Performance"),
                                    dcc.Graph(
                                        figure=metrics_bar(unsw_m) if unsw_ok else empty_fig(),
                                        config={"displayModeBar": False},
                                    ),
                                ]
                            )
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            card(
                                [
                                    section_title("IoT Severity Breakdown"),
                                    dcc.Graph(
                                        figure=severity_pie(cti_df) if cti_ok else empty_fig(),
                                        config={"displayModeBar": False},
                                    ),
                                ]
                            )
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            card(
                                [
                                    section_title("MITRE ATT&CK Tactics"),
                                    dcc.Graph(
                                        figure=mitre_tactic_bar(cti_df) if cti_ok else empty_fig(),
                                        config={"displayModeBar": False},
                                    ),
                                ]
                            )
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            card(
                                [
                                    section_title("Prediction Confidence"),
                                    dcc.Graph(
                                        figure=confidence_histogram(cti_df) if cti_ok else empty_fig(),
                                        config={"displayModeBar": False},
                                    ),
                                ]
                            )
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
        ]
    )

def page_unsw():
    if STATE["unsw_metrics"] is None:
        return html.Div(
            [
                html.H4("UNSW-NB15 Binary Classification"),
                html.P(
                    "Click ⚡ Train All Models to begin.",
                    style={"color": CLR["muted"]},
                ),
            ]
        )

    m = STATE["unsw_metrics"]
    cm = STATE["unsw_cm"]
    lbl = STATE["unsw_labels"]
    imp = STATE["unsw_imp"]
    df = STATE["unsw_df"]

    return html.Div(
        [
            html.H4("UNSW-NB15 Binary Classification", style={"marginBottom": "4px"}),
            html.P(
                "RandomForest classifier — binary label (normal / attack)",
                style={
                    "color": CLR["muted"],
                    "marginBottom": "20px",
                    "fontSize": "13px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            figure=metric_gauge(m["accuracy"], "Accuracy", "#58a6ff"),
                            config={"displayModeBar": False},
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=metric_gauge(m["precision_attack"], "Precision", "#3fb950"),
                            config={"displayModeBar": False},
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=metric_gauge(m["recall_attack"], "Recall", "#d29922"),
                            config={"displayModeBar": False},
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=metric_gauge(m["f1_attack"], "F1-Score", "#58a6ff"),
                            config={"displayModeBar": False},
                        ),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Div(
                        card(
                            [
                                section_title("Confusion Matrix"),
                                dcc.Graph(
                                    figure=confusion_heatmap(cm, lbl),
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        card(
                            [
                                section_title("Feature Importances"),
                                dcc.Graph(
                                    figure=feature_importance_bar(imp),
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
            card(
                [
                    section_title("Label Distribution"),
                    dcc.Graph(
                        figure=class_distribution(df["label"], "UNSW-NB15 Label Distribution"),
                        config={"displayModeBar": False},
                    ),
                ]
            ),
        ]
    )


def page_iot():
    if STATE["iot_metrics"] is None:
        return html.Div(
            [
                html.H4("IoT Network Model"),
                html.P(
                    "Click ⚡ Train All Models to begin.",
                    style={"color": CLR["muted"]},
                ),
            ]
        )

    m = STATE["iot_metrics"]
    cm = STATE["iot_cm"]
    lbl = STATE["iot_labels"]
    imp = STATE["iot_imp"]
    df = STATE["iot_df"]

    attack_series = None
    if df is not None and isinstance(df, pd.DataFrame) and "type" in df.columns:
        attack_series = df["type"]

    children = [
        html.H4("IoT Network Dataset Model", style={"marginBottom": "4px"}),
        html.P(
            "RandomForest — train_test_network.csv",
            style={"color": CLR["muted"], "marginBottom": "20px", "fontSize": "13px"},
        ),
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        figure=metric_gauge(m["accuracy"], "Accuracy", "#58a6ff"),
                        config={"displayModeBar": False},
                    ),
                    style={"flex": "1"},
                ),
                html.Div(
                    dcc.Graph(
                        figure=metric_gauge(m["precision_attack"], "Precision", "#3fb950"),
                        config={"displayModeBar": False},
                    ),
                    style={"flex": "1"},
                ),
                html.Div(
                    dcc.Graph(
                        figure=metric_gauge(m["recall_attack"], "Recall", "#d29922"),
                        config={"displayModeBar": False},
                    ),
                    style={"flex": "1"},
                ),
                html.Div(
                    dcc.Graph(
                        figure=metric_gauge(m["f1_attack"], "F1-Score", "#58a6ff"),
                        config={"displayModeBar": False},
                    ),
                    style={"flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        html.Div(
            [
                html.Div(
                    card(
                        [
                            section_title("Confusion Matrix"),
                            dcc.Graph(
                                figure=confusion_heatmap(cm, lbl),
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),
                    style={"flex": "1"},
                ),
                html.Div(
                    card(
                        [
                            section_title("Feature Importances"),
                            dcc.Graph(
                                figure=feature_importance_bar(imp),
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),
                    style={"flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
    ]

    if attack_series is not None:
        children.append(
            card(
                [
                    section_title("Attack Type Distribution"),
                    dcc.Graph(
                        figure=attack_type_donut(attack_series),
                        config={"displayModeBar": False},
                    ),
                ]
            )
        )

    return html.Div(children)


def page_cti():
    cti_df = STATE["cti_df"]
    if cti_df is None:
        return html.Div(
            [
                html.H4("CTI Records"),
                html.P("Train models first.", style={"color": CLR["muted"]}),
            ]
        )

    display_cols = [
        c for c in [
            "event_id",
            "attack_detected",
            "confidence",
            "severity",
            "attack_type",
            "ioc.src_ip",
            "ioc.dst_ip",
            "ioc.protocol",
            "mitre_attack.tactic",
            "mitre_attack.technique_id",
            "description",
        ] if c in cti_df.columns
    ]

    table_data = cti_df[display_cols].head(500).to_dict("records")
    columns = [{"name": c.split(".")[-1], "id": c} for c in display_cols]

    return html.Div(
        [
            html.H4("Cyber Threat Intelligence Records", style={"marginBottom": "4px"}),
            html.P(
                f"{len(cti_df):,} records generated from IoT model predictions",
                style={"color": CLR["muted"], "marginBottom": "20px", "fontSize": "13px"},
            ),
            html.Div(
                [
                    html.Div(
                        card(
                            [
                                section_title("Severity Breakdown"),
                                dcc.Graph(
                                    figure=severity_pie(cti_df),
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        card(
                            [
                                section_title("MITRE ATT&CK Tactics"),
                                dcc.Graph(
                                    figure=mitre_tactic_bar(cti_df),
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                        style={"flex": "1"},
                    ),
                    html.Div(
                        card(
                            [
                                section_title("Confidence Distribution"),
                                dcc.Graph(
                                    figure=confidence_histogram(cti_df),
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
            card(
                [
                    section_title("CTI Event Table (first 500 rows)"),
                    dash_table.DataTable(
                        data=table_data,
                        columns=columns,
                        page_size=20,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#21262d",
                            "color": "#e6edf3",
                            "fontWeight": "600",
                            "border": f"1px solid {CLR['border']}",
                        },
                        style_cell={
                            "backgroundColor": "#161b22",
                            "color": "#e6edf3",
                            "border": f"1px solid {CLR['border']}",
                            "fontSize": "12px",
                            "padding": "8px",
                            "maxWidth": "200px",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                        },
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": '{severity} = "high"',
                                    "column_id": "severity",
                                },
                                "color": CLR["red"],
                                "fontWeight": "600",
                            },
                            {
                                "if": {
                                    "filter_query": '{severity} = "medium"',
                                    "column_id": "severity",
                                },
                                "color": CLR["yellow"],
                            },
                            {
                                "if": {
                                    "filter_query": '{severity} = "low"',
                                    "column_id": "severity",
                                },
                                "color": CLR["green"],
                            },
                        ],
                    ),
                ]
            ),
        ]
    )


def page_ioc():
    return html.Div(
        [
            html.H4("IoC Extractor & Threat Report Analyser", style={"marginBottom": "4px"}),
            html.P(
                "Paste or edit a threat report below and extract Indicators of Compromise.",
                style={"color": CLR["muted"], "marginBottom": "20px", "fontSize": "13px"},
            ),
            card(
                [
                    section_title("Threat Report Text"),
                    dcc.Textarea(
                        id="ioc-input",
                        value=SAMPLE_REPORT,
                        style={
                            "width": "100%",
                            "height": "180px",
                            "background": "#21262d",
                            "color": "#e6edf3",
                            "border": f"1px solid {CLR['border']}",
                            "borderRadius": "6px",
                            "padding": "10px",
                            "fontFamily": "monospace",
                            "fontSize": "12px",
                            "resize": "vertical",
                        },
                    ),
                    html.Button(
                        "Extract IoCs",
                        id="btn-extract",
                        n_clicks=0,
                        type="button",
                        style={
                            "marginTop": "10px",
                            "background": CLR["accent"],
                            "color": "#fff",
                            "border": "none",
                            "borderRadius": "6px",
                            "padding": "8px 20px",
                            "cursor": "pointer",
                            "fontWeight": "600",
                        },
                    ),
                ]
            ),
            html.Div(id="ioc-output"),
            card(
                [
                    section_title("Structured CTI Record (JSON)"),
                    html.Pre(
                        id="cti-record-out",
                        style={
                            "background": "#21262d",
                            "padding": "12px",
                            "borderRadius": "6px",
                            "fontSize": "12px",
                            "color": "#e6edf3",
                            "maxHeight": "300px",
                            "overflowY": "auto",
                            "whiteSpace": "pre-wrap",
                        },
                    ),
                ]
            ),
        ]
    )


def page_data():
    return html.Div(
        [
            html.H4("Data Explorer", style={"marginBottom": "4px"}),
            html.P(
                "Inspect distributions of key network features.",
                style={"color": CLR["muted"], "marginBottom": "20px", "fontSize": "13px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                section_title("Select Dataset"),
                                dcc.Dropdown(
                                    id="data-dataset-select",
                                    options=[
                                        {"label": "UNSW-NB15", "value": "unsw"},
                                        {"label": "IoT Network", "value": "iot"},
                                    ],
                                    value="unsw",
                                    clearable=False,
                                    style={"background": "#21262d", "color": "#000"},
                                ),
                            ]
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                section_title("Select Feature"),
                                dcc.Dropdown(
                                    id="data-col-select",
                                    clearable=False,
                                    style={"background": "#21262d", "color": "#000"},
                                ),
                            ]
                        ),
                        width=8,
                    ),
                ]
            ),
            html.Div(id="data-histogram-out", style={"marginTop": "16px"}),
            html.Div(id="data-dist-out", style={"marginTop": "8px"}),
        ]
    )


def page_logs():
    logs = STATE["training_log"] or ["No training log yet. Click ⚡ Train All Models."]
    return html.Div(
        [
            html.H4("Training Logs", style={"marginBottom": "16px"}),
            card(
                html.Pre(
                    "\n".join(logs),
                    style={
                        "background": "#21262d",
                        "padding": "14px",
                        "borderRadius": "6px",
                        "fontSize": "12px",
                        "color": "#e6edf3",
                        "maxHeight": "70vh",
                        "overflowY": "auto",
                        "whiteSpace": "pre-wrap",
                        "fontFamily": "monospace",
                    },
                )
            ),
        ]
    )


PAGE_MAP = {
    "tab-overview": page_overview,
    "tab-unsw": page_unsw,
    "tab-iot": page_iot,
    "tab-cti": page_cti,
    "tab-ioc": page_ioc,
    "tab-data": page_data,
    "tab-logs": page_logs,
}

# ──────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────

@app.callback(
    Output("active-tab", "data"),
    [Input(tab_id, "n_clicks") for _, _, tab_id in NAV_ITEMS],
    State("active-tab", "data"),
)
def set_active_tab(*args):
    current_tab = args[-1]
    triggered_id = ctx.triggered_id
    if triggered_id is None:
        return current_tab or "tab-overview"
    return triggered_id


@app.callback(
    [Output(tab_id, "style") for _, _, tab_id in NAV_ITEMS],
    Input("active-tab", "data"),
)
def highlight_active_tab(active_tab):
    return [nav_button_style(tab_id == active_tab) for _, _, tab_id in NAV_ITEMS]


@app.callback(
    Output("page-content", "children"),
    Input("active-tab", "data"),
    Input("poll-interval", "n_intervals"),
)
def render_page(tab, _):
    try:
        renderer = PAGE_MAP.get(tab, page_overview)
        return renderer()
    except Exception as e:
        return card(
            [
                html.H4("Page Error", style={"color": CLR["red"]}),
                html.Pre(
                    str(e),
                    style={
                        "whiteSpace": "pre-wrap",
                        "fontSize": "12px",
                        "color": CLR["text"],
                    },
                ),
            ]
        )


@app.callback(
    Output("poll-interval", "disabled"),
    Output("train-trigger", "data"),
    Input("btn-train", "n_clicks"),
    State("train-trigger", "data"),
    prevent_initial_call=True,
)
def start_training(n_clicks, trigger):
    if not n_clicks:
        return True, trigger

    STATE["training_done"] = False
    STATE["training_error"] = None
    STATE["training_log"] = ["[INFO] Training started..."]

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return False, (trigger or 0) + 1


@app.callback(
    Output("train-status", "children"),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    prevent_initial_call=True,
)
def update_status(_):
    if STATE["training_error"]:
        return f"❌ Error: {STATE['training_error'][:60]}", True
    if STATE["training_done"]:
        return "✅ Training complete", True

    last = STATE["training_log"][-1] if STATE["training_log"] else "..."
    return f"⏳ {last[:55]}", False


@app.callback(
    Output("ioc-output", "children"),
    Output("cti-record-out", "children"),
    Input("btn-extract", "n_clicks"),
    State("ioc-input", "value"),
    prevent_initial_call=True,
)
def extract_ioc_callback(n_clicks, text):
    if not text:
        return html.P("No text provided.", style={"color": CLR["muted"]}), ""

    iocs = extract_iocs(text)
    STATE["iocs"] = iocs

    cti = build_cti_record(text, iocs, "Manual extraction — NLP summary not run here.")
    cti_json = json.dumps(cti, indent=2)

    fig = ioc_table(iocs)
    ioc_out = card(
        [
            section_title("Extracted IoCs"),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ]
    )
    return ioc_out, cti_json


@app.callback(
    Output("data-col-select", "options"),
    Output("data-col-select", "value"),
    Input("data-dataset-select", "value"),
)
def populate_col_dropdown(dataset):
    df = STATE["unsw_df"] if dataset == "unsw" else STATE["iot_df"]
    if df is None:
        return [], None

    num_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    opts = [{"label": c, "value": c} for c in num_cols[:40]]
    return opts, (num_cols[0] if num_cols else None)


@app.callback(
    Output("data-histogram-out", "children"),
    Output("data-dist-out", "children"),
    Input("data-col-select", "value"),
    State("data-dataset-select", "value"),
)
def update_histogram(col, dataset):
    df = STATE["unsw_df"] if dataset == "unsw" else STATE["iot_df"]
    if df is None or col is None or col not in df.columns:
        return html.P("No data loaded yet.", style={"color": CLR["muted"]}), ""

    hist_fig = feature_histogram(df[col], col)

    target = "label" if "label" in df.columns else None
    dist_out = ""
    if target:
        title = "UNSW-NB15 Label Distribution" if dataset == "unsw" else "IoT Label Distribution"
        dist_fig = class_distribution(df[target], title)
        dist_out = card(dcc.Graph(figure=dist_fig, config={"displayModeBar": False}))

    return card(dcc.Graph(figure=hist_fig, config={"displayModeBar": False})), dist_out


# ──────────────────────────────────────────────────────────────
# TRAINING WORKER
# ──────────────────────────────────────────────────────────────

def log(msg):
    print(msg)
    STATE["training_log"].append(msg)


def run_training():
    try:
        log("[INFO] Loading UNSW-NB15 dataset...")
        if os.path.exists(UNSW_CSV):
            unsw_df = load_unsw(UNSW_CSV, MAX_ROWS)
            STATE["unsw_df"] = unsw_df
            log(f"[INFO] UNSW shape: {unsw_df.shape}")

            X, y, pre = prepare_unsw(unsw_df, target="label")
            log("[INFO] Training UNSW binary classifier...")
            model, Xtr, Xte, ytr, yte, ypred, metrics, cm_raw, report = train_pipeline(
    X, y, pre, N_ESTIMATORS, RANDOM_STATE, TEST_SIZE
)
            log("[DEBUG] UNSW confusion matrix:")
            log(str(cm_raw))
            log("[DEBUG] UNSW classification report:")
            log(report)

            save_model(model, os.path.join(os.path.dirname(__file__), "models", "unsw_model.pkl"))
            cm, labels = get_confusion_matrix(yte, ypred)
            imp = get_feature_importances(model)

            STATE["unsw_metrics"] = metrics
            STATE["unsw_cm"] = cm
            STATE["unsw_labels"] = labels
            STATE["unsw_imp"] = imp
            STATE["unsw_y_test"] = yte
            STATE["unsw_y_pred"] = ypred

            log(f"[INFO] UNSW — Accuracy: {metrics['accuracy']}, F1 (Attack): {metrics['f1_attack']}")
        else:
            log(f"[WARN] UNSW CSV not found at {UNSW_CSV}. Skipping.")

        log("[INFO] Loading IoT dataset...")
        if os.path.exists(IOT_CSV):
            iot_df = load_iot(IOT_CSV)
            STATE["iot_df"] = iot_df
            log(f"[INFO] IoT shape: {iot_df.shape}")

            log("[DEBUG] IoT label distribution:")
            log(str(iot_df["label"].value_counts(dropna=False)))

            dup_count = iot_df.duplicated().sum()
            log(f"[DEBUG] IoT duplicate rows: {dup_count}")
            tmp = iot_df.drop(columns=[c for c in ["src_ip", "dst_ip", "src_port", "dst_port", "type"] if c in iot_df.columns], errors="ignore")
            log(f"[DEBUG] IoT duplicates after dropping metadata: {tmp.duplicated().sum()}")
            
            
            X, y, le, pre, meta = prepare_iot(iot_df, target="label")
            log(f"[DEBUG] IoT shape after preprocessing/dedup: {X.shape}")
            log("[INFO] Training IoT classifier...")
            model_iot, Xtr, Xte, ytr, yte, ypred, metrics, cm_raw, report = train_pipeline(
    X, y, pre, N_ESTIMATORS, RANDOM_STATE, TEST_SIZE
)
            log("[DEBUG] IoT confusion matrix:")
            log(str(cm_raw))
            log("[DEBUG] IoT classification report:")
            log(report)

            save_model(model_iot, os.path.join(os.path.dirname(__file__), "models", "iot_model.pkl"))
            cm, labels = get_confusion_matrix(yte, ypred)
            imp = get_feature_importances(model_iot)

            STATE["iot_metrics"] = metrics
            STATE["iot_cm"] = cm
            STATE["iot_labels"] = labels
            STATE["iot_imp"] = imp

            log(f"[INFO] IoT — Accuracy: {metrics['accuracy']}, F1 (Attack): {metrics['f1_attack']}")

            log("[INFO] Generating enriched CTI records...")
            sample_size = min(2000, len(iot_df))
            sampled_df = iot_df.sample(sample_size, random_state=42)

            cti_raw = generate_cti_records(model_iot, sampled_df, "label")
            cti_df = cti_to_dataframe(cti_raw)
            STATE["cti_df"] = cti_df

            log(f"[INFO] CTI records generated: {len(cti_df)}")
        else:
            log(f"[WARN] IoT CSV not found at {IOT_CSV}. Skipping.")

        STATE["training_done"] = True
        log("[INFO] ✅ All training complete.")

    except Exception as exc:
        err = traceback.format_exc()
        STATE["training_error"] = str(exc)
        STATE["training_log"].append(f"[ERROR] {err}")
        print(err)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)

    print("\n🛡️  IoT ThreatIQ Dashboard starting...")
    print(f"   Open http://{APP_HOST}:{APP_PORT} in your browser\n")

    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG)