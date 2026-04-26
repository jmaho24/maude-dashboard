"""
FDA MAUDE — Script 05: Dashboard
==================================
Two-tab dashboard:
  Tab 1: Complaint Classifier — centered, clean, prediction-first
  Tab 2: Data Explorer — charts, filters, manufacturer breakdown

Run locally:
    cd C:\Capstone
    venv\Scripts\activate
    python scripts\05_app.py
    Open http://localhost:8050
"""

import os
import re
import joblib
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "aggregated")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "maude_classifier.pkl")

# ── Load data ─────────────────────────────────────────────────────────────────
events_by_year     = pd.read_csv(os.path.join(DATA_DIR, "events_by_year.csv"))
class_distribution = pd.read_csv(os.path.join(DATA_DIR, "class_distribution.csv"))
top_manufacturers  = pd.read_csv(os.path.join(DATA_DIR, "top_manufacturers.csv"))

# ── Load model ────────────────────────────────────────────────────────────────
pipeline = joblib.load(MODEL_PATH)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_LABELS = {"M": "Malfunction", "IN": "Injury", "D": "Death", "O": "Other"}
CLASS_COLORS = {
    "Malfunction": "#4A90D9",
    "Injury"     : "#F5A623",
    "Death"      : "#E74C3C",
    "Other"      : "#8E8E8E",
}
BADGE_COLORS = {
    "M" : {"bg": "#1a3a5c", "border": "#4A90D9", "text": "#4A90D9"},
    "IN": {"bg": "#3d2a00", "border": "#F5A623", "text": "#F5A623"},
    "D" : {"bg": "#3d0000", "border": "#E74C3C", "text": "#E74C3C"},
    "O" : {"bg": "#2a2a2a", "border": "#8E8E8E", "text": "#8E8E8E"},
}

EXAMPLE_NARRATIVES = [
    ("Malfunction", "The infusion pump displayed an occlusion alarm and ceased delivery of the programmed medication dose. The device was removed from service and returned to the manufacturer for inspection."),
    ("Injury",      "Patient was hospitalized following device use. The patient sustained internal hemorrhaging requiring emergency surgical intervention. The device was identified as the probable cause of injury."),
    ("Death",       "The patient was found unresponsive approximately four hours after device implantation. Emergency services were contacted. The patient was pronounced deceased at the hospital. The device was explanted and sent for analysis."),
]

DARK_BG      = "#0f1117"
CARD_BG      = "#1a1d27"
BORDER_COLOR = "#2d3147"
TEXT_PRIMARY = "#e8eaf6"
TEXT_MUTED   = "#8b90a7"
PAGE_SIZE    = 10

MANUFACTURERS = sorted(top_manufacturers["MANUFACTURER_G1_NAME"].unique().tolist())
YEAR_MIN      = int(events_by_year["YEAR"].min())
YEAR_MAX      = int(events_by_year["YEAR"].max())

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="FDA MAUDE Complaint Intelligence",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server

# ── Styles ────────────────────────────────────────────────────────────────────
CARD_STYLE = {
    "background"  : CARD_BG,
    "border"      : f"1px solid {BORDER_COLOR}",
    "borderRadius": "10px",
    "padding"     : "20px",
    "marginBottom": "16px",
}

LABEL_STYLE = {
    "color"        : TEXT_MUTED,
    "fontSize"     : "11px",
    "fontWeight"   : "600",
    "letterSpacing": "0.08em",
    "textTransform": "uppercase",
    "marginBottom" : "8px",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(color=TEXT_PRIMARY, family="system-ui, sans-serif"),
    margin       =dict(l=40, r=20, t=10, b=40),
    legend       =dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis        =dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR),
    yaxis        =dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR),
)

BTN_STYLE = {
    "background"  : CARD_BG,
    "color"       : TEXT_PRIMARY,
    "border"      : f"1px solid {BORDER_COLOR}",
    "borderRadius": "6px",
    "padding"     : "6px 16px",
    "fontSize"    : "13px",
    "cursor"      : "pointer",
    "fontFamily"  : "system-ui, sans-serif",
}

BTN_DISABLED_STYLE = {
    **BTN_STYLE,
    "color"  : TEXT_MUTED,
    "cursor" : "not-allowed",
    "opacity": "0.4",
}

TAB_STYLE = {
    "background"  : DARK_BG,
    "color"       : TEXT_MUTED,
    "border"      : f"1px solid {BORDER_COLOR}",
    "borderRadius": "6px 6px 0 0",
    "padding"     : "10px 24px",
    "fontSize"    : "13px",
    "fontWeight"  : "500",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "background"  : CARD_BG,
    "color"       : TEXT_PRIMARY,
    "borderBottom": f"2px solid #4A90D9",
}

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"background": DARK_BG, "minHeight": "100vh", "fontFamily": "system-ui, sans-serif"},
    children=[

        # Header
        html.Div(
            style={
                "background"  : CARD_BG,
                "borderBottom": f"1px solid {BORDER_COLOR}",
                "padding"     : "20px 32px",
                "marginBottom": "0",
            },
            children=[
                html.H1(
                    "FDA MAUDE Complaint Intelligence System",
                    style={"color": TEXT_PRIMARY, "margin": "0", "fontSize": "22px", "fontWeight": "600"},
                ),
                html.P(
                    f"Trained on {int(class_distribution['COUNT'].sum()):,} adverse event reports · 1992–2025 · SGD LinearSVC · 92.48% accuracy",
                    style={"color": TEXT_MUTED, "margin": "4px 0 0", "fontSize": "13px"},
                ),
            ],
        ),

        # Tabs
        dcc.Tabs(
            id="main-tabs",
            value="tab-classify",
            children=[
                dcc.Tab(label="Complaint Classifier", value="tab-classify", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="Data Explorer",        value="tab-explore",  style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            ],
            style={"padding": "0 24px", "background": DARK_BG, "borderBottom": f"1px solid {BORDER_COLOR}"},
            colors={"border": BORDER_COLOR, "primary": "#4A90D9", "background": DARK_BG},
        ),

        html.Div(id="tab-content", style={"padding": "24px"}),
    ],
)


# ── Tab content ───────────────────────────────────────────────────────────────
@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab(tab):
    if tab == "tab-classify":
        return render_classifier_tab()
    else:
        return render_explorer_tab()


def render_classifier_tab():
    return html.Div([

        # Hero section
        html.Div(
            style={"textAlign": "center", "padding": "40px 0 32px"},
            children=[
                html.H2(
                    "Classify a complaint narrative",
                    style={"color": TEXT_PRIMARY, "fontSize": "28px", "fontWeight": "600", "margin": "0 0 10px"},
                ),
                html.P(
                    "Paste any FDA MAUDE complaint narrative below. The model will instantly predict the event category.",
                    style={"color": TEXT_MUTED, "fontSize": "15px", "margin": "0"},
                ),
            ],
        ),

        # Main classifier card
        html.Div(
            style={
                **CARD_STYLE,
                "maxWidth"  : "820px",
                "margin"    : "0 auto 24px",
                "padding"   : "32px",
            },
            children=[
                dcc.Textarea(
                    id="narrative-input",
                    placeholder="Paste a complaint narrative here...",
                    style={
                        "width"      : "100%",
                        "height"     : "140px",
                        "background" : DARK_BG,
                        "border"     : f"1px solid {BORDER_COLOR}",
                        "borderRadius": "8px",
                        "color"      : TEXT_PRIMARY,
                        "padding"    : "14px",
                        "fontSize"   : "14px",
                        "resize"     : "vertical",
                        "boxSizing"  : "border-box",
                        "fontFamily" : "system-ui, sans-serif",
                        "lineHeight" : "1.6",
                    },
                    maxLength=5000,
                ),

                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginTop": "14px"},
                    children=[
                        html.Button(
                            "Classify",
                            id="classify-btn",
                            n_clicks=0,
                            style={
                                "background"  : "#4A90D9",
                                "color"       : "#ffffff",
                                "border"      : "none",
                                "borderRadius": "8px",
                                "padding"     : "10px 36px",
                                "fontSize"    : "15px",
                                "fontWeight"  : "600",
                                "cursor"      : "pointer",
                                "letterSpacing": "0.02em",
                            },
                        ),
                        html.P(
                            "Decision support only — results should be reviewed by a qualified professional.",
                            style={"color": TEXT_MUTED, "fontSize": "11px", "margin": "0", "maxWidth": "340px", "textAlign": "right"},
                        ),
                    ],
                ),

                html.Div(id="prediction-output", style={"marginTop": "24px"}),
            ],
        ),

        # Example narratives
        html.Div(
            style={"maxWidth": "820px", "margin": "0 auto 24px"},
            children=[
                html.P("Try an example", style={**LABEL_STYLE, "marginBottom": "12px"}),
                html.Div(
                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                    children=[
                        html.Button(
                            f"Example: {label}",
                            id=f"example-{label.lower()}",
                            n_clicks=0,
                            style={
                                **BTN_STYLE,
                                "color"       : CLASS_COLORS.get(label, TEXT_MUTED),
                                "border"      : f"1px solid {CLASS_COLORS.get(label, BORDER_COLOR)}",
                                "borderRadius": "20px",
                                "padding"     : "6px 16px",
                                "fontSize"    : "12px",
                            },
                        )
                        for label, _ in EXAMPLE_NARRATIVES
                    ],
                ),
            ],
        ),

        # Model metrics row
        html.Div(
            style={"maxWidth": "820px", "margin": "0 auto"},
            children=[
                html.P("Model performance", style={**LABEL_STYLE, "marginBottom": "12px"}),
                html.Div(
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                    children=[
                        html.Div(
                            style={
                                **CARD_STYLE,
                                "flex"        : "1",
                                "minWidth"    : "120px",
                                "textAlign"   : "center",
                                "padding"     : "16px",
                                "marginBottom": "0",
                            },
                            children=[
                                html.P(f1, style={"color": color, "fontSize": "22px", "fontWeight": "700", "margin": "0 0 4px"}),
                                html.P(label, style={"color": TEXT_MUTED, "fontSize": "11px", "margin": "0"}),
                            ]
                        )
                        for label, f1, color in [
                            ("Overall accuracy", "92.48%", TEXT_PRIMARY),
                            ("Malfunction F1",   "0.951",  "#4A90D9"),
                            ("Injury F1",        "0.913",  "#F5A623"),
                            ("Death F1",         "0.803",  "#E74C3C"),
                            ("Other F1",         "0.197",  "#8E8E8E"),
                        ]
                    ],
                ),
            ],
        ),
    ])


def render_explorer_tab():
    return html.Div([
        html.Div(
            style={"display": "flex", "gap": "20px"},
            children=[

                # Sidebar
                html.Div(
                    style={"width": "220px", "flexShrink": "0"},
                    children=[
                        html.Div(style=CARD_STYLE, children=[
                            html.P("Filters", style={**LABEL_STYLE, "marginBottom": "16px"}),

                            html.P("Year range", style=LABEL_STYLE),
                            dcc.RangeSlider(
                                id="year-slider",
                                min=YEAR_MIN, max=YEAR_MAX,
                                value=[YEAR_MIN, YEAR_MAX],
                                marks={
                                    YEAR_MIN: {"label": str(YEAR_MIN), "style": {"color": TEXT_MUTED, "fontSize": "10px"}},
                                    2000    : {"label": "2000",        "style": {"color": TEXT_MUTED, "fontSize": "10px"}},
                                    2010    : {"label": "2010",        "style": {"color": TEXT_MUTED, "fontSize": "10px"}},
                                    2020    : {"label": "2020",        "style": {"color": TEXT_MUTED, "fontSize": "10px"}},
                                    YEAR_MAX: {"label": str(YEAR_MAX), "style": {"color": TEXT_MUTED, "fontSize": "10px"}},
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),

                            html.Div(style={"height": "16px"}),

                            html.P("Event type", style=LABEL_STYLE),
                            dcc.Checklist(
                                id="event-type-filter",
                                options=[
                                    {"label": " Malfunction", "value": "M"},
                                    {"label": " Injury",      "value": "IN"},
                                    {"label": " Death",       "value": "D"},
                                    {"label": " Other",       "value": "O"},
                                ],
                                value=["M", "IN", "D", "O"],
                                labelStyle={"color": "#ffffff", "fontSize": "13px", "display": "block", "lineHeight": "2.2"},
                                inputStyle={"marginRight": "8px"},
                            ),
                        ]),
                    ],
                ),

                # Charts
                html.Div(
                    style={"flex": "1", "minWidth": "0"},
                    children=[

                        html.Div(style=CARD_STYLE, children=[
                            html.P("Adverse event reports over time", style={**LABEL_STYLE, "marginBottom": "4px"}),
                            dcc.Graph(id="time-series-chart", style={"height": "280px"}, config={"displayModeBar": False}),
                        ]),

                        html.Div(style={"display": "flex", "gap": "16px"}, children=[

                            html.Div(style={**CARD_STYLE, "flex": "1", "marginBottom": "0"}, children=[
                                html.P("Class distribution", style={**LABEL_STYLE, "marginBottom": "4px"}),
                                dcc.Graph(id="distribution-chart", style={"height": "300px"}, config={"displayModeBar": False}),
                            ]),

                            html.Div(style={**CARD_STYLE, "flex": "2", "marginBottom": "0"}, children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                                    children=[
                                        html.P(id="manufacturer-chart-title", children="Top manufacturers by complaint volume", style={**LABEL_STYLE, "marginBottom": "0"}),
                                        dcc.Dropdown(
                                            id="manufacturer-filter",
                                            options=[{"label": m, "value": m} for m in MANUFACTURERS],
                                            placeholder="Filter by manufacturer...",
                                            clearable=True,
                                            style={"fontSize": "12px", "width": "240px"},
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="manufacturer-chart", style={"height": "260px"}, config={"displayModeBar": False}),
                                html.Div(
                                    style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "12px", "marginTop": "8px"},
                                    children=[
                                        html.Button("← Prev", id="prev-btn", n_clicks=0, style=BTN_DISABLED_STYLE),
                                        html.Span(id="page-indicator", style={"color": TEXT_MUTED, "fontSize": "12px"}),
                                        html.Button("Next →", id="next-btn", n_clicks=0, style=BTN_STYLE),
                                        dcc.Store(id="current-page", data=0),
                                    ],
                                ),
                            ]),
                        ]),
                    ],
                ),
            ],
        ),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

# Example buttons populate the text area
@callback(
    Output("narrative-input", "value"),
    Input("example-malfunction", "n_clicks"),
    Input("example-injury",      "n_clicks"),
    Input("example-death",       "n_clicks"),
    prevent_initial_call=True,
)
def load_example(m, i, d):
    from dash import ctx
    triggered = ctx.triggered_id
    mapping = {
        "example-malfunction": EXAMPLE_NARRATIVES[0][1],
        "example-injury"     : EXAMPLE_NARRATIVES[1][1],
        "example-death"      : EXAMPLE_NARRATIVES[2][1],
    }
    return mapping.get(triggered, "")


@callback(
    Output("prediction-output", "children"),
    Input("classify-btn", "n_clicks"),
    State("narrative-input", "value"),
    prevent_initial_call=True,
)
def classify_narrative(n_clicks, narrative):
    if not narrative or not narrative.strip():
        return html.P("Please enter a narrative before classifying.", style={"color": TEXT_MUTED, "fontSize": "13px"})

    clean           = re.sub(r"[^\w\s.,;:()\-]", "", narrative.strip())[:5000]
    predicted_code  = pipeline.predict([clean])[0]
    predicted_label = CLASS_LABELS.get(predicted_code, predicted_code)

    scores         = pipeline.decision_function([clean])[0]
    classes        = pipeline.classes_
    scores_shifted = scores - scores.min()
    total          = scores_shifted.sum()
    confidence     = {
        cls: float(scores_shifted[i] / total * 100) if total > 0 else 25.0
        for i, cls in enumerate(classes)
    }

    badge = BADGE_COLORS.get(predicted_code, BADGE_COLORS["O"])

    return html.Div([
        html.Div(
            style={
                "background"  : badge["bg"],
                "border"      : f"2px solid {badge['border']}",
                "borderRadius": "8px",
                "padding"     : "16px 20px",
                "marginBottom": "20px",
                "display"     : "flex",
                "alignItems"  : "center",
                "gap"         : "16px",
            },
            children=[
                html.Div(
                    predicted_label.upper(),
                    style={
                        "color"        : badge["text"],
                        "fontSize"     : "20px",
                        "fontWeight"   : "700",
                        "letterSpacing": "0.06em",
                    },
                ),
                html.Div(
                    style={"borderLeft": f"1px solid {badge['border']}", "paddingLeft": "16px"},
                    children=[
                        html.P("Predicted event category", style={"color": badge["text"], "fontSize": "11px", "fontWeight": "600", "margin": "0 0 2px", "opacity": "0.7", "textTransform": "uppercase", "letterSpacing": "0.06em"}),
                        html.P(f"Classified using SGD LinearSVC · 92.48% overall accuracy", style={"color": badge["text"], "fontSize": "12px", "margin": "0", "opacity": "0.6"}),
                    ],
                ),
            ],
        ),

        html.P("Confidence scores", style=LABEL_STYLE),
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={
                        "flex"        : "1",
                        "minWidth"    : "140px",
                        "background"  : DARK_BG,
                        "borderRadius": "8px",
                        "padding"     : "12px 14px",
                        "border"      : f"1px solid {BORDER_COLOR}",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"},
                            children=[
                                html.Span(CLASS_LABELS.get(cls, cls), style={"color": TEXT_PRIMARY, "fontSize": "13px", "fontWeight": "500"}),
                                html.Span(f"{confidence.get(cls, 0):.1f}%", style={"color": CLASS_COLORS.get(CLASS_LABELS.get(cls, cls), TEXT_MUTED), "fontSize": "13px", "fontWeight": "600"}),
                            ],
                        ),
                        html.Div(
                            style={"background": BORDER_COLOR, "borderRadius": "3px", "height": "4px"},
                            children=[
                                html.Div(style={
                                    "background"  : CLASS_COLORS.get(CLASS_LABELS.get(cls, cls), "#888"),
                                    "borderRadius": "3px",
                                    "height"      : "4px",
                                    "width"       : f"{confidence.get(cls, 0):.1f}%",
                                }),
                            ],
                        ),
                    ],
                )
                for cls in ["M", "IN", "D", "O"]
            ],
        ),
    ])


@callback(
    Output("time-series-chart", "figure"),
    Input("year-slider", "value"),
    Input("event-type-filter", "value"),
)
def update_time_series(year_range, event_types):
    df = events_by_year[
        (events_by_year["YEAR"] >= year_range[0]) &
        (events_by_year["YEAR"] <= year_range[1]) &
        (events_by_year["EVENT_TYPE"].isin(event_types or []))
    ]
    fig = go.Figure()
    for event_type in (event_types or []):
        subset = df[df["EVENT_TYPE"] == event_type]
        label  = CLASS_LABELS.get(event_type, event_type)
        color  = CLASS_COLORS.get(label, "#888")
        fig.add_trace(go.Scatter(
            x=subset["YEAR"], y=subset["COUNT"],
            mode="lines", name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Reports: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(**PLOT_LAYOUT)
    return fig


@callback(
    Output("distribution-chart", "figure"),
    Input("event-type-filter", "value"),
)
def update_distribution(event_types):
    df     = class_distribution[class_distribution["EVENT_TYPE"].isin(event_types or [])]
    colors = [CLASS_COLORS.get(row["EVENT_LABEL"], "#888") for _, row in df.iterrows()]
    fig    = go.Figure(go.Bar(
        x=df["EVENT_LABEL"], y=df["COUNT"],
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, showlegend=False)
    return fig


@callback(
    Output("current-page", "data"),
    Input("event-type-filter", "value"),
    Input("manufacturer-filter", "value"),
    Input("prev-btn", "n_clicks"),
    Input("next-btn", "n_clicks"),
    State("current-page", "data"),
)
def update_page(event_types, selected_manufacturer, prev_clicks, next_clicks, current_page):
    from dash import ctx
    triggered = ctx.triggered_id
    if triggered in ("event-type-filter", "manufacturer-filter"):
        return 0
    elif triggered == "prev-btn":
        return max(0, current_page - 1)
    elif triggered == "next-btn":
        return current_page + 1
    return current_page


@callback(
    Output("manufacturer-chart", "figure"),
    Output("manufacturer-chart-title", "children"),
    Output("page-indicator", "children"),
    Output("prev-btn", "style"),
    Output("next-btn", "style"),
    Input("current-page", "data"),
    Input("event-type-filter", "value"),
    Input("manufacturer-filter", "value"),
)
def update_manufacturers(page, event_types, selected_manufacturer):
    df = top_manufacturers[top_manufacturers["EVENT_TYPE"].isin(event_types or [])]

    if selected_manufacturer:
        mfr_data = df[df["MANUFACTURER_G1_NAME"] == selected_manufacturer]
        fig      = go.Figure()
        for event_type in ["M", "IN", "D", "O"]:
            subset = mfr_data[mfr_data["EVENT_TYPE"] == event_type]
            if len(subset) == 0:
                continue
            label = CLASS_LABELS.get(event_type, event_type)
            color = CLASS_COLORS.get(label, "#888")
            fig.add_trace(go.Bar(
                name=label,
                x=subset["COUNT"],
                y=subset["MANUFACTURER_G1_NAME"],
                orientation="h",
                marker_color=color,
                hovertemplate=f"<b>{label}</b><br>Reports: %{{x:,}}<extra></extra>",
            ))
        fig.update_layout(**PLOT_LAYOUT, barmode="stack", showlegend=True)
        return fig, f"Event breakdown — {selected_manufacturer[:40]}", "", BTN_DISABLED_STYLE, BTN_DISABLED_STYLE

    all_sorted  = (
        df.groupby("MANUFACTURER_G1_NAME")["COUNT"]
        .sum().sort_values(ascending=False).index.tolist()
    )
    total_pages = max(1, -(-len(all_sorted) // PAGE_SIZE))
    page        = min(page, total_pages - 1)
    start       = page * PAGE_SIZE
    end         = start + PAGE_SIZE
    page_mfrs   = list(reversed(all_sorted[start:end]))
    df_page     = df[df["MANUFACTURER_G1_NAME"].isin(page_mfrs)]

    fig = go.Figure()
    for event_type in ["M", "IN", "D", "O"]:
        subset = df_page[df_page["EVENT_TYPE"] == event_type].set_index("MANUFACTURER_G1_NAME")
        label  = CLASS_LABELS.get(event_type, event_type)
        color  = CLASS_COLORS.get(label, "#888")
        counts = [subset.loc[m, "COUNT"] if m in subset.index else 0 for m in page_mfrs]
        fig.add_trace(go.Bar(
            name=label, x=counts, y=page_mfrs, orientation="h",
            marker_color=color,
            hovertemplate=f"<b>{label}</b><br>%{{y}}<br>Reports: %{{x:,}}<extra></extra>",
        ))
    fig.update_layout(**PLOT_LAYOUT, barmode="stack", showlegend=True)

    rank_end   = min(end, len(all_sorted))
    title      = f"Manufacturers by complaint volume (ranked #{start+1}–{rank_end})"
    page_text  = f"Page {page+1} of {total_pages}"
    prev_style = BTN_DISABLED_STYLE if page == 0 else BTN_STYLE
    next_style = BTN_DISABLED_STYLE if page >= total_pages - 1 else BTN_STYLE

    return fig, title, page_text, prev_style, next_style


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)