import base64
import io
import json
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html

from config import MODEL_SCENARIO_RESULT_KEYS, SCENARIO_RESULT_KEYS
from loader import load_historical_data, standardize_accelerator_name
from loadgen_parser import LoadgenParser
from statistical_tests import AnomalyResult, get_group_stats, test_anomaly
from system_loader import load_system

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Inference v6.0 results - Results.csv")

historical_df = load_historical_data(DATA_PATH)
group_stats_df = get_group_stats(historical_df)

app = Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H1("MLPerf Inference Anomaly Detector", style={"color": "#2c3e50"}),
        html.P(
            f"Loaded {len(historical_df)} historical results across "
            f"{len(group_stats_df)} unique (model, scenario, accelerator) groups.",
            style={"color": "#7f8c8d"},
        ),

        html.Hr(),
        html.H2("Submit New Result"),
        html.Div(
            style={"display": "flex", "gap": "40px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "1", "minWidth": "300px"},
                    children=[
                        html.H4("1. Upload loadgen detail log"),
                        dcc.Upload(
                            id="upload-log",
                            children=html.Div(["Drag & drop or ", html.A("select mlperf_log_detail.txt")]),
                            style={
                                "width": "100%", "height": "80px", "lineHeight": "80px",
                                "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "8px",
                                "textAlign": "center", "borderColor": "#3498db", "color": "#3498db",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(id="log-filename", style={"marginTop": "6px", "color": "#27ae60", "fontSize": "14px"}),
                    ],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "300px"},
                    children=[
                        html.H4("2. Upload system description JSON"),
                        dcc.Upload(
                            id="upload-system",
                            children=html.Div(["Drag & drop or ", html.A("select system.json")]),
                            style={
                                "width": "100%", "height": "80px", "lineHeight": "80px",
                                "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "8px",
                                "textAlign": "center", "borderColor": "#9b59b6", "color": "#9b59b6",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(id="system-filename", style={"marginTop": "6px", "color": "#27ae60", "fontSize": "14px"}),
                    ],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "300px"},
                    children=[
                        html.H4("3. Model & scenario"),
                        html.Label("Model MLC"),
                        dcc.Dropdown(
                            id="dropdown-model",
                            options=[{"label": m, "value": m} for m in sorted(historical_df["Model MLC"].unique())],
                            placeholder="Select model...",
                            style={"marginBottom": "12px"},
                        ),
                        html.Label("Scenario"),
                        dcc.Dropdown(
                            id="dropdown-scenario",
                            options=[{"label": s, "value": s} for s in sorted(SCENARIO_RESULT_KEYS.keys())],
                            placeholder="Select scenario...",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "20px"},
            children=[
                html.Button(
                    "Run Anomaly Detection",
                    id="btn-run",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#2ecc71", "color": "white", "border": "none",
                        "padding": "12px 28px", "fontSize": "16px", "borderRadius": "6px",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),

        html.Div(id="result-section", style={"marginTop": "30px"}),

        html.Hr(),
        html.H2("Historical Data Groups"),
        html.P("Summary statistics for each (Model, Scenario, Accelerator) group (normalized by accelerator count):"),
        dash_table.DataTable(
            id="stats-table",
            columns=[
                {"name": c, "id": c}
                for c in ["Model MLC", "Scenario", "Accelerator", "Count", "Mean", "Std", "Min", "Max"]
            ],
            data=group_stats_df.round(4).to_dict("records"),
            sort_action="native",
            filter_action="native",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
            style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
            ],
        ),
    ],
)


def _decode_upload(contents):
    _, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)


@callback(Output("log-filename", "children"), Input("upload-log", "filename"))
def show_log_name(filename):
    return f"Selected: {filename}" if filename else ""


@callback(Output("system-filename", "children"), Input("upload-system", "filename"))
def show_system_name(filename):
    return f"Selected: {filename}" if filename else ""


@callback(
    Output("result-section", "children"),
    Input("btn-run", "n_clicks"),
    State("upload-log", "contents"),
    State("upload-log", "filename"),
    State("upload-system", "contents"),
    State("dropdown-model", "value"),
    State("dropdown-scenario", "value"),
    prevent_initial_call=True,
)
def run_detection(n_clicks, log_contents, log_filename, system_contents, model, scenario):
    errors = []
    if not log_contents:
        errors.append("Upload a loadgen detail log file.")
    if not system_contents:
        errors.append("Upload a system description JSON file.")
    if not model:
        errors.append("Select a model.")
    if not scenario:
        errors.append("Select a scenario.")
    if errors:
        return html.Ul([html.Li(e, style={"color": "red"}) for e in errors])

    # Parse system JSON
    try:
        sys_bytes = _decode_upload(system_contents)
        sys_info = load_system_from_bytes(sys_bytes)
    except Exception as exc:
        return html.P(f"Error parsing system JSON: {exc}", style={"color": "red"})

    # Parse log file
    try:
        log_bytes = _decode_upload(log_contents)
        result_key = MODEL_SCENARIO_RESULT_KEYS.get((model, scenario), SCENARIO_RESULT_KEYS.get(scenario))
        if result_key is None:
            return html.P(f"Unknown scenario: {scenario}", style={"color": "red"})

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(log_bytes)
            tmp_path = tmp.name
        try:
            parser = LoadgenParser(tmp_path, strict=False)
            raw_result = parser[result_key]
        finally:
            os.unlink(tmp_path)

        if raw_result is None:
            return html.P(f"Key '{result_key}' not found in log for scenario {scenario}.", style={"color": "red"})
        result = float(raw_result)
    except Exception as exc:
        return html.P(f"Error parsing log file: {exc}", style={"color": "red"})

    anomaly: AnomalyResult = test_anomaly(
        historical_df,
        model=model,
        scenario=scenario,
        accelerator_std=sys_info["accelerator_std"],
        result=result,
        total_accelerators=sys_info["total_accelerators"],
    )

    return _render_result(anomaly, sys_info)


def load_system_from_bytes(data: bytes) -> dict:
    obj = json.loads(data.decode("utf-8"))
    raw_accel = obj.get("accelerator_model_name", "")
    per_node = int(obj.get("accelerators_per_node", 1))
    nodes = int(obj.get("number_of_nodes", 1))
    total = per_node * nodes
    return {
        "accelerator_model_name": raw_accel,
        "accelerator_std": standardize_accelerator_name(raw_accel),
        "accelerators_per_node": per_node,
        "number_of_nodes": nodes,
        "total_accelerators": total,
        "system_name": obj.get("system_name", ""),
        "submitter": obj.get("submitter", ""),
        "framework": obj.get("framework", ""),
    }


def _render_result(anomaly: AnomalyResult, sys_info: dict):
    color = "#e74c3c" if anomaly.is_anomaly else "#27ae60"
    label = "ANOMALY DETECTED" if anomaly.is_anomaly else "RESULT IS NORMAL"

    info_rows = [
        ("Model", anomaly.model),
        ("Scenario", anomaly.scenario),
        ("Accelerator (raw)", sys_info["accelerator_model_name"]),
        ("Accelerator (standardized)", anomaly.accelerator),
        ("Total Accelerators", str(anomaly.total_accelerators)),
        ("Raw Result", f"{anomaly.new_value:.4f}"),
        ("Normalized Result (per accel)", f"{anomaly.normalized_value:.4f}"),
    ]
    if anomaly.mean is not None:
        info_rows += [
            ("Historical Mean (per accel)", f"{anomaly.mean:.4f}"),
            ("Historical Std", f"{anomaly.std:.4f}"),
        ]
    if anomaly.z_score is not None:
        info_rows += [
            ("Z-Score", f"{anomaly.z_score:.4f}"),
            ("P-Value", f"{anomaly.p_value:.4f}"),
        ]
    info_rows.append(("Reason", anomaly.reason))

    table = html.Table(
        [html.Tr([html.Td(k, style={"fontWeight": "bold", "paddingRight": "20px", "paddingBottom": "6px"}),
                  html.Td(v)]) for k, v in info_rows],
        style={"marginTop": "12px"},
    )

    fig = _build_distribution_plot(anomaly)

    return html.Div([
        html.Div(
            label,
            style={
                "backgroundColor": color, "color": "white", "padding": "16px 24px",
                "borderRadius": "8px", "fontSize": "20px", "fontWeight": "bold",
                "display": "inline-block", "marginBottom": "16px",
            },
        ),
        table,
        dcc.Graph(figure=fig, style={"marginTop": "20px"}),
    ])


def _build_distribution_plot(anomaly: AnomalyResult) -> go.Figure:
    fig = go.Figure()

    hist_vals = anomaly.historical_values
    if len(hist_vals) < 2:
        fig.add_annotation(text="Insufficient data for distribution plot", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False, font_size=16)
        return fig

    mean = anomaly.mean if anomaly.mean is not None else np.mean(hist_vals)
    std = anomaly.std if anomaly.std is not None else np.std(hist_vals, ddof=1)

    span = max(abs(anomaly.normalized_value - mean), std * 3.5) * 1.2 if std > 0 else 1
    x_min = mean - span
    x_max = mean + span
    x = np.linspace(x_min, x_max, 300)

    from scipy.stats import norm
    y = norm.pdf(x, mean, std) if std > 0 else np.zeros_like(x)

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Historical distribution",
                             line={"color": "#3498db", "width": 2}))

    fig.add_trace(go.Histogram(
        x=hist_vals, histnorm="probability density",
        name="Historical samples", opacity=0.4,
        marker_color="#3498db",
    ))

    marker_color = "#e74c3c" if anomaly.is_anomaly else "#27ae60"
    fig.add_vline(
        x=anomaly.normalized_value,
        line_color=marker_color, line_width=3, line_dash="dash",
        annotation_text=f"New result ({anomaly.normalized_value:.3f})",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Distribution: {anomaly.model} / {anomaly.scenario} / {anomaly.accelerator}",
        xaxis_title="Normalized result (per accelerator)",
        yaxis_title="Density",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        template="plotly_white",
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
