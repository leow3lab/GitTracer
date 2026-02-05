"""Repository configuration component for GitTracer."""

import os
from dash import html
import dash_bootstrap_components as dbc

_ALERT_COLORS = {"success": "success", "error": "danger", "warning": "warning", "info": "info"}


def create_repo_config():
    """Create the repository configuration card component."""
    label_style = {"fontFamily": "inherit"}
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4("Repository Configuration", className="sketch-header mb-0"),
                className="sketch-card-header",
            ),
            dbc.CardBody(
                [
                    html.Label(
                        "Repository Path (Local Path or GitHub URL):",
                        className="fw-bold mb-2",
                        style=label_style,
                    ),
                    dbc.Input(
                        id="repo-path-input",
                        placeholder="e.g., /path/to/vllm-ascend or https://github.com/vllm-project/vllm-ascend",
                        type="text",
                        value=os.getcwd(),
                        className="sketch-input mb-3",
                    ),
                    html.Label("Branch Name:", className="fw-bold mb-2", style=label_style),
                    dbc.Input(
                        id="branch-name-input",
                        value="main",
                        type="text",
                        className="sketch-input mb-3",
                    ),
                    html.Label(
                        "Number of Commits to Analyze:", className="fw-bold mb-2", style=label_style
                    ),
                    dbc.Input(
                        id="top-k-input",
                        value=50,
                        type="number",
                        min=1,
                        max=1000,
                        className="sketch-input mb-3",
                    ),
                    dbc.Button(
                        html.Div([html.I(className="bi bi-github me-2"), "Fetch & Analyze"]),
                        id="btn-fetch-commits",
                        color="primary",
                        size="lg",
                        className="sketch-button w-100",
                    ),
                    html.Small(
                        "Tip: if fetch fails, try again or verify `git clone` works in your terminal with the same URL first.",
                        className="text-muted mt-3 d-block",
                        style={"fontStyle": "italic"},
                    ),
                ]
            ),
        ],
        className="sketch-card",
    )


def create_status_alert(message, alert_type="info"):
    """Create a status alert component."""
    return dbc.Alert(
        message,
        id="status-alert",
        color=_ALERT_COLORS.get(alert_type, "info"),
        className="sketch-alert",
        dismissable=True,
        is_open=True,
    )
