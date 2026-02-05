"""Repository configuration component for GitTracer."""

import os
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_repo_config():
    """
    Create the repository configuration card component.

    Returns:
        dbc.Card: Configuration card with repo path, branch, and top_k inputs
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H4("Repository Configuration", className="sketch-header mb-0"),
                className="sketch-card-header",
            ),
            dbc.CardBody(
                [
                    # Repository Path Input
                    html.Label(
                        "Local Repository Path:",
                        className="fw-bold mb-2",
                        style={"fontFamily": "inherit"},
                    ),
                    dbc.Input(
                        id="repo-path-input",
                        placeholder="e.g., /path/to/your/repo",
                        type="text",
                        value=os.getcwd(),
                        className="sketch-input mb-3",
                    ),
                    # Branch Name Input
                    html.Label(
                        "Branch Name:", className="fw-bold mb-2", style={"fontFamily": "inherit"}
                    ),
                    dbc.Input(
                        id="branch-name-input",
                        value="main",
                        type="text",
                        className="sketch-input mb-3",
                    ),
                    # Top K Commits Input
                    html.Label(
                        "Number of Commits to Analyze:",
                        className="fw-bold mb-2",
                        style={"fontFamily": "inherit"},
                    ),
                    dbc.Input(
                        id="top-k-input",
                        value=50,
                        type="number",
                        min=1,
                        max=1000,
                        className="sketch-input mb-3",
                    ),
                    # Analyze Button
                    dbc.Button(
                        html.Div([html.I(className="bi bi-github me-2"), "Fetch & Analyze"]),
                        id="btn-fetch-commits",
                        color="primary",
                        size="lg",
                        className="sketch-button w-100",
                    ),
                    # Help Text
                    html.Small(
                        "Enter a local Git repository path to fetch commit history.",
                        className="text-muted mt-3 d-block",
                        style={"fontStyle": "italic"},
                    ),
                ]
            ),
        ],
        className="sketch-card",
    )


def create_status_alert(message, alert_type="info"):
    """
    Create a status alert component.

    Args:
        message: The message to display
        alert_type: One of 'success', 'error', 'warning', 'info'

    Returns:
        dbc.Alert: Alert component with sketch styling
    """
    color_map = {"success": "success", "error": "danger", "warning": "warning", "info": "info"}

    return dbc.Alert(
        message,
        id="status-alert",
        color=color_map.get(alert_type, "info"),
        className="sketch-alert",
        dismissable=True,
        is_open=True,
    )
