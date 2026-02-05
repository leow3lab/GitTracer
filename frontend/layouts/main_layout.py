"""Main layout for GitTracer Dash application."""

from dash import dcc, html
import dash_bootstrap_components as dbc
from frontend.styles import get_stylesheet
from frontend.layouts.components import (
    create_repo_config,
    create_trajectory_view,
)


def create_layout():
    """Create the main layout for GitTracer app."""
    return dbc.Container(
        [
            dcc.Store(id="stored-commits"),
            dcc.Store(id="stored-trajectories"),
            # Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1(
                                "GitTracer",
                                className="text-center mb-2",
                                style={"fontSize": "2.5rem", "fontWeight": "bold"},
                            ),
                            html.P(
                                "SWE Trajectory Analysis Platform",
                                className="text-center text-muted",
                                style={"fontSize": "1.1rem"},
                            ),
                        ],
                        width=12,
                    )
                ],
                className="mb-4",
            ),
            # Main Content
            dbc.Row(
                [
                    # Left - Configuration
                    dbc.Col(
                        [
                            create_repo_config(),
                            html.Div(id="branches-container", className="mt-3"),
                        ],
                        width=4,
                        lg=4,
                    ),
                    # Right - Trajectory View
                    dbc.Col(
                        [html.Div(id="trajectory-container", children=create_trajectory_view([]))],
                        width=8,
                        lg=8,
                    ),
                ]
            ),
            # Status Messages
            dbc.Row([dbc.Col([html.Div(id="status-container")], width=12)]),
        ],
        fluid=True,
        className="py-4",
    )


def get_external_stylesheets():
    """Get external stylesheets for the Dash app."""
    return [dbc.themes.FLATLY, get_stylesheet()]
