"""Main layout for GitTracer Dash application."""

from dash import dcc, html
import dash_bootstrap_components as dbc
from frontend.styles import get_stylesheet
from frontend.layouts.components import create_repo_config


def create_layout():
    """
    Create the main layout for GitTracer app.

    Returns:
        dbc.Container: The root layout container
    """
    return dbc.Container([
        dcc.Store(id='stored-commits'),
        dcc.Store(id='stored-trajectories'),
        dcc.Store(id='selected-commit'),

        # Header Section
        dbc.Row([
            dbc.Col([
                html.H1(
                    "GitTracer",
                    className="text-center mb-2",
                    style={'fontSize': '2.5rem', 'fontWeight': 'bold'}
                ),
                html.P(
                    "SWE Trajectory Analysis Platform",
                    className="text-center text-muted",
                    style={'fontSize': '1.1rem'}
                )
            ], width=12)
        ], className="mb-4"),

        # Main Content Area
        dbc.Row([
            # Left Column - Configuration
            dbc.Col([
                create_repo_config()
            ], width=4, lg=4),

            # Right Column - Tabs and Content
            dbc.Col([
                html.Div(id="main-content-area")
            ], width=8, lg=8)
        ]),

        # Status Messages
        dbc.Row([
            dbc.Col([
                html.Div(id="status-container")
            ], width=12)
        ])
    ], fluid=True, className="py-4")


def get_external_stylesheets():
    """
    Get external stylesheets for the Dash app.

    Returns:
        list: List of stylesheet paths/URLs
    """
    # Use bootstrap as base, will override with custom CSS
    return [dbc.themes.FLATLY, get_stylesheet()]
