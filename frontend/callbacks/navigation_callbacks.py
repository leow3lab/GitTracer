"""Callbacks for navigation and tab switching."""

from dash import Input, Output, html
import dash_bootstrap_components as dbc
from frontend.layouts.components import create_trajectory_view


def register_callbacks(app):
    """Register navigation and tab callbacks with the Dash app."""

    @app.callback(
        Output("main-content-area", "children"),
        Input("stored-commits", "data"),
        prevent_initial_call=False,
    )
    def update_main_content(commits_data):
        """Update the main content area based on available data."""
        return dbc.Tabs(
            [
                dbc.Tab(
                    label="Commits",
                    tab_id="tab-commits",
                    children=html.Div(
                        [
                            html.H4("Commit History", className="mb-3"),
                            html.Div(
                                id="commit-table-container",
                                children=[
                                    html.P("No repository analyzed yet.", className="text-muted")
                                ],
                            ),
                        ],
                        id="commits-tab-content",
                    ),
                ),
                dbc.Tab(
                    label="Commit Details",
                    tab_id="tab-details",
                    children=html.Div(
                        [
                            html.H4("Commit Details", className="mb-3"),
                            html.Div(
                                id="commit-detail-content",
                                className="sketch-code p-3",
                                style={
                                    "minHeight": "400px",
                                    "maxHeight": "700px",
                                    "overflowY": "auto",
                                },
                                children="Select a commit to view details.",
                            ),
                        ]
                    ),
                ),
                dbc.Tab(
                    label="Trajectories",
                    tab_id="tab-trajectories",
                    children=create_trajectory_view(commits_data or []),
                ),
            ],
            id="main-tabs",
            active_tab="tab-commits",
            className="sketch-tabs",
        )
