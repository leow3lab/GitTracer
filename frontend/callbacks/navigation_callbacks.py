"""Callbacks for navigation and tab switching."""

from dash import Input, Output
import dash_bootstrap_components as dbc
from frontend.layouts.components import (
    create_commits_tab,
    create_commits_detail_view,
    create_trajectory_view
)


def register_callbacks(app):
    """
    Register navigation and tab callbacks with the Dash app.

    Args:
        app: Dash application instance
    """
    @app.callback(
        Output("main-content-area", "children"),
        [Input("stored-commits", "data")],
        prevent_initial_call=False
    )
    def update_main_content(commits_data):
        """Update the main content area based on available data."""
        # Create tabs with content
        tabs = dbc.Tabs([
            dbc.Tab(
                label="Commits",
                tab_id="tab-commits",
                children=create_commits_tab()
            ),
            dbc.Tab(
                label="Commit Details",
                tab_id="tab-details",
                children=create_commits_detail_view()
            ),
            dbc.Tab(
                label="Trajectories",
                tab_id="tab-trajectories",
                children=create_trajectory_view(commits_data or [])
            ),
        ], id="main-tabs", active_tab="tab-commits", className="sketch-tabs")

        return tabs
