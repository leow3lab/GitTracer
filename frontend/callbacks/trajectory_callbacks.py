"""Callbacks for trajectory display and interaction."""

from dash import Input, Output, State
import dash_bootstrap_components as dbc
from frontend.layouts.components import create_commit_detail_with_markdown


def register_callbacks(app):
    """
    Register trajectory display callbacks with the Dash app.

    Args:
        app: Dash application instance
    """

    @app.callback(
        [Output("commit-detail-content", "children"), Output("selected-commit", "data")],
        [Input("commit-table", "derived_virtual_selected_rows")],
        [State("stored-commits", "data")],
        prevent_initial_call=True,
    )
    def display_commit_detail(selected_rows, stored_commits):
        """Display detailed view of selected commit."""
        if not selected_rows or not stored_commits:
            return "Select a commit to view details.", None

        row_idx = selected_rows[0]

        # Handle pagination offset
        if row_idx >= len(stored_commits):
            return "Invalid commit selection.", None

        commit_data = stored_commits[row_idx]

        detail_view = create_commit_detail_with_markdown(
            commit_data, idx=commit_data.get("idx", row_idx + 1)
        )

        return detail_view, commit_data
