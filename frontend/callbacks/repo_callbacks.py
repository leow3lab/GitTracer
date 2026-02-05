"""Callbacks for repository analysis functionality."""

import os
import sys
from dash import Input, Output, State
import dash_bootstrap_components as dbc

# Import GitDataFetcher from root app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
try:
    from app import GitDataFetcher
except ImportError:
    GitDataFetcher = None


def validate_repo_path(path):
    """
    Validate that a repository path exists and is a git repository.

    Args:
        path: Path to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    if not path:
        return False, "Repository path is required"

    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"

    git_dir = os.path.join(path, '.git')
    if not os.path.exists(git_dir):
        return False, f"Not a Git repository (no .git directory found)"

    return True, None


def register_callbacks(app):
    """
    Register repository analysis callbacks with the Dash app.

    Args:
        app: Dash application instance
    """
    @app.callback(
        [Output("stored-commits", "data"),
         Output("status-container", "children"),
         Output("commit-table-container", "children")],
        [Input("btn-fetch-commits", "n_clicks")],
        [State("repo-path-input", "value"),
         State("branch-name-input", "value"),
         State("top-k-input", "value")],
        prevent_initial_call=False
    )
    def fetch_commits(n_clicks, repo_path, branch, top_k):
        """Fetch commits from the specified repository."""
        # Initial state - no clicks yet
        if n_clicks is None or n_clicks == 0:
            return None, "", "Enter a repository path and click 'Fetch & Analyze' to begin."

        # Validate inputs
        if not repo_path:
            return None, dbc.Alert("Please enter a repository path.", color="danger"), ""

        is_valid, error_msg = validate_repo_path(repo_path)
        if not is_valid:
            return None, dbc.Alert(error_msg, color="danger"), ""

        # Fetch commits using GitDataFetcher
        if GitDataFetcher is None:
            return None, dbc.Alert("GitDataFetcher not available", color="warning"), ""

        try:
            fetcher = GitDataFetcher(repo_path, branch, top_k)
            commits, error = fetcher.fetch_commits()

            if error:
                return None, dbc.Alert(f"Error fetching commits: {error}", color="danger"), ""

            if not commits:
                return None, dbc.Alert(f"No commits found in branch '{branch}'", color="warning"), ""

            # Import here to avoid circular dependency
            from frontend.layouts.components import create_commit_table

            table = create_commit_table(commits)

            return (
                commits,
                dbc.Alert(f"Successfully loaded {len(commits)} commits from '{branch}'", color="success"),
                table
            )

        except Exception as e:
            return None, dbc.Alert(f"Unexpected error: {str(e)}", color="danger"), ""
