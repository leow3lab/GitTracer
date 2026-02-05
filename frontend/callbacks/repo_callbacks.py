"""Callbacks for repository analysis functionality."""

import os
import re
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc

try:
    from frontend.git_cmd import GitDataFetcher
except ImportError:
    GitDataFetcher = None


def is_git_url(value: str) -> bool:
    """Lightweight check for common Git URL formats."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    return (
        v.startswith(("http://", "https://", "ssh://", "git://", "file://"))
        or re.match(r"^[\w.-]+@[\w.-]+:.*", v) is not None
    )


def validate_repo_path(path):
    """Validate that a repository path exists and is a git repository."""
    if not path:
        return False, "Repository path is required"

    if is_git_url(path):
        return True, None

    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"

    if not os.path.exists(os.path.join(path, ".git")):
        return False, f"Not a Git repository (no .git directory found)"

    return True, None


def register_callbacks(app):
    """Register repository analysis callbacks with the Dash app."""

    @app.callback(
        Output("stored-commits", "data"),
        Output("status-container", "children"),
        Output("trajectory-container", "children"),
        Output("branches-container", "children"),
        Input("btn-fetch-commits", "n_clicks"),
        State("repo-path-input", "value"),
        State("branch-name-input", "value"),
        State("top-k-input", "value"),
        prevent_initial_call=True,
    )
    def fetch_commits(n_clicks, repo_path, branch, top_k):
        """Fetch commits from the specified repository."""
        if not repo_path:
            return None, dbc.Alert("Please enter a repository path.", color="danger"), "", ""

        is_valid, error_msg = validate_repo_path(repo_path)
        if not is_valid:
            return None, dbc.Alert(error_msg, color="danger"), "", ""

        if GitDataFetcher is None:
            return None, dbc.Alert("GitDataFetcher not available", color="warning"), "", ""

        fetcher = None
        try:
            fetcher = GitDataFetcher(repo_path, branch or "main", top_k or 50)
            commits, error = fetcher.fetch_commits()

            if error:
                return None, dbc.Alert(f"Error: {error}", color="danger"), "", ""

            if not commits:
                return (
                    None,
                    dbc.Alert(f"No commits found in branch '{branch}'", color="warning"),
                    "",
                    "",
                )

            # Get branches
            branches = fetcher.get_branches()

            # Create branches display
            branches_html = (
                html.Div(
                    [
                        html.H6("Available Branches:", className="mt-3 mb-2"),
                        html.Div(
                            [
                                html.Span(b, className="badge bg-secondary me-1 mb-1")
                                for b in branches[:20]  # Show first 20
                            ],
                            className="flex-wrap",
                        ),
                        html.Small(f"Total: {len(branches)} branches", className="text-muted"),
                    ]
                )
                if branches
                else ""
            )

            from frontend.layouts.components import create_trajectory_view

            return (
                commits,
                dbc.Alert(
                    f"Successfully loaded {len(commits)} commits from '{fetcher.branch}'",
                    color="success",
                ),
                create_trajectory_view(commits),
                branches_html,
            )
        except Exception as e:
            err_msg = str(e)
            # Add helpful tips for common errors
            if "timed out" in err_msg.lower() or "network" in err_msg.lower():
                err_msg += "\n\n💡 Tip: If you're in China, try setting a proxy:"
                err_msg += "\n   git config --global http.proxy http://127.0.0.1:7890"
                err_msg += "\n   git config --global https.proxy http://127.0.0.1:7890"
            return None, dbc.Alert(err_msg, color="danger"), "", ""
