"""Dash application factory for GitTracer."""

from dash import Dash
import dash_bootstrap_components as dbc
from frontend.layouts.main_layout import create_layout, get_external_stylesheets
from frontend.callbacks import repo_callbacks, trajectory_callbacks


def create_app(name="GitTracer"):
    """Create and configure the Dash application."""
    app = Dash(
        name,
        external_stylesheets=get_external_stylesheets(),
        suppress_callback_exceptions=True,
        title="GitTracer - SWE Trajectory Analysis",
    )

    app.layout = create_layout()

    # Register callbacks
    repo_callbacks.register_callbacks(app)
    trajectory_callbacks.register_callbacks(app)

    return app
