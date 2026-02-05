"""Dash application factory for GitTracer."""

from dash import Dash
import dash_bootstrap_components as dbc
from frontend.layouts.main_layout import create_layout, get_external_stylesheets
from frontend.callbacks import repo_callbacks, navigation_callbacks, trajectory_callbacks


def create_app(name="GitTracer"):
    """
    Create and configure the Dash application.

    Args:
        name: Name of the Dash application

    Returns:
        Dash: Configured Dash application instance
    """
    app = Dash(
        name,
        external_stylesheets=get_external_stylesheets(),
        suppress_callback_exceptions=True,
        title="GitTracer - SWE Trajectory Analysis"
    )

    app.layout = create_layout()

    # Register all callbacks
    repo_callbacks.register_callbacks(app)
    navigation_callbacks.register_callbacks(app)
    trajectory_callbacks.register_callbacks(app)

    return app
