"""GitTracer - SWE Trajectory Analysis Platform

Entry point for the Dash application.
Uses the modular frontend package.
"""

from frontend.app import create_app as create_frontend_app
from frontend.git_cmd import GitCommit, GitDataFetcher

# =============================================================================
# Application Entry Point
# =============================================================================


def create_app():
    """Create and return the Dash application."""
    return create_frontend_app("GitTracer")


app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=True)
