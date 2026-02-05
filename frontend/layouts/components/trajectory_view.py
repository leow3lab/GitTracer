"""Trajectory visualization component for GitTracer."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_trajectory_view(trajectories_data):
    """
    Create a trajectory visualization view.

    Args:
        trajectories_data: List of trajectory dictionaries with keys:
            - id: Trajectory ID
            - type: 'feature' or 'bug_fix'
            - title: Trajectory title
            - description: Brief description
            - commit_count: Number of commits in trajectory
            - commits: List of commit hashes

    Returns:
        html.Div: Container for trajectory visualization
    """
    if not trajectories_data:
        return html.Div([
            html.H4("No Trajectories Found", className="text-center text-muted mt-5"),
            html.P("Analyze a repository to identify feature and bug-fix trajectories.", className="text-center"),
            html.Div(
                html.I(className="bi bi-diagram-3", style={'fontSize': '4rem'}),
                className="text-center mt-4 text-muted"
            )
        ])

    trajectory_cards = []
    for traj in trajectories_data:
        card = create_trajectory_card(traj)
        trajectory_cards.append(card)

    return html.Div([
        html.H4("Identified Trajectories", className="mb-4"),
        html.Div(trajectory_cards, id="trajectory-cards-container")
    ])


def create_trajectory_card(trajectory):
    """
    Create a single trajectory card.

    Args:
        trajectory: Dictionary with trajectory data

    Returns:
        dbc.Card: Card component with trajectory info
    """
    type_badge_color = {
        'feature': 'primary',
        'bug_fix': 'danger',
        'refactor': 'warning',
        'docs': 'info',
        'test': 'success'
    }.get(trajectory.get('type', 'unknown'), 'secondary')

    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span(
                    trajectory.get('type', 'unknown').replace('_', ' ').title(),
                    className=f"badge bg-{type_badge_color} me-2"
                ),
                html.H5(trajectory.get('title', 'Untitled Trajectory'), className="d-inline-block mb-0")
            ])
        ]),
        dbc.CardBody([
            html.P(trajectory.get('description', 'No description available'), className="mb-3"),
            html.Hr(),
            html.Small([
                html.I(className="bi bi-layers me-1"),
                f"{trajectory.get('commit_count', 0)} commits"
            ], className="text-muted"),
            dbc.Button(
                "View Details",
                id=f"btn-trajectory-{trajectory.get('id')}",
                color="outline-primary",
                size="sm",
                className="mt-3 sketch-button"
            )
        ])
    ], className="sketch-card mb-3")


def create_trajectory_timeline(commits):
    """
    Create a timeline view for a trajectory.

    Args:
        commits: List of commit objects in chronological order

    Returns:
        html.Div: Timeline component
    """
    timeline_items = []
    for i, commit in enumerate(commits):
        item = dbc.Card([
            dbc.CardBody([
                html.H6(f"Commit {i+1}: {commit.get('commit', 'unknown')[:8]}", className="mb-1"),
                html.P(commit.get('subject', 'No subject'), className="mb-2 small text-muted"),
                html.Small(f"Author: {commit.get('author', 'Unknown')}")
            ])
        ], className="mb-2")
        timeline_items.append(item)

    return html.Div([
        html.H5("Trajectory Timeline", className="mb-3"),
        html.Div(timeline_items, className="sketch-timeline")
    ])
