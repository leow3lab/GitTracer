"""Trajectory visualization component for GitTracer."""

from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime

_TYPE_COLORS = {
    "feature": "primary",
    "bug_fix": "danger",
    "refactor": "warning",
    "docs": "info",
    "test": "success",
}


def _format_timestamp(ts):
    """Format Unix timestamp to readable string."""
    if not ts:
        return "Unknown"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def create_trajectory_view(commits_data):
    """Create trajectory view - shows commits when no AI analysis yet."""
    if not commits_data:
        return html.Div(
            [
                html.H4("No Data Yet", className="text-center text-muted mt-5"),
                html.P(
                    "Enter a repository path and click 'Fetch & Analyze' to begin.",
                    className="text-center",
                ),
                html.Div(
                    html.I(className="bi bi-diagram-3", style={"fontSize": "4rem"}),
                    className="text-center mt-4 text-muted",
                ),
            ]
        )

    # Show commits as timeline items
    return html.Div(
        [
            html.H4(f"Commits ({len(commits_data)})", className="mb-4"),
            html.Div(
                [create_commit_card(c, i) for i, c in enumerate(commits_data)],
                className="sketch-timeline",
            ),
        ]
    )


def create_commit_card(commit, idx):
    """Create a single commit card with hash, time, and view details button."""
    commit_hash = commit.get("commit", "unknown")
    short_hash = commit_hash[:8]
    timestamp = commit.get("timestamp", 0)
    author = commit.get("author", "Unknown")
    subject = commit.get("subject", "No subject")

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    # Header: Hash + Time
                    html.Div(
                        [
                            html.Span(f"#{idx + 1}", className="badge bg-secondary me-2"),
                            html.Code(short_hash, className="me-2"),
                            html.Small(_format_timestamp(timestamp), className="text-muted"),
                        ],
                        className="mb-2",
                    ),
                    # Subject
                    html.P(subject, className="mb-1", style={"fontWeight": "500"}),
                    # Author
                    html.Small(f"By {author}", className="text-muted"),
                    # View Details button
                    html.Div(
                        [
                            html.Hr(className="my-2"),
                            dbc.Button(
                                "View Details",
                                id={"type": "btn-details", "index": idx},
                                color="outline-primary",
                                size="sm",
                                className="sketch-button",
                            ),
                            html.Div(
                                id={"type": "details-content", "index": idx},
                                className="mt-3 sketch-code p-3",
                                style={"display": "none"},
                            ),
                        ]
                    ),
                ]
            )
        ],
        className="sketch-timeline-item",
    )


def _create_markdown_for_commit(commit):
    """Create markdown content for a commit."""
    dt = datetime.fromtimestamp(commit.get("timestamp", 0))
    return f"""### Commit: {commit.get('commit', 'unknown')[:12]}

**Subject:** {commit.get('subject', 'No subject')}

**Author:** {commit.get('author', 'Unknown')} <{commit.get('email', 'unknown')}>

**Date:** {dt.strftime('%Y-%m-%d %H:%M:%S')}

---

#### Code Changes

```diff
{commit.get('diff', 'No diff available')}
```
"""


def create_trajectory_card(trajectory):
    """Create a single trajectory card (for future AI analysis)."""
    color = _TYPE_COLORS.get(trajectory.get("type", "unknown"), "secondary")
    t = trajectory.get("type", "unknown").replace("_", " ").title()

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span(t, className=f"badge bg-{color} me-2"),
                    html.H5(
                        trajectory.get("title", "Untitled Trajectory"),
                        className="d-inline-block mb-0",
                    ),
                ]
            ),
            dbc.CardBody(
                [
                    html.P(
                        trajectory.get("description", "No description available"), className="mb-3"
                    ),
                    html.Hr(),
                    html.Small(
                        [
                            html.I(className="bi bi-layers me-1"),
                            f"{trajectory.get('commit_count', 0)} commits",
                        ],
                        className="text-muted",
                    ),
                    dbc.Button(
                        "View Details",
                        id=f"btn-trajectory-{trajectory.get('id')}",
                        color="outline-primary",
                        size="sm",
                        className="mt-3 sketch-button",
                    ),
                ]
            ),
        ],
        className="sketch-card mb-3",
    )
