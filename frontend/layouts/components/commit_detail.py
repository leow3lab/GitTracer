"""Commit detail view component for displaying individual commit information."""

from dash import html, dcc
from datetime import datetime
import dash_bootstrap_components as dbc


def _format_timestamp(timestamp):
    """Format Unix timestamp to readable string."""
    return datetime.fromtimestamp(timestamp or 0).strftime("%Y-%m-%d %H:%M:%S")


def create_commit_detail_view(commit_data):
    """Create a detailed view of a single commit."""
    if not commit_data:
        return html.Div("No commit selected.", className="text-muted p-4")

    h = commit_data.get("commit", "unknown")[:8]
    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H4(
                                [html.Span("Commit ", className="text-muted"), html.Code(h)],
                                className="mb-3",
                            ),
                            html.H5(commit_data.get("subject", "No Subject"), className="mb-3"),
                            html.Div(
                                [
                                    html.Strong("Author: "),
                                    html.Span(
                                        f"{commit_data.get('author', 'Unknown')} ",
                                        className="text-muted",
                                    ),
                                    html.Span(
                                        f"<{commit_data.get('email', 'unknown')}>",
                                        className="text-muted",
                                    ),
                                ],
                                className="mb-2",
                            ),
                            html.Div(
                                [
                                    html.Strong("Date: "),
                                    html.Span(
                                        _format_timestamp(commit_data.get("timestamp")),
                                        className="text-muted",
                                    ),
                                ],
                                className="mb-0",
                            ),
                        ]
                    )
                ],
                className="sketch-card mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Code Changes", className="sketch-header"),
                    dbc.CardBody(
                        [
                            html.Pre(
                                commit_data.get("diff", "No code changes in this commit."),
                                className="sketch-code mb-0",
                                style={
                                    "maxHeight": "600px",
                                    "overflow": "auto",
                                    "fontSize": "0.85rem",
                                    "lineHeight": "1.4",
                                },
                            )
                        ]
                    ),
                ],
                className="sketch-card",
            ),
        ]
    )


def create_commit_markdown(commit_data, idx):
    """Generate markdown representation of a commit."""
    return f"""### [{idx}] Commit: {commit_data.get('commit', 'Unknown')}
**Subject:** {commit_data.get('subject', 'No Subject')}
**Author:** {commit_data.get('author', 'Unknown')} (<{commit_data.get('email', 'unknown')}>)
**Date:** {_format_timestamp(commit_data.get('timestamp'))}

---

#### Code Changes
```diff
{commit_data.get("diff", "No code changes in this commit.")}
```"""


def create_commit_detail_with_markdown(commit_data, idx):
    """Create a commit detail view using Markdown rendering."""
    return html.Div(
        [dcc.Markdown(create_commit_markdown(commit_data, idx), dangerously_allow_html=True)],
        className="p-3 bg-light rounded",
    )
