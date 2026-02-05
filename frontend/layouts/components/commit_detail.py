"""Commit detail view component for displaying individual commit information."""

from dash import html, dcc
from datetime import datetime
import dash_bootstrap_components as dbc


def create_commit_detail_view(commit_data):
    """
    Create a detailed view of a single commit.

    Args:
        commit_data: Dictionary with commit data including:
            - commit: Full commit hash
            - subject: Commit subject/message
            - author: Author name
            - email: Author email
            - timestamp: Unix timestamp
            - diff: Git diff output

    Returns:
        html.Div: Detailed commit view
    """
    if not commit_data:
        return html.Div("No commit selected.", className="text-muted p-4")

    dt_object = datetime.fromtimestamp(commit_data.get('timestamp', 0))
    formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')

    return html.Div([
        # Commit Header
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.Span("Commit ", className="text-muted"),
                    html.Code(commit_data.get('commit', 'unknown')[:8])
                ], className="mb-3"),
                html.H5(commit_data.get('subject', 'No Subject'), className="mb-3"),
                html.Div([
                    html.Strong("Author: "),
                    html.Span(f"{commit_data.get('author', 'Unknown')} ", className="text-muted"),
                    html.Span(f"<{commit_data.get('email', 'unknown')}>", className="text-muted")
                ], className="mb-2"),
                html.Div([
                    html.Strong("Date: "),
                    html.Span(formatted_time, className="text-muted")
                ], className="mb-0")
            ])
        ], className="sketch-card mb-3"),

        # Code Changes Section
        dbc.Card([
            dbc.CardHeader("Code Changes", className="sketch-header"),
            dbc.CardBody([
                html.Pre(
                    commit_data.get('diff', 'No code changes in this commit.'),
                    className="sketch-code mb-0",
                    style={
                        'maxHeight': '600px',
                        'overflow': 'auto',
                        'fontSize': '0.85rem',
                        'lineHeight': '1.4'
                    }
                )
            ])
        ], className="sketch-card")
    ])


def create_commit_markdown(commit_data, idx):
    """
    Generate markdown representation of a commit.

    Args:
        commit_data: Dictionary with commit data
        idx: Commit index number

    Returns:
        str: Markdown formatted commit
    """
    dt_object = datetime.fromtimestamp(commit_data.get('timestamp', 0))
    formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')

    md_lines = [
        f"### [{idx}] Commit: {commit_data.get('commit', 'Unknown')}",
        f"**Subject:** {commit_data.get('subject', 'No Subject')}  ",
        f"**Author:** {commit_data.get('author', 'Unknown')} (<{commit_data.get('email', 'unknown')}>)  ",
        f"**Date:** {formatted_time}",
        "",
        "---",
        "",
        "#### Code Changes",
        "```diff",
        commit_data.get('diff', 'No code changes in this commit.'),
        "```"
    ]

    return "\n".join(md_lines)


def create_commit_detail_with_markdown(commit_data, idx):
    """
    Create a commit detail view using Markdown rendering.

    Args:
        commit_data: Dictionary with commit data
        idx: Commit index number

    Returns:
        dcc.Markdown: Markdown rendered commit detail
    """
    markdown_content = create_commit_markdown(commit_data, idx)

    return html.Div([
        dcc.Markdown(
            markdown_content,
            dangerously_allow_html=True
        )
    ], className="p-3 bg-light rounded")
