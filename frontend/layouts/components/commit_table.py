"""Commit table component for displaying Git commit history."""

from dash import dash_table, html
import dash_bootstrap_components as dbc


def create_commit_table(commits_data):
    """
    Create a data table component for displaying commits.

    Args:
        commits_data: List of commit dictionaries with keys:
            - idx: Commit index
            - commit: Commit hash
            - subject: Commit subject/message
            - author: Author name
            - timestamp: Unix timestamp

    Returns:
        dash_table.DataTable: Formatted data table
    """
    if not commits_data:
        return html.Div([
            html.H4("No Commits Found", className="text-muted"),
            html.P("Fetch commits from a repository to see them here.")
        ], className="text-center p-4")

    return dash_table.DataTable(
        id='commit-table',
        columns=[
            {"name": "#", "id": "idx", "type": "numeric", "width": "50px"},
            {"name": "Hash", "id": "commit", "type": "text", "width": "120px"},
            {"name": "Subject", "id": "subject", "type": "text"},
            {"name": "Author", "id": "author", "type": "text", "width": "150px"},
        ],
        data=commits_data,
        style_table={
            'overflowX': 'auto',
            'border': '2px solid #34495e'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'fontFamily': "'Comic Sans MS', 'Chalkboard SE', sans-serif",
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#ecf0f1',
            'fontWeight': 'bold',
            'borderBottom': '3px solid #34495e'
        },
        style_data={
            'backgroundColor': 'white',
            'borderBottom': '1px solid #ddd'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#fafafa'
            }
        ],
        row_selectable='single',
        page_size=25,
        page_action='native',
        sort_action='native',
        sort_mode='multi',
        filter_action='native',
        export_format='xlsx',
        export_headers='display'
    )


def create_commits_tab():
    """
    Create the commits tab container.

    Returns:
        html.Div: Container for the commits tab
    """
    return html.Div([
        html.Div([
            html.H4("Commit History", className="mb-3"),
            html.Div(id="commit-table-container", children=[
                html.P("No repository analyzed yet.", className="text-muted")
            ])
        ])
    ], id="commits-tab-content")


def create_commits_detail_view():
    """
    Create the commit detail view container.

    Returns:
        html.Div: Container for displaying selected commit details
    """
    return html.Div([
        html.H4("Commit Details", className="mb-3"),
        html.Div(
            id="commit-detail-content",
            className="sketch-code p-3",
            style={
                'minHeight': '400px',
                'maxHeight': '700px',
                'overflowY': 'auto'
            },
            children="Select a commit to view details."
        )
    ])
