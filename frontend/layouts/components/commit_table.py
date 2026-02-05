"""Commit table component for displaying Git commit history."""

from dash import dash_table, html, dcc
import dash_bootstrap_components as dbc

_TABLE_STYLES = {
    "table": {"overflowX": "auto", "border": "2px solid #34495e"},
    "cell": {
        "textAlign": "left",
        "padding": "12px",
        "fontFamily": "'Comic Sans MS', 'Chalkboard SE', sans-serif",
        "fontSize": "14px",
    },
    "header": {
        "backgroundColor": "#ecf0f1",
        "fontWeight": "bold",
        "borderBottom": "3px solid #34495e",
    },
    "data": {"backgroundColor": "white", "borderBottom": "1px solid #ddd"},
}
_COLUMN_WIDTHS = [
    {"if": {"column_id": "idx"}, "width": "50px"},
    {"if": {"column_id": "commit"}, "width": "120px"},
    {"if": {"column_id": "author"}, "width": "150px"},
]
_COLUMNS = [
    {"name": "#", "id": "idx", "type": "numeric"},
    {"name": "Hash", "id": "commit", "type": "text"},
    {"name": "Subject", "id": "subject", "type": "text"},
    {"name": "Author", "id": "author", "type": "text"},
]


def create_commit_table(commits_data):
    """Create a data table component for displaying commits."""
    if not commits_data:
        return html.Div(
            [
                html.H4("No Commits Found", className="text-muted"),
                html.P("Fetch commits from a repository to see them here."),
            ],
            className="text-center p-4",
        )

    return dash_table.DataTable(
        id="commit-table",
        columns=_COLUMNS,
        data=commits_data,
        style_table=_TABLE_STYLES["table"],
        style_cell=_TABLE_STYLES["cell"],
        style_header=_TABLE_STYLES["header"],
        style_data=_TABLE_STYLES["data"],
        style_cell_conditional=_COLUMN_WIDTHS,
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}],
        row_selectable="single",
        page_size=25,
        page_action="native",
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        export_format="xlsx",
        export_headers="display",
    )


def create_commits_tab():
    """Create the commits tab container with loading wrapper."""
    return html.Div(
        [
            html.H4("Commit History", className="mb-3"),
            dcc.Loading(
                id="loading-commit-table",
                type="default",
                children=[
                    html.Div(
                        id="commit-table-container",
                        children=[html.P("No repository analyzed yet.", className="text-muted")],
                    )
                ],
            ),
        ],
        id="commits-tab-content",
    )


def create_commits_detail_view():
    """Create the commit detail view container."""
    return html.Div(
        [
            html.H4("Commit Details", className="mb-3"),
            html.Div(
                id="commit-detail-content",
                className="sketch-code p-3",
                style={"minHeight": "400px", "maxHeight": "700px", "overflowY": "auto"},
                children="Select a commit to view details.",
            ),
        ],
    )
