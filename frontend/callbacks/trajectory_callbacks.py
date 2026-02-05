"""Trajectory callbacks for view details toggle."""

from dash import Input, Output, State, ALL, html, dcc
from dash.exceptions import PreventUpdate


def _create_markdown_for_commit(commit):
    """Create markdown content for a commit."""
    from datetime import datetime

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


def register_callbacks(app):
    """Register trajectory callbacks."""

    @app.callback(
        Output({"type": "details-content", "index": ALL}, "children"),
        Output({"type": "details-content", "index": ALL}, "style"),
        Input({"type": "btn-details", "index": ALL}, "n_clicks"),
        State("stored-commits", "data"),
        State({"type": "details-content", "index": ALL}, "style"),
        prevent_initial_call=True,
    )
    def toggle_details(n_clicks_list, commits_data, current_styles):
        """Toggle commit details visibility with markdown."""
        if not n_clicks_list or not commits_data:
            raise PreventUpdate

        new_children = []
        new_styles = []

        for i, (n_clicks, style) in enumerate(zip(n_clicks_list, current_styles)):
            if i < len(commits_data):
                commit = commits_data[i]
                if n_clicks and n_clicks % 2 == 1:
                    # Show markdown
                    new_children.append(dcc.Markdown(_create_markdown_for_commit(commit)))
                    new_styles.append({"display": "block"})
                else:
                    # Hide
                    new_children.append(html.Div())
                    new_styles.append({"display": "none"})
            else:
                new_children.append(html.Div())
                new_styles.append({"display": "none"})

        return new_children, new_styles
