# Frontend Dash App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-ready Dash frontend with hand-drawn/sketch UI style for GitTracer - a SWE trajectory analysis platform that allows users to fetch repositories, analyze commits, and visualize feature/bug-fix trajectories.

**Architecture:**
- Modular Dash app with separate component files for layouts, callbacks, and styles
- Custom CSS-based hand-drawn UI style (PaperCSS-inspired with custom sketch borders, hand-drawn fonts)
- State management using dcc.Store for commits, trajectories, and analysis results
- Future-proof structure for FastAPI backend integration

**Tech Stack:**
- Dash (Plotly), Dash Bootstrap Components (base structure only, custom styling)
- Pandas for data manipulation
- GitPython for repository operations (existing)
- Custom CSS for hand-drawn/sketch aesthetic

---

## Project Structure

```
GitTracer/
├── app.py                          # Main entry point
├── frontend/
│   ├── __init__.py
│   ├── app.py                      # Dash app factory
│   ├── layouts/
│   │   ├── __init__.py
│   │   ├── main_layout.py          # Root layout
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── repo_config.py      # Repository configuration card
│   │   │   ├── commit_table.py     # Commits list view
│   │   │   ├── trajectory_view.py  # Trajectory visualization
│   │   │   └── commit_detail.py    # Single commit detail view
│   ├── callbacks/
│   │   ├── __init__.py
│   │   ├── repo_callbacks.py       # Repository analysis callbacks
│   │   ├── navigation_callbacks.py # Tab/routing callbacks
│   │   └── trajectory_callbacks.py # Trajectory display callbacks
│   └── styles/
│       ├── __init__.py
│       └── sketch_style.css        # Hand-drawn UI styles
└── tests/
    └── frontend/
        ├── test_components.py
        └── test_callbacks.py
```

---

## Task 1: Project Structure Setup

**Files:**
- Create: `frontend/__init__.py`
- Create: `frontend/layouts/__init__.py`
- Create: `frontend/layouts/components/__init__.py`
- Create: `frontend/callbacks/__init__.py`
- Create: `frontend/styles/__init__.py`
- Create: `tests/frontend/__init__.py`
- Create: `tests/frontend/test_components.py`

**Step 1: Create frontend package structure**

```bash
# Create all directories
mkdir -p frontend/layouts/components
mkdir -p frontend/callbacks
mkdir -p frontend/styles
mkdir -p tests/frontend

# Create __init__.py files
touch frontend/__init__.py
touch frontend/layouts/__init__.py
touch frontend/layouts/components/__init__.py
touch frontend/callbacks/__init__.py
touch frontend/styles/__init__.py
touch tests/frontend/__init__.py
```

**Step 2: Verify directory structure**

Run: `tree frontend -I '__pycache__'` or `find frontend -type f -name '*.py'`
Expected: All __init__.py files created in proper directories

**Step 3: Write initial component test**

Create `tests/frontend/test_components.py`:

```python
import pytest

def test_frontend_package_exists():
    """Verify frontend package structure is created"""
    from frontend import layouts, callbacks, styles
    assert hasattr(layouts, 'components')
    assert hasattr(callbacks, '__file__')
    assert hasattr(styles, '__file__')
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_components.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/ tests/frontend/
git commit -m "feat: set up frontend package structure"
```

---

## Task 2: Hand-Drawn CSS Style Sheet

**Files:**
- Create: `frontend/styles/sketch_style.css`
- Modify: `frontend/styles/__init__.py`

**Step 1: Write the failing test**

Create `tests/frontend/test_styles.py`:

```python
import os
import pytest

CSS_PATH = "frontend/styles/sketch_style.css"

def test_sketch_css_exists():
    """Verify sketch style CSS file exists"""
    assert os.path.exists(CSS_PATH), "sketch_style.css not found"

def test_sketch_css_has_hand_drawn_border_class():
    """Verify CSS contains hand-drawn border class"""
    with open(CSS_PATH) as f:
        content = f.read()
    assert ".sketch-border" in content or ".hand-drawn" in content

def test_sketch_css_has_paper_background():
    """Verify CSS has paper-like background"""
    with open(CSS_PATH) as f:
        content = f.read()
    assert any(x in content for x in ["#f9f7f1", "#faf8f5", "paper", "parchment"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_styles.py -v`
Expected: FAIL with "sketch_style.css not found"

**Step 3: Create sketch style CSS**

Create `frontend/styles/sketch_style.css`:

```css
/* GitTracer Hand-Drawn Sketch Style */

:root {
    --paper-bg: #faf8f5;
    --ink-color: #2c3e50;
    --pencil-gray: #7f8c8d;
    --sketch-border: #34495e;
    --highlight-blue: #3498db;
    --highlight-green: #27ae60;
    --highlight-red: #e74c3c;
    --highlight-yellow: #f39c12;
}

body {
    background-color: var(--paper-bg);
    font-family: 'Comic Sans MS', 'Chalkboard SE', 'Marker Felt', sans-serif;
    color: var(--ink-color);
}

/* Hand-drawn card borders */
.sketch-card {
    background: white;
    border: 2px solid var(--sketch-border);
    border-radius: 2px;
    box-shadow: 3px 3px 0 rgba(0,0,0,0.1);
    position: relative;
    margin-bottom: 1.5rem;
}

.sketch-card::before {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    right: -1px;
    bottom: -1px;
    border: 1px dashed rgba(0,0,0,0.2);
    border-radius: 3px;
    pointer-events: none;
}

/* Sketch header style */
.sketch-header {
    border-bottom: 2px solid var(--sketch-border);
    padding: 1rem;
    font-weight: bold;
    font-size: 1.2rem;
    position: relative;
}

.sketch-header::after {
    content: '';
    position: absolute;
    bottom: -3px;
    left: 0;
    width: 30%;
    height: 3px;
    background: var(--highlight-blue);
}

/* Hand-drawn button */
.sketch-button {
    background: white;
    border: 2px solid var(--sketch-border);
    padding: 0.5rem 1.5rem;
    cursor: pointer;
    font-family: inherit;
    font-weight: bold;
    transition: all 0.1s;
    position: relative;
}

.sketch-button:hover {
    background: var(--highlight-blue);
    color: white;
    transform: translate(-1px, -1px);
    box-shadow: 2px 2px 0 rgba(0,0,0,0.2);
}

.sketch-button:active {
    transform: translate(1px, 1px);
    box-shadow: none;
}

/* Sketch input fields */
.sketch-input {
    border: 2px solid var(--pencil-gray);
    border-radius: 0;
    padding: 0.5rem;
    background: #fffefb;
    font-family: inherit;
}

.sketch-input:focus {
    outline: none;
    border-color: var(--highlight-blue);
    border-style: dashed;
}

/* Sketch table */
.sketch-table {
    border: 2px solid var(--sketch-border);
    background: white;
}

.sketch-table thead {
    border-bottom: 3px solid var(--sketch-border);
}

.sketch-table th {
    padding: 0.75rem;
    text-align: left;
    font-weight: bold;
}

.sketch-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #ddd;
}

.sketch-table tr:hover td {
    background: rgba(52, 152, 219, 0.1);
}

/* Status alerts */
.sketch-alert {
    border-left: 4px solid var(--sketch-border);
    padding: 1rem;
    margin: 1rem 0;
    background: white;
}

.sketch-alert.success {
    border-left-color: var(--highlight-green);
}

.sketch-alert.error {
    border-left-color: var(--highlight-red);
}

.sketch-alert.warning {
    border-left-color: var(--highlight-yellow);
}

/* Tab styling */
.sketch-tabs {
    border-bottom: 2px solid var(--sketch-border);
    margin-bottom: 1rem;
}

.sketch-tab {
    padding: 0.5rem 1.5rem;
    border: 2px solid transparent;
    border-bottom: none;
    cursor: pointer;
    background: transparent;
    font-family: inherit;
    font-weight: bold;
}

.sketch-tab.active {
    border-color: var(--sketch-border);
    border-bottom-color: white;
    background: white;
    margin-bottom: -2px;
}

/* Loading sketch animation */
.sketch-loading {
    display: inline-block;
    animation: sketch-pulse 1.5s ease-in-out infinite;
}

@keyframes sketch-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Code diff styling */
.sketch-code {
    background: #f8f8f8;
    border: 1px dashed var(--pencil-gray);
    border-left: 3px solid var(--sketch-border);
    padding: 1rem;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 0.9rem;
    overflow-x: auto;
}

/* Trajectory timeline */
.sketch-timeline {
    position: relative;
    padding-left: 2rem;
}

.sketch-timeline::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: repeating-linear-gradient(
        to bottom,
        var(--sketch-border) 0px,
        var(--sketch-border) 10px,
        transparent 10px,
        transparent 15px
    );
}

.sketch-timeline-item {
    position: relative;
    padding: 1rem;
    margin-bottom: 1rem;
    background: white;
    border: 2px solid var(--sketch-border);
}

.sketch-timeline-item::before {
    content: '';
    position: absolute;
    left: -2.4rem;
    top: 1rem;
    width: 12px;
    height: 12px;
    background: var(--highlight-blue);
    border: 2px solid var(--sketch-border);
    border-radius: 50%;
}
```

**Step 4: Update styles __init__.py**

Modify `frontend/styles/__init__.py`:

```python
"""Sketch style hand-drawn UI styles for GitTracer."""

CSS_PATH = "frontend/styles/sketch_style.css"

def get_stylesheet():
    """Return the external stylesheet path for Dash app."""
    return CSS_PATH
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/frontend/test_styles.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add frontend/styles/ tests/frontend/test_styles.py
git commit -m "feat: add hand-drawn sketch CSS styles"
```

---

## Task 3: Main Layout Factory

**Files:**
- Create: `frontend/app.py`
- Create: `frontend/layouts/main_layout.py`

**Step 1: Write the failing test**

Add to `tests/frontend/test_components.py`:

```python
def test_main_layout_factory_exists():
    """Verify main layout can be imported"""
    from frontend.layouts.main_layout import create_layout
    layout = create_layout()
    assert layout is not None
    assert hasattr(layout, 'children')

def test_dash_app_factory():
    """Verify Dash app factory function"""
    from frontend.app import create_app
    app = create_app()
    assert app is not None
    assert hasattr(app, 'layout')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_components.py::test_main_layout_factory_exists -v`
Expected: FAIL with import error

**Step 3: Create main layout module**

Create `frontend/layouts/main_layout.py`:

```python
"""Main layout for GitTracer Dash application."""

from dash import dcc, html
import dash_bootstrap_components as dbc
from frontend.styles import get_stylesheet


def create_layout():
    """
    Create the main layout for GitTracer app.

    Returns:
        dbc.Container: The root layout container
    """
    return dbc.Container([
        dcc.Store(id='stored-commits'),
        dcc.Store(id='stored-trajectories'),
        dcc.Store(id='selected-commit'),

        # Header Section
        dbc.Row([
            dbc.Col([
                html.H1(
                    "GitTracer",
                    className="sketch-header text-center",
                    style={'fontSize': '2.5rem', 'marginBottom': '0.5rem'}
                ),
                html.P(
                    "SWE Trajectory Analysis Platform",
                    className="text-center text-muted",
                    style={'fontSize': '1.1rem'}
                )
            ], width=12)
        ], className="mb-4"),

        # Main Content Area
        dbc.Row([
            # Left Column - Configuration
            dbc.Col([
                html.Div(id="repo-config-container")
            ], width=4, lg=4),

            # Right Column - Tabs and Content
            dbc.Col([
                html.Div(id="main-content-area")
            ], width=8, lg=8)
        ]),

        # Status Messages
        dbc.Row([
            dbc.Col([
                html.Div(id="status-container")
            ], width=12)
        ])
    ], fluid=True, className="py-4")


def get_external_stylesheets():
    """
    Get external stylesheets for the Dash app.

    Returns:
        list: List of stylesheet paths/URLs
    """
    # Use bootstrap as base, will override with custom CSS
    return [dbc.themes.FLATLY, get_stylesheet()]
```

**Step 4: Create Dash app factory**

Create `frontend/app.py`:

```python
"""Dash application factory for GitTracer."""

from dash import Dash
import dash_bootstrap_components as dbc
from frontend.layouts.main_layout import create_layout, get_external_stylesheets


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

    return app
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/frontend/test_components.py::test_main_layout_factory_exists tests/frontend/test_components.py::test_dash_app_factory -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add frontend/app.py frontend/layouts/main_layout.py
git commit -m "feat: create main layout and app factory"
```

---

## Task 4: Repository Configuration Component

**Files:**
- Create: `frontend/layouts/components/repo_config.py`

**Step 1: Write the failing test**

Add to `tests/frontend/test_components.py`:

```python
def test_repo_config_component():
    """Verify repository configuration component"""
    from frontend.layouts.components.repo_config import create_repo_config
    component = create_repo_config()
    assert component is not None
    # Should have input fields for repo path, branch, and top_k
    assert hasattr(component, 'children')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_components.py::test_repo_config_component -v`
Expected: FAIL with import error

**Step 3: Create repo config component**

Create `frontend/layouts/components/repo_config.py`:

```python
"""Repository configuration component for GitTracer."""

import os
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_repo_config():
    """
    Create the repository configuration card component.

    Returns:
        dbc.Card: Configuration card with repo path, branch, and top_k inputs
    """
    return dbc.Card([
        dbc.CardHeader(
            html.H4("Repository Configuration", className="sketch-header mb-0"),
            className="sketch-card-header"
        ),
        dbc.CardBody([
            # Repository Path Input
            html.Label(
                "Local Repository Path:",
                className="fw-bold mb-2",
                style={'fontFamily': 'inherit'}
            ),
            dbc.Input(
                id="repo-path-input",
                placeholder="e.g., /path/to/your/repo",
                type="text",
                value=os.getcwd(),
                className="sketch-input mb-3"
            ),

            # Branch Name Input
            html.Label(
                "Branch Name:",
                className="fw-bold mb-2",
                style={'fontFamily': 'inherit'}
            ),
            dbc.Input(
                id="branch-name-input",
                value="main",
                type="text",
                className="sketch-input mb-3"
            ),

            # Top K Commits Input
            html.Label(
                "Number of Commits to Analyze:",
                className="fw-bold mb-2",
                style={'fontFamily': 'inherit'}
            ),
            dbc.Input(
                id="top-k-input",
                value=50,
                type="number",
                min=1,
                max=1000,
                className="sketch-input mb-3"
            ),

            # Analyze Button
            dbc.Button(
                html.Div([
                    html.I(className="bi bi-github me-2"),
                    "Fetch & Analyze"
                ]),
                id="btn-fetch-commits",
                color="primary",
                size="lg",
                className="sketch-button w-100"
            ),

            # Help Text
            html.Small(
                "Enter a local Git repository path to fetch commit history.",
                className="text-muted mt-3 d-block",
                style={'fontStyle': 'italic'}
            )
        ])
    ], className="sketch-card")


def create_status_alert(message, alert_type="info"):
    """
    Create a status alert component.

    Args:
        message: The message to display
        alert_type: One of 'success', 'error', 'warning', 'info'

    Returns:
        dbc.Alert: Alert component with sketch styling
    """
    color_map = {
        'success': 'success',
        'error': 'danger',
        'warning': 'warning',
        'info': 'info'
    }

    return dbc.Alert(
        message,
        id="status-alert",
        color=color_map.get(alert_type, 'info'),
        className="sketch-alert",
        dismissable=True,
        is_open=True
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_components.py::test_repo_config_component -v`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/layouts/components/repo_config.py
git commit -m "feat: add repository configuration component"
```

---

## Task 5: Commit Table Component

**Files:**
- Create: `frontend/layouts/components/commit_table.py`

**Step 1: Write the failing test**

Add to `tests/frontend/test_components.py`:

```python
def test_commit_table_component():
    """Verify commit table component"""
    from frontend.layouts.components.commit_table import create_commit_table
    sample_data = [
        {'idx': 1, 'commit': 'abc123', 'subject': 'Test commit', 'author': 'Test Author'}
    ]
    table = create_commit_table(sample_data)
    assert table is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_components.py::test_commit_table_component -v`
Expected: FAIL with import error

**Step 3: Create commit table component**

Create `frontend/layouts/components/commit_table.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_components.py::test_commit_table_component -v`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/layouts/components/commit_table.py
git commit -m "feat: add commit table component"
```

---

## Task 6: Trajectory View Component

**Files:**
- Create: `frontend/layouts/components/trajectory_view.py`

**Step 1: Write the failing test**

Add to `tests/frontend/test_components.py`:

```python
def test_trajectory_view_component():
    """Verify trajectory view component"""
    from frontend.layouts.components.trajectory_view import create_trajectory_view
    view = create_trajectory_view([])
    assert view is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_components.py::test_trajectory_view_component -v`
Expected: FAIL with import error

**Step 3: Create trajectory view component**

Create `frontend/layouts/components/trajectory_view.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_components.py::test_trajectory_view_component -v`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/layouts/components/trajectory_view.py
git commit -m "feat: add trajectory view component"
```

---

## Task 7: Commit Detail Component

**Files:**
- Create: `frontend/layouts/components/commit_detail.py`

**Step 1: Write the failing test**

Add to `tests/frontend/test_components.py`:

```python
def test_commit_detail_component():
    """Verify commit detail component"""
    from frontend.layouts.components.commit_detail import create_commit_detail_view
    detail = create_commit_detail_view({
        'commit': 'abc123',
        'subject': 'Test',
        'author': 'Test Author',
        'diff': 'sample diff'
    })
    assert detail is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_components.py::test_commit_detail_component -v`
Expected: FAIL with import error

**Step 3: Create commit detail component**

Create `frontend/layouts/components/commit_detail.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_components.py::test_commit_detail_component -v`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/layouts/components/commit_detail.py
git commit -m "feat: add commit detail component"
```

---

## Task 8: Layout Components Module Export

**Files:**
- Modify: `frontend/layouts/components/__init__.py`

**Step 1: Write the failing test**

Add to `tests/frontend/test_components.py`:

```python
def test_layout_components_module_exports():
    """Verify all components can be imported from module"""
    from frontend.layouts.components import (
        create_repo_config,
        create_commit_table,
        create_trajectory_view,
        create_commit_detail_view
    )
    # All should be callable
    assert callable(create_repo_config)
    assert callable(create_commit_table)
    assert callable(create_trajectory_view)
    assert callable(create_commit_detail_view)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_components.py::test_layout_components_module_exports -v`
Expected: FAIL with import error (exports not defined)

**Step 3: Create module exports**

Modify `frontend/layouts/components/__init__.py`:

```python
"""Layout components for GitTracer frontend."""

from frontend.layouts.components.repo_config import (
    create_repo_config,
    create_status_alert
)
from frontend.layouts.components.commit_table import (
    create_commit_table,
    create_commits_tab,
    create_commits_detail_view
)
from frontend.layouts.components.trajectory_view import (
    create_trajectory_view,
    create_trajectory_card,
    create_trajectory_timeline
)
from frontend.layouts.components.commit_detail import (
    create_commit_detail_view,
    create_commit_markdown,
    create_commit_detail_with_markdown
)

__all__ = [
    # Repo config
    'create_repo_config',
    'create_status_alert',
    # Commit table
    'create_commit_table',
    'create_commits_tab',
    'create_commits_detail_view',
    # Trajectory view
    'create_trajectory_view',
    'create_trajectory_card',
    'create_trajectory_timeline',
    # Commit detail
    'create_commit_detail_view',
    'create_commit_markdown',
    'create_commit_detail_with_markdown'
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/frontend/test_components.py::test_layout_components_module_exports -v`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/layouts/components/__init__.py
git commit -m "feat: export all layout components from module"
```

---

## Task 9: Repository Analysis Callbacks

**Files:**
- Create: `frontend/callbacks/repo_callbacks.py`

**Step 1: Write the failing test**

Create `tests/frontend/test_callbacks.py`:

```python
import pytest
from unittest.mock import Mock, patch

def test_repo_callbacks_module_exists():
    """Verify repo callbacks module exists and has required functions"""
    from frontend.callbacks.repo_callbacks import register_callbacks
    assert callable(register_callbacks)

@pytest.mark.parametrize("repo_path,exists,expected", [
    ("/fake/path", False, "not found"),
    ("/tmp", True, None)  # Assuming /tmp exists
])
def test_validate_repo_path(repo_path, exists, expected):
    """Test repository path validation logic"""
    from frontend.callbacks.repo_callbacks import validate_repo_path
    result = validate_repo_path(repo_path)
    if not exists:
        assert "not found" in result.lower() or "does not exist" in result.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_callbacks.py -v`
Expected: FAIL with import error

**Step 3: Create repository callbacks**

Create `frontend/callbacks/repo_callbacks.py`:

```python
"""Callbacks for repository analysis functionality."""

import os
from dash import Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd

# Import existing GitDataFetcher from root app.py
# This will be refactored later to a shared module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
try:
    from app import GitDataFetcher
except ImportError:
    # Fallback for testing
    GitDataFetcher = None


def validate_repo_path(path):
    """
    Validate that a repository path exists and is a git repository.

    Args:
        path: Path to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    if not path:
        return False, "Repository path is required"

    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"

    git_dir = os.path.join(path, '.git')
    if not os.path.exists(git_dir):
        return False, f"Not a Git repository (no .git directory found)"

    return True, None


def register_callbacks(app):
    """
    Register repository analysis callbacks with the Dash app.

    Args:
        app: Dash application instance
    """
    @app.callback(
        [Output("stored-commits", "data"),
         Output("status-container", "children"),
         Output("commit-table-container", "children")],
        [Input("btn-fetch-commits", "n_clicks")],
        [State("repo-path-input", "value"),
         State("branch-name-input", "value"),
         State("top-k-input", "value")],
        prevent_initial_call=False
    )
    def fetch_commits(n_clicks, repo_path, branch, top_k):
        """Fetch commits from the specified repository."""
        # Initial state - no clicks yet
        if n_clicks is None or n_clicks == 0:
            return None, "", "Enter a repository path and click 'Fetch & Analyze' to begin."

        # Validate inputs
        if not repo_path:
            return None, dbc.Alert("Please enter a repository path.", color="danger"), ""

        is_valid, error_msg = validate_repo_path(repo_path)
        if not is_valid:
            return None, dbc.Alert(error_msg, color="danger"), ""

        # Fetch commits using GitDataFetcher
        if GitDataFetcher is None:
            return None, dbc.Alert("GitDataFetcher not available", color="warning"), ""

        try:
            fetcher = GitDataFetcher(repo_path, branch, top_k)
            commits, error = fetcher.fetch_commits()

            if error:
                return None, dbc.Alert(f"Error fetching commits: {error}", color="danger"), ""

            if not commits:
                return None, dbc.Alert(f"No commits found in branch '{branch}'", color="warning"), ""

            # Import here to avoid circular dependency
            from frontend.layouts.components import create_commit_table

            table = create_commit_table(commits)

            return (
                commits,
                dbc.Alert(f"Successfully loaded {len(commits)} commits from '{branch}'", color="success"),
                table
            )

        except Exception as e:
            return None, dbc.Alert(f"Unexpected error: {str(e)}", color="danger"), ""
```

**Step 4: Update main app to include callbacks**

Modify `frontend/app.py`:

```python
"""Dash application factory for GitTracer."""

from dash import Dash
import dash_bootstrap_components as dbc
from frontend.layouts.main_layout import create_layout, get_external_stylesheets
from frontend.callbacks import repo_callbacks


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

    return app
```

**Step 5: Update main layout with repo config**

Modify `frontend/layouts/main_layout.py`:

```python
"""Main layout for GitTracer Dash application."""

from dash import dcc, html
import dash_bootstrap_components as dbc
from frontend.styles import get_stylesheet
from frontend.layouts.components import create_repo_config


def create_layout():
    """
    Create the main layout for GitTracer app.

    Returns:
        dbc.Container: The root layout container
    """
    return dbc.Container([
        dcc.Store(id='stored-commits'),
        dcc.Store(id='stored-trajectories'),
        dcc.Store(id='selected-commit'),

        # Header Section
        dbc.Row([
            dbc.Col([
                html.H1(
                    "GitTracer",
                    className="text-center mb-2",
                    style={'fontSize': '2.5rem', 'fontWeight': 'bold'}
                ),
                html.P(
                    "SWE Trajectory Analysis Platform",
                    className="text-center text-muted",
                    style={'fontSize': '1.1rem'}
                )
            ], width=12)
        ], className="mb-4"),

        # Main Content Area
        dbc.Row([
            # Left Column - Configuration
            dbc.Col([
                create_repo_config()
            ], width=4, lg=4),

            # Right Column - Tabs and Content
            dbc.Col([
                html.Div(id="main-content-area")
            ], width=8, lg=8)
        ]),

        # Status Messages
        dbc.Row([
            dbc.Col([
                html.Div(id="status-container")
            ], width=12)
        ])
    ], fluid=True, className="py-4")


def get_external_stylesheets():
    """
    Get external stylesheets for the Dash app.

    Returns:
        list: List of stylesheet paths/URLs
    """
    return [dbc.themes.FLATLY, get_stylesheet()]
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/frontend/test_callbacks.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add frontend/callbacks/repo_callbacks.py frontend/app.py frontend/layouts/main_layout.py tests/frontend/test_callbacks.py
git commit -m "feat: add repository analysis callbacks"
```

---

## Task 10: Navigation and Tab Callbacks

**Files:**
- Create: `frontend/callbacks/navigation_callbacks.py`

**Step 1: Write the failing test**

Add to `tests/frontend/test_callbacks.py`:

```python
def test_navigation_callbacks_module_exists():
    """Verify navigation callbacks module exists"""
    from frontend.callbacks.navigation_callbacks import register_callbacks
    assert callable(register_callbacks)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_callbacks.py::test_navigation_callbacks_module_exists -v`
Expected: FAIL with import error

**Step 3: Create navigation callbacks**

Create `frontend/callbacks/navigation_callbacks.py`:

```python
"""Callbacks for navigation and tab switching."""

from dash import Input, Output, State
import dash_bootstrap_components as dbc
from frontend.layouts.components import (
    create_commits_tab,
    create_commits_detail_view,
    create_trajectory_view
)


def register_callbacks(app):
    """
    Register navigation and tab callbacks with the Dash app.

    Args:
        app: Dash application instance
    """
    @app.callback(
        Output("main-content-area", "children"),
        [Input("stored-commits", "data")],
        prevent_initial_call=False
    )
    def update_main_content(commits_data):
        """Update the main content area based on available data."""
        # Create tabs with content
        tabs = dbc.Tabs([
            dbc.Tab(
                label="Commits",
                tab_id="tab-commits",
                children=create_commits_tab()
            ),
            dbc.Tab(
                label="Commit Details",
                tab_id="tab-details",
                children=create_commits_detail_view()
            ),
            dbc.Tab(
                label="Trajectories",
                tab_id="tab-trajectories",
                children=create_trajectory_view(commits_data or [])
            ),
        ], id="main-tabs", active_tab="tab-commits", className="sketch-tabs")

        return tabs
```

**Step 4: Update app factory to register navigation callbacks**

Modify `frontend/app.py`:

```python
"""Dash application factory for GitTracer."""

from dash import Dash
import dash_bootstrap_components as dbc
from frontend.layouts.main_layout import create_layout, get_external_stylesheets
from frontend.callbacks import repo_callbacks, navigation_callbacks


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

    return app
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/frontend/test_callbacks.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add frontend/callbacks/navigation_callbacks.py frontend/app.py
git commit -m "feat: add navigation and tab callbacks"
```

---

## Task 11: Trajectory Display Callbacks

**Files:**
- Create: `frontend/callbacks/trajectory_callbacks.py`

**Step 1: Write the failing test**

Add to `tests/frontend/test_callbacks.py`:

```python
def test_trajectory_callbacks_module_exists():
    """Verify trajectory callbacks module exists"""
    from frontend.callbacks.trajectory_callbacks import register_callbacks
    assert callable(register_callbacks)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/frontend/test_callbacks.py::test_trajectory_callbacks_module_exists -v`
Expected: FAIL with import error

**Step 3: Create trajectory callbacks**

Create `frontend/callbacks/trajectory_callbacks.py`:

```python
"""Callbacks for trajectory display and interaction."""

from dash import Input, Output, State
import dash_bootstrap_components as dbc
from frontend.layouts.components import create_commit_detail_with_markdown


def register_callbacks(app):
    """
    Register trajectory display callbacks with the Dash app.

    Args:
        app: Dash application instance
    """
    @app.callback(
        [Output("commit-detail-content", "children"),
         Output("selected-commit", "data")],
        [Input("commit-table", "derived_virtual_selected_rows")],
        [State("stored-commits", "data")],
        prevent_initial_call=True
    )
    def display_commit_detail(selected_rows, stored_commits):
        """Display detailed view of selected commit."""
        if not selected_rows or not stored_commits:
            return "Select a commit to view details.", None

        row_idx = selected_rows[0]

        # Handle pagination offset
        if row_idx >= len(stored_commits):
            return "Invalid commit selection.", None

        commit_data = stored_commits[row_idx]

        detail_view = create_commit_detail_with_markdown(
            commit_data,
            idx=commit_data.get('idx', row_idx + 1)
        )

        return detail_view, commit_data

    @app.callback(
        Output("status-alert", "is_open"),
        [Input("status-alert", "n_dismisses")],
        prevent_initial_call=False
    )
    def dismiss_alert(n_dismisses):
        """Allow dismissing the status alert."""
        if n_dismisses:
            return False
        return True
```

**Step 4: Update app factory to register trajectory callbacks**

Modify `frontend/app.py`:

```python
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
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/frontend/test_callbacks.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add frontend/callbacks/trajectory_callbacks.py frontend/app.py
git commit -m "feat: add trajectory display callbacks"
```

---

## Task 12: Update Root App.py Entry Point

**Files:**
- Modify: `app.py` (root level)

**Step 1: Write the failing test**

Create `tests/test_app_entry.py`:

```python
import pytest

def test_app_entry_point_runs():
    """Verify the app can be imported without errors"""
    # Import should not raise any errors
    import app
    assert hasattr(app, 'app') or hasattr(app, 'server')

def test_frontend_app_factory():
    """Verify frontend app factory works"""
    from frontend.app import create_app
    test_app = create_app("TestGitTracer")
    assert test_app is not None
    assert hasattr(test_app, 'server')
```

**Step 2: Run test to verify current state**

Run: `pytest tests/test_app_entry.py -v`
Expected: May PASS or FAIL depending on current state

**Step 3: Update root app.py to use new frontend**

Modify `app.py`:

```python
"""
GitTracer - SWE Trajectory Analysis Platform

Entry point for the Dash application.
Uses the modular frontend package.
"""

import os
import subprocess
from datetime import datetime


# =============================================================================
# Data Models (Core Logic)
# =============================================================================

class GitCommit:
    """Data model representing a single Git commit."""

    def __init__(self, data, idx):
        self.idx = idx
        self.commit_id = data.get("commit", "Unknown")
        self.author = data.get("author", "Unknown")
        self.email = data.get("email", "Unknown")
        self.timestamp = data.get("timestamp", 0)
        self.subject = data.get("subject", "No Subject")
        self.diff = data.get("diff", "")

    def to_md(self):
        """Convert commit to Markdown format."""
        dt_object = datetime.fromtimestamp(self.timestamp)
        formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        md_body = [
            f"### [{self.idx}] Commit: {self.commit_id}",
            f"Subject: {self.subject}  ",
            f"Author: {self.author} (<{self.email}>)  ",
            f"Date: {formatted_time}",
            "",
            "---",
            "",
            "#### Code Changes",
            "```diff",
            self.diff if self.diff.strip() else "No code changes in this commit.",
            "```"
        ]
        return "\n".join(md_body)


class GitDataFetcher:
    """Handles Git repository interaction using subprocess."""

    def __init__(self, repo_path, branch="main", top_k=100):
        self.repo_path = os.path.abspath(repo_path)
        self.branch = branch
        self.top_k = top_k

    def _run_git(self, args):
        """Run a git command and return output."""
        return subprocess.check_output(
            ["git"] + args,
            cwd=self.repo_path,
            encoding='utf-8',
            errors='ignore'
        )

    def fetch_commits(self):
        """
        Fetch commit history from the repository.

        Returns:
            tuple: (list of commit dicts, error message or None)
        """
        log_format = "%H|%an|%ae|%at|%s"
        try:
            log_output = self._run_git([
                "log", self.branch, f"--pretty=format:{log_format}", f"-n", str(self.top_k)
            ])
        except Exception as e:
            return None, str(e)

        commits_list = []
        lines = [l for l in log_output.split('\n') if l]

        for i, line in enumerate(lines):
            current_idx = i + 1
            parts = line.split('|')
            if len(parts) < 5:
                continue
            h, name, email, time_val, subj = parts[:5]

            # Get diff content
            try:
                diff_content = self._run_git(["show", "--patch", "-m", "--pretty=format:", h])
            except Exception:
                diff_content = ""

            commits_list.append({
                "idx": current_idx,
                "commit": h,
                "author": name,
                "email": email,
                "timestamp": int(time_val),
                "subject": subj,
                "diff": diff_content
            })

        return commits_list, None


# =============================================================================
# Application Entry Point
# =============================================================================

def create_app():
    """Create and return the Dash application."""
    from frontend.app import create_app as create_frontend_app
    return create_frontend_app("GitTracer")


# Create app instance for development
app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=True)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app_entry.py -v`
Expected: All PASS

**Step 5: Verify app can start manually**

Run: `python app.py` (should start Dash server)
Expected: Server starts on http://127.0.0.1:8050

Press Ctrl+C to stop the server.

**Step 6: Commit**

```bash
git add app.py tests/test_app_entry.py
git commit -m "feat: update root app.py to use modular frontend"
```

---

## Task 13: Integration Test - Full App

**Files:**
- Create: `tests/frontend/test_integration.py`

**Step 1: Write the failing test**

Create `tests/frontend/test_integration.py`:

```python
"""Integration tests for the full frontend application."""

import pytest
from unittest.mock import Mock, patch


def test_full_app_creation():
    """Verify full app can be created with all components."""
    from frontend.app import create_app

    app = create_app("TestApp")

    # Verify app structure
    assert app is not None
    assert hasattr(app, 'layout')
    assert hasattr(app, 'callback_map')

    # Check that callbacks are registered
    # At minimum: repo callbacks should be registered
    callback_ids = list(app.callback_map.keys())
    assert len(callback_ids) > 0


def test_layout_structure():
    """Verify layout has expected structure."""
    from frontend.app import create_app

    app = create_app("TestApp")
    layout = app.layout

    # Layout should be a Container with children
    assert hasattr(layout, 'children')
    assert len(layout.children) > 0


@patch('frontend.callbacks.repo_callbacks.GitDataFetcher')
def test_repo_fetch_flow(mock_fetcher_class):
    """Test the repository fetch callback flow."""
    from frontend.app import create_app
    from frontend.callbacks.repo_callbacks import validate_repo_path

    # Test validation
    is_valid, error = validate_repo_path("/fake/nonexistent/path")
    assert not is_valid
    assert error is not None

    # Test with a real path (tmp should exist on most systems)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        is_valid, error = validate_repo_path(tmpdir)
        # Not a git repo, so should fail
        assert not is_valid
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/frontend/test_integration.py -v`
Expected: All PASS

**Step 3: Run all frontend tests**

Run: `pytest tests/frontend/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/frontend/test_integration.py
git commit -m "test: add integration tests for full app"
```

---

## Task 14: Requirements File

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

**Step 1: Write the failing test**

Create `tests/test_requirements.py`:

```python
import os
import pytest

def test_requirements_files_exist():
    """Verify requirements files exist"""
    assert os.path.exists("requirements.txt"), "requirements.txt not found"
    assert os.path.exists("requirements-dev.txt"), "requirements-dev.txt not found"

def test_requirements_have_dash():
    """Verify Dash is in requirements"""
    with open("requirements.txt") as f:
        content = f.read()
    assert "dash" in content.lower()

def test_dev_requirements_have_pytest():
    """Verify pytest is in dev requirements"""
    with open("requirements-dev.txt") as f:
        content = f.read()
    assert "pytest" in content.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_requirements.py -v`
Expected: FAIL - files don't exist

**Step 3: Create requirements.txt**

Create `requirements.txt`:

```txt
# Web Framework
dash>=2.14.0
dash-bootstrap-components>=1.5.0

# Data Processing
pandas>=2.0.0

# Git Operations
gitpython>=3.1.40

# WSGI Server
uvicorn>=0.24.0
gunicorn>=21.2.0

# FastAPI (for future backend integration)
fastapi>=0.104.0
```

**Step 4: Create requirements-dev.txt**

Create `requirements-dev.txt`:

```txt
-r requirements.txt

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0

# Code Quality
black>=23.10.0
flake8>=6.1.0
mypy>=1.6.0

# Development
ipython>=8.17.0
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_requirements.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt tests/test_requirements.py
git commit -m "feat: add project requirements files"
```

---

## Task 15: README Documentation

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

Create `tests/test_readme.py`:

```python
import os
import pytest

def test_readme_exists():
    """Verify README.md exists"""
    assert os.path.exists("README.md"), "README.md not found"

def test_readme_has_required_sections():
    """Verify README has required sections"""
    with open("README.md") as f:
        content = f.read().lower()

    required_sections = [
        'installation',
        'usage',
        'development'
    ]

    for section in required_sections:
        assert section in content, f"Section '{section}' not found in README"
```

**Step 2: Run test to verify current state**

Run: `pytest tests/test_readme.py -v`
Expected: FAIL - README is likely empty

**Step 3: Update README.md**

Modify `README.md`:

```markdown
# GitTracer - SWE Trajectory Analysis Platform

A platform for extracting and analyzing Software Engineering (SWE) trajectories from real open-source repositories. GitTracer analyzes Git commit histories to automatically identify, classify, and aggregate Feature development and Bug Fix trajectories.

## Features

- **Repository Analysis**: Clone and scan Git repositories for commit history
- **Commit Classification**: Automatic categorization (Feature, Bug Fix, Refactor, Docs, Test)
- **Trajectory Clustering**: Link related commits into complete development trajectories
- **Hand-Drawn UI**: Unique sketch-style interface for a personalized feel
- **Export**: Export trajectories as Markdown bundles for further analysis

## Installation

### Prerequisites

- Python 3.9+
- Git command-line tool

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd GitTracer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -r requirements-dev.txt
```

## Usage

### Running the Application

Start the Dash application:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:8050`

### Analyzing a Repository

1. Enter the local path to a Git repository
2. Specify the branch name (default: `main`)
3. Set the number of commits to analyze
4. Click "Fetch & Analyze"

## Development

### Running Tests

```bash
pytest tests/
```

With coverage:
```bash
pytest tests/ --cov=frontend --cov-report=html
```

### Project Structure

```
GitTracer/
├── app.py                  # Entry point
├── frontend/               # Dash frontend package
│   ├── app.py             # App factory
│   ├── layouts/           # Layout components
│   ├── callbacks/         # Dash callbacks
│   └── styles/            # CSS stylesheets
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## License

TBD
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_readme.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add README.md tests/test_readme.py
git commit -m "docs: add comprehensive README documentation"
```

---

## Task 16: Final Verification

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `pytest tests/ -v --cov=frontend`
Expected: All tests PASS with reasonable coverage

**Step 2: Verify app starts**

Run: `python app.py` (in background)
Expected: Server starts successfully

Verify: Open http://127.0.0.1:8050 in browser
Expected: See GitTracer UI with repository configuration panel

Stop server with Ctrl+C

**Step 3: Check git status**

Run: `git status`
Expected: No uncommitted changes (except maybe __pycache__)

**Step 4: Final commit**

If any files are untracked and should be included:

```bash
git add .
git commit -m "chore: finalize frontend implementation"
```

---

## Implementation Complete Checklist

- [x] Project structure set up
- [x] Hand-drawn CSS styles created
- [x] Main layout and app factory
- [x] Repository configuration component
- [x] Commit table component
- [x] Trajectory view component
- [x] Commit detail component
- [x] Repository analysis callbacks
- [x] Navigation callbacks
- [x] Trajectory display callbacks
- [x] Root app.py updated
- [x] Requirements files added
- [x] README documentation added
- [x] All tests passing

---

## Next Steps (Future Enhancements)

1. **FastAPI Backend Integration**: Add `/api/fetch`, `/api/analyze` endpoints
2. **AI Analysis Module**: Implement LLM-based commit classification using LiteLLM
3. **Trajectory Clustering**: Use DSPy framework for semantic trajectory grouping
4. **MLflow Tracking**: Track all LLM prompts and responses
5. **Export Functionality**: Markdown bundle download for trajectories
6. **Authentication**: User management for multi-tenant deployment
