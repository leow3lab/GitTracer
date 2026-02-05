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
- [uv](https://github.com/astral-sh/uv) (recommended Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd GitTracer
```

2. Install dependencies with uv:
```bash
uv sync --dev
```

Alternatively, with pip:
```bash
pip install -e .
```

For development:
```bash
uv sync --dev --all-extras
```

## Usage

### Running the Application

Start the Dash application:

```bash
# With uv (recommended)
uv run python app.py

# Or directly with the venv Python
.venv/bin/python app.py
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
# With uv
uv run pytest tests/

# Or directly
.venv/bin/pytest tests/
```

With coverage:
```bash
uv run pytest tests/ --cov=frontend --cov-report=html
```

### Code Formatting

```bash
# Format code with black
uv run black frontend/ tests/

# Check linting
uv run flake8 frontend/ tests/
```

### Project Structure

```
GitTracer/
├── app.py                  # Entry point with GitDataFetcher
├── frontend/               # Dash frontend package
│   ├── app.py             # App factory
│   ├── layouts/           # Layout components
│   │   ├── main_layout.py # Main layout
│   │   └── components/    # UI components
│   │       ├── repo_config.py
│   │       ├── commit_table.py
│   │       ├── trajectory_view.py
│   │       └── commit_detail.py
│   ├── callbacks/         # Dash callbacks
│   │   ├── repo_callbacks.py
│   │   ├── navigation_callbacks.py
│   │   └── trajectory_callbacks.py
│   └── styles/            # CSS stylesheets
│       └── sketch_style.css
├── tests/                 # Test suite
│   └── frontend/
├── pyproject.toml         # Project configuration
└── docs/                  # Documentation
```

## Technology Stack

- **Frontend**: Dash (Plotly), Dash Bootstrap Components
- **Backend**: Python with Git subprocess integration
- **Style**: Hand-drawn/sketch CSS (custom)

## License

TBD
