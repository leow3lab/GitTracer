# Frontend Dash App - Trajectory View Only

**Goal:** Single-page Dash app for GitTracer trajectory analysis with hand-drawn/sketch UI style.

**Architecture:**
- Single page with trajectory cards
- Each trajectory card is expandable to show commit details
- Auto-clone remote repos to local cache
- Hand-drawn UI with custom CSS

---

## Project Structure

```
frontend/
├── app.py
├── layouts/
│   ├── main_layout.py
│   └── components/
│       ├── repo_config.py      # Repo input (local or URL) + fetch button
│       └── trajectory_view.py  # Expandable trajectory cards with commit details
├── callbacks/
│   ├── repo_callbacks.py       # Fetch & analyze callback
│   └── trajectory_callbacks.py # Expand/collapse trajectory cards
└── styles/
    └── sketch_style.css        # Hand-drawn UI styles

data/repos/                      # Local cache for cloned repos
└── org_repo_hash/              # e.g., github.com_vllm_vllm-a1b2c3d4/
```

---

## Components

### 1. Repo Config
- Repo path input (supports local path OR Git URL)
  - Local: `/path/to/repo`
  - HTTPS: `https://github.com/org/repo.git`
  - SSH: `git@github.com:org/repo.git`
- Branch name input
- "Fetch & Analyze" button
- Help text: "For remote URLs, ensure `git clone <url>` works in terminal first"
- Status alert (success/error/loading)
  - Loading: "Cloning remote repo..."
  - Success: "Successfully loaded N commits from 'main' (remote repo)"
  - Error: "Failed to clone <url>: <error>"

### 2. Trajectory View
- Empty state: "No trajectories yet"
- Trajectory cards (expandable):
  - Type badge (feature/bug_fix/refactor/docs/test)
  - Title + description
  - Commit count
  - "View Details" toggle
- Expanded card shows:
  - Timeline of commits
  - Each commit: hash, subject, author, date
  - Click commit to view diff

### 3. Commit Detail (inline)
- Shown below commit when clicked
- Code diff in scrollable pre block
- Collapse button

---

## Backend: GitDataFetcher Enhancements

### URL Detection
```python
@staticmethod
def _is_git_url(path: str) -> bool:
    """Check if path is a Git URL."""
    if not isinstance(path, str):
        return False
    v = path.strip()
    prefixes = ("http://", "https://", "ssh://", "git://", "file://")
    return v.startswith(prefixes) or re.match(r"^[\w.-]+@[\w.-]+:.*", v)
```

### Cache Directory
- Location: `<project_root>/data/repos/`
- Naming: `org_repo_slug-url_hash/`
  - Example: `github.com_vllm_vllm-a1b2c3d4/`

### Clone/Fetch Flow
```python
def _ensure_local_repo(self, repo_input: str, branch: str) -> str:
    """Ensure repo is local. Clone if URL, otherwise use local path."""
    if self._is_git_url(repo_input):
        cache_dir = self._compute_cache_dir(repo_input)
        if os.path.exists(cache_dir):
            self._run_git(["fetch", "--all", "--prune"], cwd=cache_dir)
        else:
            os.makedirs(cache_dir, exist_ok=True)
            self._run_git(
                ["clone", "--no-tags", "--filter=blob:none", repo_input, cache_dir]
            )
        self._run_git(["checkout", branch], cwd=cache_dir)
        return cache_dir
    return os.path.abspath(repo_input)
```

### Error Messages
- Clone fail: `"Failed to clone <url>: <stderr>"`
- Fetch fail: `"Failed to fetch updates: <stderr>"`
- Checkout fail: `"Branch <branch> not found in <url>"`

---

## Tasks

1. Setup `frontend/` package structure
2. Create `sketch_style.css` with hand-drawn aesthetic
3. Create `main_layout.py` with repo config + trajectory view
4. Create `repo_config.py` component (URL support)
5. Create `trajectory_view.py` with expandable cards
6. Enhance `GitDataFetcher` with URL detection and auto-clone
7. Implement fetch callback in `repo_callbacks.py` with loading state
8. Implement expand/collapse in `trajectory_callbacks.py`
9. Update `app.py` entry point
10. Test local path and URL inputs

---

## Tech Stack

- Dash + Dash Bootstrap Components
- Custom CSS for sketch aesthetic
- subprocess for git commands (no GitPython)
