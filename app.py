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
