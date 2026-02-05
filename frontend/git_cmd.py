"""Core git command wrapper. Cross-platform (Windows/Linux)."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime

# Detect git executable based on platform
_GIT_EXE = "git.exe" if platform.system() == "Windows" else "git"


def _find_git() -> str | None:
    """Find git executable in PATH."""
    return shutil.which("git") or shutil.which("git.exe")


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
        dt = datetime.fromtimestamp(self.timestamp)
        return f"""### [{self.idx}] Commit: {self.commit_id}
Subject: {self.subject}
Author: {self.author} (<{self.email}>)
Date: {dt.strftime("%Y-%m-%d %H:%M:%S")}

---

#### Code Changes
```diff
{self.diff if self.diff.strip() else "No code changes in this commit."}
```"""


class GitDataFetcher:
    """Handles Git repository interaction using subprocess."""

    def __init__(self, repo_path, branch="main", top_k=100):
        self.original_repo_input = repo_path
        self.branch = branch
        self.top_k = top_k
        self._git_cmd = _find_git() or _GIT_EXE
        self.repo_path = self._ensure_local_repo(repo_path)
        self._branches = None  # Cache for branches

    def _run_git(self, args, timeout=60):
        """Run git command. Returns stdout."""
        try:
            return subprocess.check_output(
                [self._git_cmd] + args,
                cwd=self.repo_path,
                encoding="utf-8",
                errors="ignore",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Git command timed out after {timeout}s")

    @staticmethod
    def _is_git_url(value: str) -> bool:
        """Check if value is a Git URL."""
        if not isinstance(value, str):
            return False
        v = value.strip()
        if v.startswith(("http://", "https://", "ssh://", "git://", "file://")):
            return True
        # SCP-like SSH: git@github.com:org/repo.git
        if re.match(r"^[\w.-]+@[\w.-]+:.*", v):
            return True
        return False

    @staticmethod
    def _run_cmd(args, cwd=None, git_cmd=None, timeout=120) -> str:
        """Run command with optional git executable."""
        git = git_cmd or _GIT_EXE
        try:
            proc = subprocess.run(
                [git] + args if args else args,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="ignore",
                check=False,
                timeout=timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    proc.stderr.strip() or proc.stdout.strip() or f"Command failed: {args}"
                )
            return proc.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Command timed out after {timeout}s (network issue?): {args[0] if args else 'unknown'}"
            )

    @staticmethod
    def _compute_cache_dir(repo_url: str) -> str:
        """Compute stable cache dir: data/repos/<slug>-<hash>/"""
        url = repo_url.strip()
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        slug = re.sub(r"^[a-z]+://", "", url)  # strip scheme
        slug = slug.replace(":", "_").replace("/", "_").replace("@", "_")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug).strip("_")
        if not slug:
            slug = "repo"
        # Use os.path.join for cross-platform paths
        return os.path.abspath(os.path.join("data", "repos", f"{slug}-{h}"))

    @staticmethod
    def _default_branch(local_repo_path: str, git_cmd=None) -> str | None:
        """Detect origin's default branch (e.g. 'main' from 'origin/main')."""
        try:
            git = git_cmd or _GIT_EXE
            out = subprocess.run(
                [git, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
                cwd=local_repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="ignore",
                check=True,
                timeout=30,
            ).stdout.strip()
            if out.startswith("origin/"):
                return out.split("/", 1)[1]
        except Exception:
            pass
        return None

    def _ensure_local_repo(self, repo_input: str) -> str:
        """Clone/fetch if URL, otherwise use local path."""
        if not self._is_git_url(repo_input):
            return os.path.abspath(repo_input)

        local_dir = self._compute_cache_dir(repo_input)
        cache_parent = os.path.dirname(local_dir)
        os.makedirs(cache_parent, exist_ok=True)

        git_dir = os.path.join(local_dir, ".git")
        if not os.path.exists(git_dir):
            # Try partial clone for speed, fallback to full clone
            try:
                self._run_cmd(
                    ["clone", "--no-tags", "--filter=blob:none", repo_input, local_dir],
                    git_cmd=self._git_cmd,
                    timeout=180,  # 3 minutes for clone
                )
            except Exception as e:
                # Retry with full clone
                try:
                    self._run_cmd(
                        ["clone", "--no-tags", repo_input, local_dir],
                        git_cmd=self._git_cmd,
                        timeout=300,
                    )
                except Exception:
                    raise RuntimeError(
                        f"Failed to clone {repo_input}. Error: {str(e)}\n\nTip: Check your network/proxy settings."
                    )
        else:
            try:
                self._run_cmd(
                    ["fetch", "--all", "--prune"],
                    cwd=local_dir,
                    git_cmd=self._git_cmd,
                    timeout=120,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to fetch updates. Error: {str(e)}")

        # Best-effort checkout
        desired_branch = getattr(self, "branch", "main")
        if desired_branch:
            try:
                self._run_cmd(
                    ["checkout", desired_branch],
                    cwd=local_dir,
                    git_cmd=self._git_cmd,
                    timeout=30,
                )
            except Exception:
                default_branch = self._default_branch(local_dir, self._git_cmd)
                if default_branch:
                    self._run_cmd(
                        ["checkout", default_branch],
                        cwd=local_dir,
                        git_cmd=self._git_cmd,
                        timeout=30,
                    )
                    self.branch = default_branch

        return local_dir

    def get_branches(self) -> list[str]:
        """Get all available branches (local and remote)."""
        if self._branches is not None:
            return self._branches

        branches = []
        try:
            # Get local branches
            out = self._run_git(["branch", "--format=%(refname:short)"], timeout=10)
            branches.extend(out.strip().split("\n"))

            # Get remote branches
            out = self._run_git(["branch", "-r", "--format=%(refname:short)"], timeout=10)
            for b in out.strip().split("\n"):
                if b and not b.endswith("HEAD") and "HEAD" not in b:
                    # Strip origin/ prefix for display
                    clean_name = b.replace("origin/", "")
                    if clean_name not in branches:
                        branches.append(clean_name)
        except Exception:
            pass

        self._branches = sorted(set(branches)) if branches else [self.branch]
        return self._branches

    def fetch_commits(self):
        """Fetch commit history. Returns (commits_list, error_or_none)."""
        log_format = "%H|%an|%ae|%at|%s"
        try:
            log_output = self._run_git(
                ["log", self.branch, f"--pretty=format:{log_format}", "-n", str(self.top_k)],
                timeout=60,
            )
        except Exception as e:
            return None, str(e)

        commits_list = []
        for i, line in enumerate(l for l in log_output.split("\n") if l):
            parts = line.split("|")
            if len(parts) < 5:
                continue
            h, name, email, time_val, subj = parts[:5]

            try:
                diff_content = self._run_git(
                    ["show", "--patch", "-m", "--pretty=format:", h], timeout=30
                )
            except Exception:
                diff_content = ""

            commits_list.append(
                {
                    "idx": i + 1,
                    "commit": h,
                    "author": name,
                    "email": email,
                    "timestamp": int(time_val),
                    "subject": subj,
                    "diff": diff_content,
                }
            )

        return commits_list, None
