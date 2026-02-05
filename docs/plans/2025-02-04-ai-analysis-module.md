# AI Analysis Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-step.

**Goal:** Build an AI-powered commit classification and trajectory clustering system that analyzes Git repositories to automatically identify Feature development and Bug Fix trajectories using LLMs.

**Architecture:**
- LiteLLM for unified LLM API calls (supports OpenAI, Anthropic, etc.)
- DSPy framework for prompt optimization and agent development
- MLflow for tracking all LLM prompts, responses, and metrics
- Modular design: CommitClassifier → TrajectoryCluster → orchestrator
- Keyword extraction + semantic analysis for efficient classification

**Tech Stack:**
- LiteLLM (LLM abstraction)
- DSPy (prompt engineering framework)
- MLflow (experiment tracking)
- scikit-learn (clustering helper)
- numpy/pandas (data processing)

---

## Project Structure

```
GitTracer/
├── ai_analysis/               # NEW: AI Analysis module
│   ├── __init__.py
│   ├── config.py              # Configuration for LLM providers, MLflow
│   ├── llm/                   # LLM abstraction layer
│   │   ├── __init__.py
│   │   ├── base.py            # Base LLM client interface
│   │   ├── litellm_client.py  # LiteLLM implementation
│   │   └── prompts.py         # Prompt templates (DSPy signatures)
│   ├── classifiers/           # Commit classification
│   │   ├── __init__.py
│   │   ├── commit_classifier.py   # Main classifier
│   │   └── keyword_extractor.py   # Keyword extraction helper
│   ├── clustering/            # Trajectory clustering
│   │   ├── __init__.py
│   │   ├── trajectory_cluster.py  # Main clustering logic
│   │   └── semantic_matcher.py    # Semantic similarity
│   ├── tracking/              # MLflow tracking
│   │   ├── __init__.py
│   │   └── mlflow_tracker.py
│   └── models/                # Data models
│       ├── __init__.py
│       ├── commit.py
│       └── trajectory.py
├── tests/
│   └── ai_analysis/           # NEW: Test suite
│       ├── __init__.py
│       ├── test_classifiers.py
│       ├── test_clustering.py
│       └── test_llm.py
└── requirements.txt           # Will be updated
```

---

## Task 1: AI Analysis Package Structure

**Files:**
- Create: `ai_analysis/__init__.py`
- Create: `ai_analysis/llm/__init__.py`
- Create: `ai_analysis/classifiers/__init__.py`
- Create: `ai_analysis/clustering/__init__.py`
- Create: `ai_analysis/tracking/__init__.py`
- Create: `ai_analysis/models/__init__.py`
- Create: `tests/ai_analysis/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p ai_analysis/llm
mkdir -p ai_analysis/classifiers
mkdir -p ai_analysis/clustering
mkdir -p ai_analysis/tracking
mkdir -p ai_analysis/models
mkdir -p tests/ai_analysis
```

**Step 2: Create __init__.py files**

```bash
touch ai_analysis/__init__.py
touch ai_analysis/llm/__init__.py
touch ai_analysis/classifiers/__init__.py
touch ai_analysis/clustering/__init__.py
touch ai_analysis/tracking/__init__.py
touch ai_analysis/models/__init__.py
touch tests/ai_analysis/__init__.py
```

**Step 3: Write the failing test**

Create `tests/ai_analysis/test_package.py`:

```python
import pytest

def test_ai_analysis_package_exists():
    """Verify ai_analysis package can be imported"""
    from ai_analysis import llm, classifiers, clustering, tracking, models
    assert hasattr(llm, '__file__')
    assert hasattr(classifiers, '__file__')
    assert hasattr(clustering, '__file__')
    assert hasattr(tracking, '__file__')
    assert hasattr(models, '__file__')
```

**Step 4: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_package.py -v`
Expected: PASS (directories and __init__.py exist)

**Step 5: Add package exports**

Modify `ai_analysis/__init__.py`:

```python
"""AI Analysis Module for GitTracer.

This module provides LLM-powered commit classification and trajectory
clustering capabilities.
"""

__version__ = "0.1.0"

from ai_analysis import llm, classifiers, clustering, tracking, models

__all__ = [
    "llm",
    "classifiers",
    "clustering",
    "tracking",
    "models"
]
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/ai_analysis/test_package.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add ai_analysis/ tests/ai_analysis/
git commit -m "feat: set up ai_analysis package structure"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `ai_analysis/config.py`
- Test: `tests/ai_analysis/test_config.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_config.py`:

```python
import os
import pytest
from ai_analysis.config import get_config, Config

def test_config_singleton():
    """Verify config is a singleton"""
    from ai_analysis.config import get_config
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2

def test_config_has_llm_settings():
    """Verify config has LLM provider settings"""
    config = get_config()
    assert hasattr(config, 'llm_provider')
    assert hasattr(config, 'model_name')
    assert hasattr(config, 'api_key')

def test_config_has_mlflow_settings():
    """Verify config has MLflow settings"""
    config = get_config()
    assert hasattr(config, 'mlflow_tracking_uri')
    assert hasattr(config, 'mlflow_experiment_name')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_config.py -v`
Expected: FAIL with import error

**Step 3: Create config module**

Create `ai_analysis/config.py`:

```python
"""Configuration for AI Analysis module.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuration for AI Analysis module."""

    # LLM Provider Settings
    llm_provider: str = field(default_factory=lambda: os.getenv(
        "LLM_PROVIDER", "openai"
    ))
    model_name: str = field(default_factory=lambda: os.getenv(
        "MODEL_NAME", "gpt-4o-mini"
    ))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv(
        "OPENAI_API_KEY"  # or ANTHROPIC_API_KEY
    ))
    api_base: Optional[str] = field(default_factory=lambda: os.getenv(
        "API_BASE"
    ))
    temperature: float = field(default_factory=lambda: float(os.getenv(
        "LLM_TEMPERATURE", "0.1"
    )))
    max_tokens: int = field(default_factory=lambda: int(os.getenv(
        "LLM_MAX_TOKENS", "500"
    )))

    # MLflow Settings
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
    ))
    mlflow_experiment_name: str = "gittracer_commit_analysis"
    mlflow_enabled: bool = field(default_factory=lambda: os.getenv(
        "MLFLOW_ENABLED", "true"
    ).lower() == "true")

    # Classification Settings
    commit_types: list = field(default_factory=lambda: [
        "feature", "bug_fix", "refactor", "docs", "test"
    ])

    # Clustering Settings
    similarity_threshold: float = 0.7
    max_trajectory_size: int = 50


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the singleton config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set a new config instance (for testing)."""
    global _config
    _config = config
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/ai_analysis/test_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ai_analysis/config.py tests/ai_analysis/test_config.py
git commit -m "feat: add configuration module with environment variables"
```

---

## Task 3: Data Models

**Files:**
- Create: `ai_analysis/models/commit.py`
- Create: `ai_analysis/models/trajectory.py`
- Modify: `ai_analysis/models/__init__.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_models.py`:

```python
from ai_analysis.models.commit import ClassifiedCommit
from ai_analysis.models.trajectory import Trajectory

def test_classified_commit_model():
    """Verify ClassifiedCommit data model"""
    commit_data = {
        "commit": "abc123",
        "subject": "Add new feature",
        "author": "Test Author",
        "timestamp": 1234567890,
        "diff": "+ new code"
    }

    commit = ClassifiedCommit(
        commit_hash="abc123",
        subject="Add new feature",
        author="Test Author",
        timestamp=1234567890,
        diff="+ new code",
        commit_type="feature",
        confidence=0.95,
        keywords=["feature", "add"]
    )

    assert commit.commit_hash == "abc123"
    assert commit.commit_type == "feature"
    assert commit.confidence == 0.95
    assert commit.keywords == ["feature", "add"]

def test_trajectory_model():
    """Verify Trajectory data model"""
    trajectory = Trajectory(
        trajectory_id="traj_1",
        trajectory_type="feature",
        title="Add authentication",
        description="Implementation of user authentication",
        commit_hashes=["abc123", "def456"]
    )

    assert trajectory.trajectory_id == "traj_1"
    assert trajectory.trajectory_type == "feature"
    assert lentrajectory.commit_hashes) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_models.py -v`
Expected: FAIL with import error

**Step 3: Create commit model**

Create `ai_analysis/models/commit.py`:

```python
"""Data models for Git commits."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class CommitType(Enum):
    """Classification types for commits."""
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    DOCS = "docs"
    TEST = "test"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedCommit:
    """A commit with AI classification results."""

    # Original commit data
    commit_hash: str
    subject: str
    author: str
    timestamp: int
    diff: str
    email: Optional[str] = None
    idx: Optional[int] = None

    # Classification results
    commit_type: CommitType = CommitType.UNKNOWN
    confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)

    # Metadata
    classified_at: Optional[datetime] = None
    classification_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "commit_hash": self.commit_hash,
            "subject": self.subject,
            "author": self.author,
            "timestamp": self.timestamp,
            "diff": self.diff[:1000],  # Truncate for storage
            "commit_type": self.commit_type.value,
            "confidence": self.confidence,
            "keywords": self.keywords,
        }

    @classmethod
    def from_raw_dict(cls, data: Dict[str, Any], idx: Optional[int] = None) -> "ClassifiedCommit":
        """Create from raw Git data dictionary."""
        return cls(
            commit_hash=data.get("commit", "unknown"),
            subject=data.get("subject", "No Subject"),
            author=data.get("author", "Unknown"),
            timestamp=data.get("timestamp", 0),
            diff=data.get("diff", ""),
            email=data.get("email"),
            idx=idx or data.get("idx")
        )
```

**Step 4: Create trajectory model**

Create `ai_analysis/models/trajectory.py`:

```python
"""Data models for trajectories."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class TrajectoryType(Enum):
    """Types of trajectories."""
    FEATURE_DEVELOPMENT = "feature"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactor"
    MIXED = "mixed"


@dataclass
class Trajectory:
    """A trajectory is a group of related commits."""

    trajectory_id: str
    trajectory_type: TrajectoryType
    title: str
    description: str
    commit_hashes: List[str]

    # Additional metadata
    start_timestamp: Optional[int] = None
    end_timestamp: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None

    # Clustering metadata
    similarity_score: float = 0.0
    cluster_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def commit_count(self) -> int:
        """Return number of commits in trajectory."""
        return len(self.commit_hashes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trajectory_id": self.trajectory_id,
            "trajectory_type": self.trajectory_type.value,
            "title": self.title,
            "description": self.description,
            "commit_count": self.commit_count,
            "commit_hashes": self.commit_hashes,
            "authors": self.authors,
            "similarity_score": self.similarity_score,
        }

    def add_commit(self, commit_hash: str) -> None:
        """Add a commit hash to this trajectory."""
        if commit_hash not in self.commit_hashes:
            self.commit_hashes.append(commit_hash)
```

**Step 5: Update models __init__.py**

Modify `ai_analysis/models/__init__.py`:

```python
"""Data models for AI Analysis module."""

from ai_analysis.models.commit import ClassifiedCommit, CommitType
from ai_analysis.models.trajectory import Trajectory, TrajectoryType

__all__ = [
    "ClassifiedCommit",
    "CommitType",
    "Trajectory",
    "TrajectoryType"
]
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_models.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add ai_analysis/models/ tests/ai_analysis/test_models.py
git commit -m "feat: add data models for commits and trajectories"
```

---

## Task 4: LLM Base Interface

**Files:**
- Create: `ai_analysis/llm/base.py`
- Test: `tests/ai_analysis/test_llm_base.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_llm_base.py`:

```python
import pytest
from abc import ABC

def test_base_llm_client_is_abstract():
    """Verify base LLM client is abstract"""
    from ai_analysis.llm.base import BaseLLMClient

    # Should not be able to instantiate
    with pytest.raises(TypeError):
        BaseLLMClient()

def test_base_llm_has_required_methods():
    """Verify base client defines required interface"""
    from ai_analysis.llm.base import BaseLLMClient

    assert hasattr(BaseLLMClient, 'classify_commit')
    assert hasattr(BaseLLMClient, 'cluster_commits')
    assert hasattr(BaseLLMClient, 'extract_keywords')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_llm_base.py -v`
Expected: FAIL with import error

**Step 3: Create base LLM client**

Create `ai_analysis/llm/base.py`:

```python
"""Base interface for LLM clients."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ai_analysis.models.commit import ClassifiedCommit


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config=None):
        """Initialize the LLM client.

        Args:
            config: Optional configuration object
        """
        self.config = config
        self._client = None

    @abstractmethod
    def classify_commit(
        self,
        subject: str,
        diff: str,
        author: str,
        timestamp: int
    ) -> tuple[str, float, List[str]]:
        """Classify a single commit.

        Args:
            subject: Commit message subject
            diff: Git diff content
            author: Commit author
            timestamp: Unix timestamp

        Returns:
            Tuple of (commit_type, confidence, keywords)
        """
        pass

    @abstractmethod
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to return

        Returns:
            List of keyword strings
        """
        pass

    @abstractmethod
    def cluster_commits(
        self,
        commits: List[ClassifiedCommit],
        trajectory_type: str
    ) -> List[Dict[str, Any]]:
        """Cluster related commits into trajectories.

        Args:
            commits: List of classified commits
            trajectory_type: Type of trajectory to cluster for

        Returns:
            List of trajectory dictionaries
        """
        pass

    def health_check(self) -> bool:
        """Check if LLM client is healthy and can make requests.

        Returns:
            True if healthy, False otherwise
        """
        return self._client is not None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_llm_base.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ai_analysis/llm/base.py tests/ai_analysis/test_llm_base.py
git commit -m "feat: add base LLM client interface"
```

---

## Task 5: LiteLLM Client Implementation

**Files:**
- Create: `ai_analysis/llm/litellm_client.py`
- Modify: `ai_analysis/llm/__init__.py`
- Test: `tests/ai_analysis/test_litellm_client.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_litellm_client.py`:

```python
import pytest
from unittest.mock import Mock, patch

def test_litellm_client_can_be_instantiated():
    """Verify LiteLLMClient can be created"""
    from ai_analysis.llm.litellm_client import LiteLLMClient
    from ai_analysis.config import Config

    config = Config(
        llm_provider="openai",
        model_name="gpt-4o-mini",
        api_key="test-key"
    )

    client = LiteLLMClient(config)
    assert client is not None
    assert client.config == config

@patch('ai_analysis.llm.litellm_client.litellm_completion')
def test_classify_commit_returns_tuple(mock_completion):
    """Verify classify_commit returns correct tuple structure"""
    from ai_analysis.llm.litellm_client import LiteLLMClient
    from ai_analysis.config import Config

    # Mock the LLM response
    mock_completion.return_value = {
        "choices": [{
            "message": {
                "content": '{"type": "feature", "confidence": 0.9, "keywords": ["auth", "login"]}'
            }
        }]
    }

    config = Config(api_key="test-key")
    client = LiteLLMClient(config)

    result = client.classify_commit(
        subject="Add user authentication",
        diff="+ def authenticate():\n+     pass",
        author="Test Author",
        timestamp=1234567890
    )

    assert isinstance(result, tuple)
    assert len(result) == 3
    commit_type, confidence, keywords = result
    assert commit_type == "feature"
    assert confidence == 0.9
    assert keywords == ["auth", "login"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_litellm_client.py -v`
Expected: FAIL with import error

**Step 3: Create LiteLLM client**

Create `ai_analysis/llm/litellm_client.py`:

```python
"""LiteLLM implementation of LLM client."""

import json
import os
from typing import List, Dict, Any, Tuple
import litellm
from ai_analysis.llm.base import BaseLLMClient
from ai_analysis.models.commit import ClassifiedCommit


def litellm_completion(*args, **kwargs):
    """Wrapper for litellm.completion for easier mocking in tests."""
    return litellm.completion(*args, **kwargs)


class LiteLLMClient(BaseLLMClient):
    """LiteLLM-based client for unified LLM API access."""

    # Classification prompt template
    CLASSIFY_PROMPT = """You are a Git commit classifier. Analyze the following commit and classify it.

Commit Subject: {subject}
Author: {author}

Code Changes (diff):
{diff_truncated}

Classify this commit as one of: {commit_types}.

Respond with JSON format:
{{"type": "classification", "confidence": 0.0-1.0, "keywords": ["key1", "key2"], "reasoning": "brief explanation"}}"""

    KEYWORD_PROMPT = """Extract 3-5 important keywords from the following text.

Text: {text}

Respond with JSON format:
{{"keywords": ["key1", "key2", "key3"]}}"""

    CLUSTER_PROMPT = """You are a trajectory analyzer. Given these commits, group related ones into trajectories.

Commits:
{commit_list}

Group commits that belong to the same feature or bug fix process.
Each trajectory should have: title, description, commit indices.

Respond with JSON format:
{{"trajectories": [{{"title": "...", "description": "...", "commit_indices": [0, 1, 2]}}]}}"""

    def __init__(self, config=None):
        """Initialize LiteLLM client.

        Args:
            config: Configuration object with API keys and model settings
        """
        super().__init__(config)

        # Set API key from config
        if config and config.api_key:
            os.environ["OPENAI_API_KEY"] = config.api_key

        # Configure litellm
        litellm.set_verbose = False

        self.model = config.model_name if config else "gpt-4o-mini"
        self.temperature = config.temperature if config else 0.1
        self.max_tokens = config.max_tokens if config else 500
        self.commit_types = config.commit_types if config else [
            "feature", "bug_fix", "refactor", "docs", "test"
        ]

    def _make_request(self, prompt: str) -> str:
        """Make a request to the LLM.

        Args:
            prompt: The prompt to send

        Returns:
            The LLM response content
        """
        response = litellm_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response["choices"][0]["message"]["content"]

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM.

        Args:
            response: Raw response string

        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try to extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            return json.loads(response)
        except json.JSONDecodeError:
            # Return safe default if parsing fails
            return {"type": "unknown", "confidence": 0.0, "keywords": []}

    def classify_commit(
        self,
        subject: str,
        diff: str,
        author: str,
        timestamp: int
    ) -> Tuple[str, float, List[str]]:
        """Classify a single commit.

        Args:
            subject: Commit message subject
            diff: Git diff content
            author: Commit author
            timestamp: Unix timestamp

        Returns:
            Tuple of (commit_type, confidence, keywords)
        """
        # Truncate diff if too long (to save tokens)
        diff_truncated = diff[:2000] if len(diff) > 2000 else diff

        prompt = self.CLASSIFY_PROMPT.format(
            subject=subject,
            author=author,
            diff_truncated=diff_truncated,
            commit_types=", ".join(self.commit_types)
        )

        response = self._make_request(prompt)
        result = self._parse_json_response(response)

        commit_type = result.get("type", "unknown")
        confidence = float(result.get("confidence", 0.0))
        keywords = result.get("keywords", [])

        return commit_type, confidence, keywords

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords

        Returns:
            List of keyword strings
        """
        prompt = self.KEYWORD_PROMPT.format(text=text[:500])

        response = self._make_request(prompt)
        result = self._parse_json_response(response)

        keywords = result.get("keywords", [])
        return keywords[:max_keywords]

    def cluster_commits(
        self,
        commits: List[ClassifiedCommit],
        trajectory_type: str
    ) -> List[Dict[str, Any]]:
        """Cluster related commits into trajectories.

        Args:
            commits: List of classified commits
            trajectory_type: Type to filter for (or 'all')

        Returns:
            List of trajectory dictionaries
        """
        # Filter commits by type if specified
        if trajectory_type != "all":
            filtered = [c for c in commits if c.commit_type.value == trajectory_type]
        else:
            filtered = commits

        if not filtered:
            return []

        # Format commits for the prompt
        commit_list = ""
        for i, commit in enumerate(filtered[:20]):  # Limit to 20 for context
            commit_list += f"\n{i}. Subject: {commit.subject}\n   Type: {commit.commit_type.value}\n"

        prompt = self.CLUSTER_PROMPT.format(commit_list=commit_list)

        response = self._make_request(prompt)
        result = self._parse_json_response(response)

        trajectories = result.get("trajectories", [])
        return trajectories
```

**Step 4: Update llm __init__.py**

Modify `ai_analysis/llm/__init__.py`:

```python
"""LLM abstraction layer for AI Analysis."""

from ai_analysis.llm.base import BaseLLMClient
from ai_analysis.llm.litellm_client import LiteLLMClient

__all__ = [
    "BaseLLMClient",
    "LiteLLMClient"
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_litellm_client.py -v`
Expected: All PASS (with mocked LLM calls)

**Step 6: Commit**

```bash
git add ai_analysis/llm/ tests/ai_analysis/test_litellm_client.py
git commit -m "feat: implement LiteLLM client with classification"
```

---

## Task 6: MLflow Tracking

**Files:**
- Create: `ai_analysis/tracking/mlflow_tracker.py`
- Modify: `ai_analysis/tracking/__init__.py`
- Test: `tests/ai_analysis/test_mlflow_tracker.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_mlflow_tracker.py`:

```python
import pytest
from unittest.mock import Mock, patch, MagicMock

def test_mlflow_tracker_singleton():
    """Verify MLflow tracker is a singleton"""
    from ai_analysis.tracking.mlflow_tracker import get_tracker
    tracker1 = get_tracker()
    tracker2 = get_tracker()
    assert tracker1 is tracker2

@patch('ai_analysis.tracking.mlflow_tracker.mlflow')
def test_log_classification_logs_to_mlflow(mock_mlflow):
    """Verify classification is logged to MLflow"""
    from ai_analysis.tracking.mlflow_tracker import get_tracker
    from ai_analysis.models.commit import ClassifiedCommit, CommitType

    tracker = get_tracker()
    tracker.start_run("test_run")

    commit = ClassifiedCommit(
        commit_hash="abc123",
        subject="Add feature",
        author="Test",
        timestamp=123456,
        diff="+ code",
        commit_type=CommitType.FEATURE,
        confidence=0.9,
        keywords=["feature"]
    )

    tracker.log_classification(commit, model="gpt-4o-mini")

    # Verify mlflow.log_metric was called
    assert mock_mlflow.log_metric.called or mock_mlflow.log_params.called
    tracker.end_run()

def test_tracker_disabled_when_config_disabled():
    """Verify tracker is disabled when MLFLOW_ENABLED=false"""
    import os
    os.environ["MLFLOW_ENABLED"] = "false"

    from ai_analysis.tracking.mlflow_tracker import get_tracker, MLflowTracker
    from ai_analysis.config import Config

    # Reset config
    import ai_analysis.config
    ai_analysis.config._config = Config()

    tracker = get_tracker()
    assert not tracker.enabled
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_mlflow_tracker.py -v`
Expected: FAIL with import error

**Step 3: Create MLflow tracker**

Create `ai_analysis/tracking/mlflow_tracker.py`:

```python
"""MLflow tracking for LLM calls and metrics."""

import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from functools import wraps
import mlflow
import mlflow.sklearn

from ai_analysis.config import get_config
from ai_analysis.models.commit import ClassifiedCommit


@dataclass
class ClassificationMetrics:
    """Metrics from a classification run."""
    commit_type: str
    confidence: float
    num_keywords: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


class MLflowTracker:
    """MLflow tracker for AI Analysis operations."""

    def __init__(self, config=None):
        """Initialize MLflow tracker.

        Args:
            config: Optional configuration object
        """
        self.config = config or get_config()
        self.enabled = self.config.mlflow_enabled

        if self.enabled:
            self._setup_mlflow()

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
        except Exception as e:
            print(f"Warning: Could not setup MLflow: {e}")
            self.enabled = False

    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run.

        Args:
            run_name: Optional name for the run
        """
        if self.enabled:
            mlflow.start_run(run_name=run_name)

    def end_run(self):
        """End the current MLflow run."""
        if self.enabled:
            mlflow.end_run()

    def log_classification(
        self,
        commit: ClassifiedCommit,
        model: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ):
        """Log a classification to MLflow.

        Args:
            commit: The classified commit
            model: Model name used
            latency_ms: Request latency in milliseconds
            prompt_tokens: Input token count
            completion_tokens: Output token count
        """
        if not self.enabled:
            return

        # Log metrics
        mlflow.log_metric("confidence", commit.confidence)
        mlflow.log_metric("num_keywords", len(commit.keywords))
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("total_tokens", prompt_tokens + completion_tokens)

        # Log parameters
        mlflow.log_param("model", model)
        mlflow.log_param("commit_type", commit.commit_type.value)
        mlflow.log_param("commit_hash", commit.commit_hash[:8])

    def log_clustering(
        self,
        num_trajectories: int,
        num_commits: int,
        trajectory_type: str
    ):
        """Log clustering results to MLflow.

        Args:
            num_trajectories: Number of trajectories found
            num_commits: Number of commits clustered
            trajectory_type: Type of trajectory
        """
        if not self.enabled:
            return

        mlflow.log_metric("num_trajectories", num_trajectories)
        mlflow.log_metric("num_commits_clustered", num_commits)
        mlflow.log_param("trajectory_type", trajectory_type)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameters
        """
        if not self.enabled:
            return

        for key, value in params.items():
            mlflow.log_param(key, str(value))

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics
        """
        if not self.enabled:
            return

        for key, value in metrics.items():
            mlflow.log_metric(key, value)


# Singleton instance
_tracker: Optional[MLflowTracker] = None


def get_tracker() -> MLflowTracker:
    """Get the singleton tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker()
    return _tracker


def track_llm_call(func):
    """Decorator to track LLM calls with MLflow.

    Usage:
        @track_llm_call
        def classify_commit(...):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()

        if not tracker.enabled:
            return func(*args, **kwargs)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Log the call
            tracker.log_metrics({"latency_ms": latency_ms})

            return result
        except Exception as e:
            tracker.log_metrics({"error": 1})
            raise

    return wrapper
```

**Step 4: Update tracking __init__.py**

Modify `ai_analysis/tracking/__init__.py`:

```python
"""MLflow tracking for AI Analysis."""

from ai_analysis.tracking.mlflow_tracker import (
    MLflowTracker,
    get_tracker,
    track_llm_call
)

__all__ = [
    "MLflowTracker",
    "get_tracker",
    "track_llm_call"
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_mlflow_tracker.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add ai_analysis/tracking/ tests/ai_analysis/test_mlflow_tracker.py
git commit -m "feat: add MLflow tracking for LLM calls"
```

---

## Task 7: Commit Classifier

**Files:**
- Create: `ai_analysis/classifiers/commit_classifier.py`
- Modify: `ai_analysis/classifiers/__init__.py`
- Test: `tests/ai_analysis/test_commit_classifier.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_commit_classifier.py`:

```python
import pytest
from unittest.mock import Mock, patch

def test_commit_classifier_can_classify():
    """Verify commit classifier can classify a commit"""
    from ai_analysis.classifiers.commit_classifier import CommitClassifier
    from ai_analysis.config import Config
    from ai_analysis.models.commit import ClassifiedCommit

    config = Config(api_key="test-key")
    classifier = CommitClassifier(config)

    commit_data = {
        "commit": "abc123",
        "subject": "Add user authentication",
        "author": "Test Author",
        "timestamp": 1234567890,
        "diff": "+ def authenticate(user, password):"
    }

    with patch.object(classifier.llm_client, 'classify_commit') as mock_classify:
        mock_classify.return_value = ("feature", 0.9, ["auth", "user"])

        result = classifier.classify(commit_data)

        assert isinstance(result, ClassifiedCommit)
        assert result.commit_type.value == "feature"
        assert result.confidence == 0.9
        assert result.keywords == ["auth", "user"]

def test_commit_classifier_batch():
    """Verify classifier can handle batch of commits"""
    from ai_analysis.classifiers.commit_classifier import CommitClassifier
    from ai_analysis.config import Config

    config = Config(api_key="test-key")
    classifier = CommitClassifier(config)

    commits = [
        {
            "commit": f"abc{i}",
            "subject": f"Commit {i}",
            "author": "Test",
            "timestamp": 1234567890,
            "diff": "+ code"
        }
        for i in range(5)
    ]

    with patch.object(classifier.llm_client, 'classify_commit') as mock_classify:
        mock_classify.return_value = ("feature", 0.8, ["test"])

        results = classifier.classify_batch(commits)

        assert len(results) == 5
        assert all(isinstance(c, ClassifiedCommit) for c in results)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_commit_classifier.py -v`
Expected: FAIL with import error

**Step 3: Create commit classifier**

Create `ai_analysis/classifiers/commit_classifier.py`:

```python
"""Commit classifier using LLM."""

import time
from typing import List, Dict, Any, Optional
from ai_analysis.config import get_config
from ai_analysis.llm import LiteLLMClient
from ai_analysis.models.commit import ClassifiedCommit, CommitType
from ai_analysis.tracking import get_tracker


class CommitClassifier:
    """Classifier for Git commits using LLM."""

    def __init__(self, config=None):
        """Initialize the classifier.

        Args:
            config: Optional configuration object
        """
        self.config = config or get_config()
        self.llm_client = LiteLLMClient(self.config)
        self.tracker = get_tracker()

    def classify(self, commit_data: Dict[str, Any]) -> ClassifiedCommit:
        """Classify a single commit.

        Args:
            commit_data: Dictionary with commit data (commit, subject, author, timestamp, diff)

        Returns:
            ClassifiedCommit object with classification results
        """
        start_time = time.time()

        # Call LLM for classification
        commit_type, confidence, keywords = self.llm_client.classify_commit(
            subject=commit_data.get("subject", ""),
            diff=commit_data.get("diff", ""),
            author=commit_data.get("author", "Unknown"),
            timestamp=commit_data.get("timestamp", 0)
        )

        latency_ms = (time.time() - start_time) * 1000

        # Map string type to enum
        try:
            type_enum = CommitType(commit_type)
        except ValueError:
            type_enum = CommitType.UNKNOWN

        # Create classified commit
        result = ClassifiedCommit(
            commit_hash=commit_data.get("commit", "unknown"),
            subject=commit_data.get("subject", "No Subject"),
            author=commit_data.get("author", "Unknown"),
            timestamp=commit_data.get("timestamp", 0),
            diff=commit_data.get("diff", ""),
            email=commit_data.get("email"),
            idx=commit_data.get("idx"),
            commit_type=type_enum,
            confidence=confidence,
            keywords=keywords
        )

        # Log to MLflow
        self.tracker.log_classification(
            result,
            model=self.config.model_name,
            latency_ms=latency_ms
        )

        return result

    def classify_batch(
        self,
        commits_data: List[Dict[str, Any]],
        show_progress: bool = False
    ) -> List[ClassifiedCommit]:
        """Classify a batch of commits.

        Args:
            commits_data: List of commit dictionaries
            show_progress: Whether to show progress (if tqdm available)

        Returns:
            List of ClassifiedCommit objects
        """
        results = []

        for i, commit_data in enumerate(commits_data):
            if show_progress:
                print(f"Classifying commit {i+1}/{len(commits_data)}")

            result = self.classify(commit_data)
            results.append(result)

        return results

    def filter_by_type(
        self,
        commits: List[ClassifiedCommit],
        commit_type: str
    ) -> List[ClassifiedCommit]:
        """Filter commits by type.

        Args:
            commits: List of classified commits
            commit_type: Type to filter by (e.g., "feature", "bug_fix")

        Returns:
            Filtered list of commits
        """
        return [c for c in commits if c.commit_type.value == commit_type]
```

**Step 4: Update classifiers __init__.py**

Modify `ai_analysis/classifiers/__init__.py`:

```python
"""Commit classification module."""

from ai_analysis.classifiers.commit_classifier import CommitClassifier

__all__ = [
    "CommitClassifier"
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_commit_classifier.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add ai_analysis/classifiers/ tests/ai_analysis/test_commit_classifier.py
git commit -m "feat: add commit classifier with batch support"
```

---

## Task 8: Keyword Extractor

**Files:**
- Create: `ai_analysis/classifiers/keyword_extractor.py`
- Test: `tests/ai_analysis/test_keyword_extractor.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_keyword_extractor.py`:

```python
import pytest
from unittest.mock import patch

def test_keyword_extractor_extracts_keywords():
    """Verify keyword extractor works"""
    from ai_analysis.classifiers.keyword_extractor import KeywordExtractor
    from ai_analysis.config import Config

    config = Config(api_key="test-key")
    extractor = KeywordExtractor(config)

    with patch.object(extractor.llm_client, 'extract_keywords') as mock_extract:
        mock_extract.return_value = ["authentication", "user", "login"]

        result = extractor.extract("Add user authentication feature")

        assert result == ["authentication", "user", "login"]

def test_keyword_extractor_from_commits():
    """Verify extracting keywords from commit list"""
    from ai_analysis.classifiers.keyword_extractor import KeywordExtractor
    from ai_analysis.models.commit import ClassifiedCommit, CommitType

    extractor = KeywordExtractor()

    commits = [
        ClassifiedCommit(
            commit_hash="abc1",
            subject="Add authentication",
            author="Test",
            timestamp=123456,
            diff="+ auth code",
            commit_type=CommitType.FEATURE,
            keywords=["auth"]
        ),
        ClassifiedCommit(
            commit_hash="abc2",
            subject="Fix authentication bug",
            author="Test",
            timestamp=123457,
            diff="+ fix",
            commit_type=CommitType.BUG_FIX,
            keywords=["bug", "auth"]
        )
    ]

    with patch.object(extractor.llm_client, 'extract_keywords') as mock_extract:
        mock_extract.return_value = ["authentication", "security"]

        result = extractor.extract_from_commits(commits)

        assert "authentication" in result or "auth" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_keyword_extractor.py -v`
Expected: FAIL with import error

**Step 3: Create keyword extractor**

Create `ai_analysis/classifiers/keyword_extractor.py`:

```python
"""Keyword extraction from commits."""

from typing import List, Set
from collections import Counter
from ai_analysis.config import get_config
from ai_analysis.llm import LiteLLMClient
from ai_analysis.models.commit import ClassifiedCommit


class KeywordExtractor:
    """Extract keywords from commit messages and diffs."""

    def __init__(self, config=None):
        """Initialize keyword extractor.

        Args:
            config: Optional configuration object
        """
        self.config = config or get_config()
        self.llm_client = LiteLLMClient(self.config)

    def extract(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using LLM.

        Args:
            text: Text to extract from
            max_keywords: Maximum keywords to return

        Returns:
            List of keywords
        """
        if not text or len(text.strip()) == 0:
            return []

        return self.llm_client.extract_keywords(text, max_keywords)

    def extract_from_commits(
        self,
        commits: List[ClassifiedCommit],
        max_keywords: int = 20
    ) -> List[str]:
        """Extract aggregated keywords from multiple commits.

        Args:
            commits: List of classified commits
            max_keywords: Maximum keywords to return

        Returns:
            List of top keywords
        """
        # Collect all existing keywords
        all_keywords: List[str] = []
        for commit in commits:
            all_keywords.extend(commit.keywords)

        # Count frequency
        keyword_counts = Counter(all_keywords)

        # Get top keywords
        top_keywords = [kw for kw, count in keyword_counts.most_common(max_keywords)]

        return top_keywords

    def extract_from_text_batch(
        self,
        texts: List[str],
        max_per_text: int = 5
    ) -> List[str]:
        """Extract keywords from multiple texts.

        Args:
            texts: List of texts to process
            max_per_text: Max keywords per text

        Returns:
            Combined list of unique keywords
        """
        all_keywords: Set[str] = set()

        for text in texts:
            keywords = self.extract(text, max_per_text)
            all_keywords.update(keywords)

        return list(all_keywords)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_keyword_extractor.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ai_analysis/classifiers/keyword_extractor.py tests/ai_analysis/test_keyword_extractor.py
git commit -m "feat: add keyword extraction from commits"
```

---

## Task 9: Trajectory Clustering

**Files:**
- Create: `ai_analysis/clustering/trajectory_cluster.py`
- Modify: `ai_analysis/clustering/__init__.py`
- Test: `tests/ai_analysis/test_trajectory_cluster.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_trajectory_cluster.py`:

```python
import pytest
from unittest.mock import patch, Mock
from ai_analysis.models.commit import ClassifiedCommit, CommitType

def test_trajectory_clusterer_creates_trajectories():
    """Verify trajectory clusterer creates trajectories"""
    from ai_analysis.clustering.trajectory_cluster import TrajectoryClusterer

    clusterer = TrajectoryClusterer()

    commits = [
        ClassifiedCommit(
            commit_hash="abc1",
            subject="Add authentication",
            author="Test",
            timestamp=123456,
            diff="+ auth",
            commit_type=CommitType.FEATURE,
            keywords=["auth"]
        ),
        ClassifiedCommit(
            commit_hash="abc2",
            subject="Add login page",
            author="Test",
            timestamp=123457,
            diff="+ login",
            commit_type=CommitType.FEATURE,
            keywords=["login", "auth"]
        )
    ]

    with patch.object(clusterer.llm_client, 'cluster_commits') as mock_cluster:
        mock_cluster.return_value = [
            {
                "title": "Authentication Feature",
                "description": "User authentication implementation",
                "commit_indices": [0, 1]
            }
        ]

        trajectories = clusterer.cluster(commits, trajectory_type="feature")

        assert len(trajectories) == 1
        assert trajectories[0].title == "Authentication Feature"

def test_trajectory_clusterer_filters_by_type():
    """Verify clustering filters by commit type"""
    from ai_analysis.clustering.trajectory_cluster import TrajectoryClusterer
    from ai_analysis.models.commit import ClassifiedCommit, CommitType

    clusterer = TrajectoryClusterer()

    commits = [
        ClassifiedCommit(
            commit_hash="abc1",
            subject="Add feature",
            author="Test",
            timestamp=123456,
            diff="+ code",
            commit_type=CommitType.FEATURE,
            keywords=[]
        ),
        ClassifiedCommit(
            commit_hash="abc2",
            subject="Fix bug",
            author="Test",
            timestamp=123457,
            diff="- bug",
            commit_type=CommitType.BUG_FIX,
            keywords=[]
        )
    ]

    with patch.object(clusterer.llm_client, 'cluster_commits') as mock_cluster:
        mock_cluster.return_value = []

        # Should only cluster features
        clusterer.cluster(commits, trajectory_type="feature")

        # Check that only feature commits were passed
        called_commits = mock_cluster.call_args[0][0]
        assert len(called_commits) == 1
        assert called_commits[0].commit_type == CommitType.FEATURE
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_trajectory_cluster.py -v`
Expected: FAIL with import error

**Step 3: Create trajectory clusterer**

Create `ai_analysis/clustering/trajectory_cluster.py`:

```python
"""Trajectory clustering for related commits."""

import time
from typing import List, Dict, Any, Optional
from ai_analysis.config import get_config
from ai_analysis.llm import LiteLLMClient
from ai_analysis.models.commit import ClassifiedCommit, CommitType
from ai_analysis.models.trajectory import Trajectory, TrajectoryType
from ai_analysis.tracking import get_tracker


class TrajectoryClusterer:
    """Cluster related commits into trajectories."""

    def __init__(self, config=None):
        """Initialize trajectory clusterer.

        Args:
            config: Optional configuration object
        """
        self.config = config or get_config()
        self.llm_client = LiteLLMClient(self.config)
        self.tracker = get_tracker()
        self.similarity_threshold = self.config.similarity_threshold

    def cluster(
        self,
        commits: List[ClassifiedCommit],
        trajectory_type: str = "all"
    ) -> List[Trajectory]:
        """Cluster commits into trajectories.

        Args:
            commits: List of classified commits
            trajectory_type: Type to filter ('all', 'feature', 'bug_fix', etc.)

        Returns:
            List of Trajectory objects
        """
        start_time = time.time()

        # Filter commits by type if specified
        if trajectory_type != "all":
            filtered = [c for c in commits if c.commit_type.value == trajectory_type]
        else:
            filtered = commits

        if not filtered:
            return []

        # Call LLM to cluster
        cluster_results = self.llm_client.cluster_commits(filtered, trajectory_type)

        # Convert results to Trajectory objects
        trajectories = []
        for i, result in enumerate(cluster_results):
            commit_indices = result.get("commit_indices", [])
            title = result.get("title", f"Trajectory {i+1}")
            description = result.get("description", "")

            # Get actual commit hashes
            trajectory_commits = [filtered[j] for j in commit_indices if j < len(filtered)]
            commit_hashes = [c.commit_hash for c in trajectory_commits]

            # Get metadata
            authors = list(set(c.author for c in trajectory_commits))
            timestamps = [c.timestamp for c in trajectory_commits]

            trajectory = Trajectory(
                trajectory_id=f"traj_{int(time.time())}_{i}",
                trajectory_type=self._map_trajectory_type(trajectory_type),
                title=title,
                description=description,
                commit_hashes=commit_hashes,
                start_timestamp=min(timestamps) if timestamps else None,
                end_timestamp=max(timestamps) if timestamps else None,
                authors=authors,
                similarity_score=0.0  # Could be computed later
            )

            trajectories.append(trajectory)

        # Log to MLflow
        self.tracker.log_clustering(
            num_trajectories=len(trajectories),
            num_commits=len(filtered),
            trajectory_type=trajectory_type
        )

        return trajectories

    def _map_trajectory_type(self, type_str: str) -> TrajectoryType:
        """Map string to TrajectoryType enum.

        Args:
            type_str: String type

        Returns:
            TrajectoryType enum
        """
        mapping = {
            "feature": TrajectoryType.FEATURE_DEVELOPMENT,
            "bug_fix": TrajectoryType.BUG_FIX,
            "refactor": TrajectoryType.REFACTORING,
        }
        return mapping.get(type_str, TrajectoryType.MIXED)

    def group_by_similarity(
        self,
        commits: List[ClassifiedCommit],
        keywords: List[str]
    ) -> List[List[ClassifiedCommit]]:
        """Group commits by keyword similarity.

        Args:
            commits: List of classified commits
            keywords: Keywords to match against

        Returns:
            List of commit groups
        """
        groups: Dict[str, List[ClassifiedCommit]] = {}

        for commit in commits:
            # Find matching keywords
            commit_keywords_lower = [k.lower() for k in commit.keywords]
            matching_keywords = [
                kw for kw in keywords
                if kw.lower() in commit_keywords_lower
            ]

            if not matching_keywords:
                # Use "other" as default group
                group_key = "other"
            else:
                group_key = matching_keywords[0]

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(commit)

        return list(groups.values())
```

**Step 4: Update clustering __init__.py**

Modify `ai_analysis/clustering/__init__.py`:

```python
"""Trajectory clustering module."""

from ai_analysis.clustering.trajectory_cluster import TrajectoryClusterer

__all__ = [
    "TrajectoryClusterer"
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_trajectory_cluster.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add ai_analysis/clustering/ tests/ai_analysis/test_trajectory_cluster.py
git commit -m "feat: add trajectory clustering for related commits"
```

---

## Task 10: Semantic Matcher (Helper)

**Files:**
- Create: `ai_analysis/clustering/semantic_matcher.py`
- Test: `tests/ai_analysis/test_semantic_matcher.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_semantic_matcher.py`:

```python
import pytest

def test_semantic_similarity():
    """Verify semantic similarity calculation"""
    from ai_analysis.clustering.semantic_matcher import SemanticMatcher

    matcher = SemanticMatcher()

    # Similar subjects
    similarity = matcher.compute_similarity(
        "Add user authentication",
        "Implement user login"
    )

    assert similarity > 0  # Should have some similarity

def test_semantic_matching_threshold():
    """Verify threshold-based matching"""
    from ai_analysis.clustering.semantic_matcher import SemanticMatcher
    from ai_analysis.models.commit import ClassifiedCommit, CommitType

    matcher = SemanticMatcher(threshold=0.3)

    commit1 = ClassifiedCommit(
        commit_hash="abc1",
        subject="Add authentication",
        author="Test",
        timestamp=123456,
        diff="+ auth",
        commit_type=CommitType.FEATURE,
        keywords=["auth"]
    )

    commit2 = ClassifiedCommit(
        commit_hash="abc2",
        subject="Add login support",
        author="Test",
        timestamp=123457,
        diff="+ login",
        commit_type=CommitType.FEATURE,
        keywords=["login"]
    )

    # Check if they match above threshold
    matches = matcher.are_related(commit1, commit2)
    assert isinstance(matches, bool)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_semantic_matcher.py -v`
Expected: FAIL with import error

**Step 3: Create semantic matcher**

Create `ai_analysis/clustering/semantic_matcher.py`:

```python
"""Semantic similarity matching for commits."""

from typing import List, Set
from difflib import SequenceMatcher
from ai_analysis.models.commit import ClassifiedCommit


class SemanticMatcher:
    """Compute semantic similarity between commits."""

    def __init__(self, threshold: float = 0.5):
        """Initialize semantic matcher.

        Args:
            threshold: Similarity threshold for considering commits related
        """
        self.threshold = threshold

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def are_related(
        self,
        commit1: ClassifiedCommit,
        commit2: ClassifiedCommit
    ) -> bool:
        """Check if two commits are semantically related.

        Args:
            commit1: First commit
            commit2: Second commit

        Returns:
            True if commits are related above threshold
        """
        # Check subject similarity
        subject_sim = self.compute_similarity(commit1.subject, commit2.subject)

        # Check keyword overlap
        keywords1 = set(k.lower() for k in commit1.keywords)
        keywords2 = set(k.lower() for k in commit2.keywords)

        if not keywords1 or not keywords2:
            keyword_sim = 0
        else:
            intersection = keywords1 & keywords2
            union = keywords1 | keywords2
            keyword_sim = len(intersection) / len(union) if union else 0

        # Combined score
        combined_sim = 0.7 * subject_sim + 0.3 * keyword_sim

        return combined_sim >= self.threshold

    def find_related_commits(
        self,
        target: ClassifiedCommit,
        commits: List[ClassifiedCommit]
    ) -> List[ClassifiedCommit]:
        """Find commits related to a target commit.

        Args:
            target: Target commit
            commits: List of commits to search

        Returns:
            List of related commits
        """
        related = []

        for commit in commits:
            if commit.commit_hash == target.commit_hash:
                continue

            if self.are_related(target, commit):
                related.append(commit)

        return related
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_semantic_matcher.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ai_analysis/clustering/semantic_matcher.py tests/ai_analysis/test_semantic_matcher.py
git commit -m "feat: add semantic similarity matcher for commits"
```

---

## Task 11: Prompt Templates (DSPy-style)

**Files:**
- Create: `ai_analysis/llm/prompts.py`
- Test: `tests/ai_analysis/test_prompts.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_prompts.py`:

```python
def test_classification_prompt_template():
    """Verify classification prompt can be formatted"""
    from ai_analysis.llm.prompts import ClassificationPrompt

    prompt = ClassificationPrompt()
    formatted = prompt.format(
        subject="Add authentication",
        diff="+ code",
        author="Test"
    )

    assert "Add authentication" in formatted
    assert "Test" in formatted
    assert "classification" in formatted.lower()

def test_clustering_prompt_template():
    """Verify clustering prompt can be formatted"""
    from ai_analysis.llm.prompts import ClusteringPrompt
    from ai_analysis.models.commit import ClassifiedCommit, CommitType

    prompt = ClusteringPrompt()

    commits = [
        ClassifiedCommit(
            commit_hash="abc1",
            subject="Add auth",
            author="Test",
            timestamp=123456,
            diff="+ code",
            commit_type=CommitType.FEATURE,
            keywords=["auth"]
        )
    ]

    formatted = prompt.format(commits)

    assert "Add auth" in formatted
    assert "trajectory" in formatted.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_prompts.py -v`
Expected: FAIL with import error

**Step 3: Create prompt templates**

Create `ai_analysis/llm/prompts.py`:

```python
"""Prompt templates for LLM interactions (DSPy-style)."""

from typing import List, Dict, Any
from ai_analysis.models.commit import ClassifiedCommit


class ClassificationPrompt:
    """Prompt template for commit classification."""

    TEMPLATE = """You are a Git commit classifier. Analyze the following commit and classify it.

Commit Subject: {subject}
Author: {author}
Timestamp: {timestamp}

Code Changes (diff):
{diff}

Classify this commit as one of: {commit_types}.

Consider the following guidelines:
- "feature": New functionality, features, capabilities
- "bug_fix": Bug fixes, error corrections, issue resolutions
- "refactor": Code restructuring without behavior change
- "docs": Documentation changes only
- "test": Test additions or modifications

Respond with JSON format only:
{{"type": "classification", "confidence": 0.0-1.0, "keywords": ["key1", "key2"], "reasoning": "brief explanation"}}"""

    def __init__(self, commit_types: List[str] = None):
        """Initialize classification prompt.

        Args:
            commit_types: List of valid commit types
        """
        self.commit_types = commit_types or [
            "feature", "bug_fix", "refactor", "docs", "test"
        ]

    def format(
        self,
        subject: str,
        diff: str,
        author: str,
        timestamp: int = 0
    ) -> str:
        """Format the prompt with commit data.

        Args:
            subject: Commit subject
            diff: Git diff
            author: Commit author
            timestamp: Unix timestamp

        Returns:
            Formatted prompt string
        """
        # Truncate diff if too long
        diff_truncated = diff[:3000] if len(diff) > 3000 else diff

        return self.TEMPLATE.format(
            subject=subject,
            author=author,
            timestamp=timestamp,
            diff=diff_truncated,
            commit_types=", ".join(self.commit_types)
        )


class KeywordExtractionPrompt:
    """Prompt template for keyword extraction."""

    TEMPLATE = """Extract {max_keywords} important keywords from the following text.

Text: {text}

Guidelines:
- Keywords should be technical terms relevant to software development
- Focus on functionality, components, or domain concepts
- Avoid common words like "add", "fix", "update"
- Each keyword should be 1-3 words

Respond with JSON format only:
{{"keywords": ["keyword1", "keyword2", "keyword3"]}}"""

    def format(self, text: str, max_keywords: int = 5) -> str:
        """Format the prompt.

        Args:
            text: Text to extract from
            max_keywords: Maximum keywords

        Returns:
            Formatted prompt
        """
        text_truncated = text[:1000] if len(text) > 1000 else text

        return self.TEMPLATE.format(
            text=text_truncated,
            max_keywords=max_keywords
        )


class ClusteringPrompt:
    """Prompt template for trajectory clustering."""

    TEMPLATE = """You are a trajectory analyzer. Given the following commits, group related ones into trajectories.

Commits:
{commit_list}

Group commits that belong to the same feature development or bug fix process.
Consider:
- Commits working on the same component or feature
- Commits with similar keywords or themes
- Temporal proximity (commits close in time are often related)

Create meaningful trajectories with descriptive titles and summaries.

Respond with JSON format only:
{{"trajectories": [{{"title": "...", "description": "...", "commit_indices": [0, 1, 2]}}]}}"""

    def format(self, commits: List[ClassifiedCommit]) -> str:
        """Format the prompt with commits.

        Args:
            commits: List of classified commits

        Returns:
            Formatted prompt
        """
        commit_list = ""
        for i, commit in enumerate(commits[:25]):  # Limit for context
            commit_list += f"\n{i}. Commit: {commit.commit_hash[:8]}\n"
            commit_list += f"   Subject: {commit.subject}\n"
            commit_list += f"   Type: {commit.commit_type.value}\n"
            commit_list += f"   Keywords: {', '.join(commit.keywords)}\n"

        return self.TEMPLATE.format(commit_list=commit_list)


class TrajectorySummaryPrompt:
    """Prompt template for generating trajectory summaries."""

    TEMPLATE = """Generate a concise summary of the following trajectory.

Trajectory: {title}
Description: {description}
Number of commits: {num_commits}

Commits:
{commits}

Generate a 2-3 sentence summary of what this trajectory accomplished."""

    def format(
        self,
        title: str,
        description: str,
        commits: List[ClassifiedCommit]
    ) -> str:
        """Format the prompt.

        Args:
            title: Trajectory title
            description: Trajectory description
            commits: List of commits in trajectory

        Returns:
            Formatted prompt
        """
        commit_list = "\n".join([
            f"- {c.subject}" for c in commits[:10]
        ])

        return self.TEMPLATE.format(
            title=title,
            description=description,
            num_commits=len(commits),
            commits=commit_list
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_prompts.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ai_analysis/llm/prompts.py tests/ai_analysis/test_prompts.py
git commit -m "feat: add DSPy-style prompt templates"
```

---

## Task 12: Main AI Analysis Orchestrator

**Files:**
- Create: `ai_analysis/orchestrator.py`
- Modify: `ai_analysis/__init__.py`
- Test: `tests/ai_analysis/test_orchestrator.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_orchestrator.py`:

```python
import pytest
from unittest.mock import patch, Mock

def test_orchestrator_analyzes_repository():
    """Verify orchestrator analyzes repository end-to-end"""
    from ai_analysis.orchestrator import AIAnalyzer
    from ai_analysis.models.commit import ClassifiedCommit, CommitType

    analyzer = AIAnalyzer()

    # Mock commits from GitDataFetcher
    raw_commits = [
        {
            "commit": "abc123",
            "subject": "Add authentication",
            "author": "Test",
            "timestamp": 123456,
            "diff": "+ def auth():"
        }
    ]

    with patch.object(analyzer.classifier, 'classify_batch') as mock_classify:
        mock_classify.return_value = [
            ClassifiedCommit(
                commit_hash="abc123",
                subject="Add authentication",
                author="Test",
                timestamp=123456,
                diff="+ def auth():",
                commit_type=CommitType.FEATURE,
                confidence=0.9,
                keywords=["auth"]
            )
        ]

        with patch.object(analyzer.clusterer, 'cluster') as mock_cluster:
            from ai_analysis.models.trajectory import Trajectory
            mock_cluster.return_value = [
                Trajectory(
                    trajectory_id="traj_1",
                    trajectory_type="feature",
                    title="Authentication",
                    description="Auth feature",
                    commit_hashes=["abc123"]
                )
            ]

            result = analyzer.analyze_repository(raw_commits)

            assert "commits" in result
            assert "trajectories" in result
            assert len(result["commits"]) == 1

def test_orchestrator_classifies_only():
    """Verify orchestrator can classify without clustering"""
    from ai_analysis.orchestrator import AIAnalyzer

    analyzer = AIAnalyzer()

    raw_commits = [
        {
            "commit": "abc123",
            "subject": "Add feature",
            "author": "Test",
            "timestamp": 123456,
            "diff": "+ code"
        }
    ]

    with patch.object(analyzer.classifier, 'classify_batch') as mock_classify:
        from ai_analysis.models.commit import ClassifiedCommit, CommitType
        mock_classify.return_value = [
            ClassifiedCommit(
                commit_hash="abc123",
                subject="Add feature",
                author="Test",
                timestamp=123456,
                diff="+ code",
                commit_type=CommitType.FEATURE,
                confidence=0.8,
                keywords=[]
            )
        ]

        result = analyzer.classify_commits(raw_commits)

        assert len(result) == 1
        assert result[0].commit_type == CommitType.FEATURE
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ai_analysis/test_orchestrator.py -v`
Expected: FAIL with import error

**Step 3: Create orchestrator**

Create `ai_analysis/orchestrator.py`:

```python
"""Main orchestrator for AI Analysis operations."""

from typing import List, Dict, Any, Optional
from ai_analysis.config import get_config
from ai_analysis.classifiers import CommitClassifier
from ai_analysis.clustering import TrajectoryClusterer
from ai_analysis.models.commit import ClassifiedCommit
from ai_analysis.models.trajectory import Trajectory
from ai_analysis.tracking import get_tracker


class AIAnalyzer:
    """Main orchestrator for AI-powered repository analysis."""

    def __init__(self, config=None):
        """Initialize the AI analyzer.

        Args:
            config: Optional configuration object
        """
        self.config = config or get_config()
        self.classifier = CommitClassifier(self.config)
        self.clusterer = TrajectoryClusterer(self.config)
        self.tracker = get_tracker()

    def classify_commits(
        self,
        raw_commits: List[Dict[str, Any]],
        show_progress: bool = False
    ) -> List[ClassifiedCommit]:
        """Classify a list of raw commits.

        Args:
            raw_commits: List of commit dictionaries from GitDataFetcher
            show_progress: Whether to show progress

        Returns:
            List of ClassifiedCommit objects
        """
        return self.classifier.classify_batch(raw_commits, show_progress)

    def cluster_trajectories(
        self,
        commits: List[ClassifiedCommit],
        trajectory_type: str = "all"
    ) -> List[Trajectory]:
        """Cluster commits into trajectories.

        Args:
            commits: List of classified commits
            trajectory_type: Type to filter ('all', 'feature', 'bug_fix', etc.)

        Returns:
            List of Trajectory objects
        """
        return self.clusterer.cluster(commits, trajectory_type)

    def analyze_repository(
        self,
        raw_commits: List[Dict[str, Any]],
        cluster_by_type: Optional[str] = "all"
    ) -> Dict[str, Any]:
        """Full analysis: classify and cluster.

        Args:
            raw_commits: List of raw commit dictionaries
            cluster_by_type: Trajectory type to cluster ('all', 'feature', etc.)

        Returns:
            Dictionary with 'commits' and 'trajectories'
        """
        # Start MLflow run
        if self.tracker.enabled:
            self.tracker.start_run("repository_analysis")

        try:
            # Step 1: Classify commits
            classified = self.classify_commits(raw_commits)

            # Step 2: Cluster into trajectories
            trajectories = self.cluster_trajectories(classified, cluster_by_type)

            # Step 3: Compile statistics
            stats = self._compute_statistics(classified, trajectories)

            result = {
                "commits": [c.to_dict() for c in classified],
                "trajectories": [t.to_dict() for t in trajectories],
                "statistics": stats
            }

            return result

        finally:
            if self.tracker.enabled:
                self.tracker.end_run()

    def _compute_statistics(
        self,
        commits: List[ClassifiedCommit],
        trajectories: List[Trajectory]
    ) -> Dict[str, Any]:
        """Compute analysis statistics.

        Args:
            commits: List of classified commits
            trajectories: List of trajectories

        Returns:
            Statistics dictionary
        """
        # Count by type
        type_counts = {}
        for commit in commits:
            ct = commit.commit_type.value
            type_counts[ct] = type_counts.get(ct, 0) + 1

        # Average confidence
        confidences = [c.confidence for c in commits if c.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_commits": len(commits),
            "type_distribution": type_counts,
            "total_trajectories": len(trajectories),
            "average_confidence": avg_confidence,
            "commits_in_trajectories": sum(t.commit_count for t in trajectories)
        }

    def get_feature_trajectories(self, commits: List[ClassifiedCommit]) -> List[Trajectory]:
        """Get feature development trajectories.

        Args:
            commits: List of classified commits

        Returns:
            List of feature trajectories
        """
        return self.cluster_trajectories(commits, "feature")

    def get_bug_fix_trajectories(self, commits: List[ClassifiedCommit]) -> List[Trajectory]:
        """Get bug fix trajectories.

        Args:
            commits: List of classified commits

        Returns:
            List of bug fix trajectories
        """
        return self.cluster_trajectories(commits, "bug_fix")
```

**Step 4: Update ai_analysis __init__.py**

Modify `ai_analysis/__init__.py`:

```python
"""AI Analysis Module for GitTracer.

This module provides LLM-powered commit classification and trajectory
clustering capabilities.
"""

__version__ = "0.1.0"

from ai_analysis import llm, classifiers, clustering, tracking, models
from ai_analysis.orchestrator import AIAnalyzer

__all__ = [
    "llm",
    "classifiers",
    "clustering",
    "tracking",
    "models",
    "AIAnalyzer"
]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_orchestrator.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add ai_analysis/orchestrator.py ai_analysis/__init__.py tests/ai_analysis/test_orchestrator.py
git commit -m "feat: add main AI analysis orchestrator"
```

---

## Task 13: Update Requirements

**Files:**
- Modify: `requirements.txt`

**Step 1: Write the failing test**

Update `tests/test_requirements.py` (or create new):

```python
import os
import pytest

def test_requirements_has_ai_dependencies():
    """Verify requirements includes AI dependencies"""
    with open("requirements.txt") as f:
        content = f.read().lower()

    required = [
        "litellm",
        "mlflow"
    ]

    for dep in required:
        assert dep in content, f"{dep} not found in requirements.txt"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_requirements.py -v`
Expected: FAIL (dependencies not present)

**Step 3: Update requirements.txt**

Modify `requirements.txt`:

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

# AI/LLM Dependencies
litellm>=1.40.0
dspy-ai>=2.4.0
mlflow>=2.10.0

# Clustering helpers
scikit-learn>=1.3.0
numpy>=1.24.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_requirements.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add requirements.txt tests/test_requirements.py
git commit -m "feat: add AI/LLM dependencies to requirements"
```

---

## Task 14: Environment Variable Documentation

**Files:**
- Create: `.env.example`
- Modify: `README.md`

**Step 1: Create .env.example**

Create `.env.example`:

```bash
# LLM Provider Configuration
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
API_KEY=your_api_key_here
API_BASE=

# LLM Parameters
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=500

# MLflow Configuration
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=gittracer_commit_analysis

# Classification Settings
# COMMIT_TYPES=feature,bug_fix,refactor,docs,test

# Clustering Settings
# SIMILARITY_THRESHOLD=0.7
# MAX_TRAJECTORY_SIZE=50
```

**Step 2: Update README.md**

Modify `README.md` - add AI Analysis section:

```markdown
# GitTracer - SWE Trajectory Analysis Platform

...

## AI Analysis Module

GitTracer uses LLMs to automatically classify commits and cluster them into trajectories.

### Configuration

Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:

- `LLM_PROVIDER`: LLM provider (openai, anthropic, etc.)
- `API_KEY`: API key for the LLM provider

Optional:
- `MODEL_NAME`: Model to use (default: gpt-4o-mini)
- `MLFLOW_ENABLED`: Enable MLflow tracking (default: true)
- `MLFLOW_TRACKING_URI`: MLflow tracking URI (default: sqlite:///mlflow.db)

### Usage

```python
from ai_analysis import AIAnalyzer

analyzer = AIAnalyzer()

# Analyze a repository
result = analyzer.analyze_repository(raw_commits)

# Access results
for trajectory in result["trajectories"]:
    print(f"{trajectory['title']}: {trajectory['commit_count']} commits")
```
```

**Step 3: Commit**

```bash
git add .env.example README.md
git commit -m "docs: add environment variable documentation"
```

---

## Task 15: Integration Test - End-to-End

**Files:**
- Create: `tests/ai_analysis/test_integration.py`

**Step 1: Write the failing test**

Create `tests/ai_analysis/test_integration.py`:

```python
import pytest
from unittest.mock import patch, MagicMock

def test_full_ai_analysis_integration():
    """Verify full AI analysis pipeline works"""
    from ai_analysis import AIAnalyzer

    analyzer = AIAnalyzer()

    # Sample raw commits
    raw_commits = [
        {
            "commit": "abc123",
            "subject": "Add user authentication",
            "author": "Developer",
            "timestamp": 1234567890,
            "diff": "+ def authenticate(user, password):\n+     return verify(user, password)"
        },
        {
            "commit": "def456",
            "subject": "Fix authentication bug",
            "author": "Developer",
            "timestamp": 1234567900,
            "diff": "-     return verify(user, password)\n+     return verify(user, password) and check_db(user)"
        }
    ]

    with patch.object(analyzer.classifier.llm_client, 'classify_commit') as mock_classify:
        # Mock classification responses
        mock_classify.side_effect = [
            ("feature", 0.9, ["authentication", "user"]),
            ("bug_fix", 0.85, ["authentication", "bug"])
        ]

        with patch.object(analyzer.clusterer.llm_client, 'cluster_commits') as mock_cluster:
            mock_cluster.return_value = [
                {
                    "title": "Authentication System",
                    "description": "User authentication implementation and fixes",
                    "commit_indices": [0, 1]
                }
            ]

            result = analyzer.analyze_repository(raw_commits)

            # Verify structure
            assert "commits" in result
            assert "trajectories" in result
            assert "statistics" in result

            # Verify commits classified
            assert len(result["commits"]) == 2
            assert result["commits"][0]["commit_type"] == "feature"
            assert result["commits"][1]["commit_type"] == "bug_fix"

            # Verify trajectories created
            assert len(result["trajectories"]) == 1
            assert result["trajectories"][0]["title"] == "Authentication System"

def test_mlflow_tracking_integration():
    """Verify MLflow tracking works end-to-end"""
    from ai_analysis import AIAnalyzer
    from ai_analysis.config import Config

    # Enable MLflow for test
    config = Config(
        api_key="test-key",
        mlflow_enabled=True,
        mlflow_tracking_uri="file:///tmp/mlflow_test"
    )

    analyzer = AIAnalyzer(config)

    raw_commits = [
        {
            "commit": "abc123",
            "subject": "Test commit",
            "author": "Test",
            "timestamp": 123456,
            "diff": "+ code"
        }
    ]

    with patch.object(analyzer.classifier.llm_client, 'classify_commit') as mock_classify:
        mock_classify.return_value = ("feature", 0.9, ["test"])

        with patch('ai_analysis.tracking.mlflow_tracker.mlflow') as mock_mlflow:
            result = analyzer.analyze_repository(raw_commits)

            # Verify MLflow was called
            assert mock_mlflow.start_run.called or mock_mlflow.log_metric.called
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/ai_analysis/test_integration.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/ai_analysis/test_integration.py
git commit -m "test: add end-to-end integration tests"
```

---

## Task 16: Final Verification

**Files:**
- None (verification only)

**Step 1: Run all AI analysis tests**

Run: `pytest tests/ai_analysis/ -v --cov=ai_analysis`
Expected: All tests PASS with reasonable coverage

**Step 2: Verify package structure**

Run: `python -c "from ai_analysis import AIAnalyzer; print('AI Analysis module loaded successfully')"`
Expected: Prints "AI Analysis module loaded successfully"

**Step 3: Verify all exports work**

Run: `python -c "from ai_analysis import *; print(dir())"`
Expected: Shows all exported classes

**Step 4: Check git status**

Run: `git status`
Expected: No uncommitted changes (except __pycache__)

**Step 5: Final commit**

If any files are untracked:

```bash
git add .
git commit -m "chore: finalize AI analysis module implementation"
```

---

## Implementation Complete Checklist

- [x] Package structure set up
- [x] Configuration module with environment variables
- [x] Data models (ClassifiedCommit, Trajectory)
- [x] Base LLM client interface
- [x] LiteLLM client implementation
- [x] MLflow tracking integration
- [x] Commit classifier with batch support
- [x] Keyword extractor
- [x] Trajectory clusterer
- [x] Semantic matcher
- [x] Prompt templates (DSPy-style)
- [x] Main AI orchestrator
- [x] Requirements updated with AI dependencies
- [x] Environment documentation (.env.example)
- [x] Integration tests
- [x] All tests passing

---

## Next Steps (Future Enhancements)

1. **DSPy Integration**: Replace manual prompts with DSPy signatures and optimizers
2. **Batch Processing**: Add async batch processing for large repositories
3. **Caching**: Implement caching for LLM responses to reduce API costs
4. **Local Models**: Add support for local LLMs (Ollama, llama.cpp)
5. **Cost Estimation**: Add token counting and cost estimation before API calls
6. **Multi-language Support**: Add support for non-English commit messages
