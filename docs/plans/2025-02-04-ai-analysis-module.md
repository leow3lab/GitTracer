# AI Analysis Module Implementation Plan

**Goal:** Build an AI-powered commit classification and trajectory clustering system using LLMs.

**Architecture:**
- LiteLLM for unified LLM API (OpenAI, Anthropic, etc.)
- DSPy framework for prompt optimization
- MLflow for tracking all LLM calls
- Modular: CommitClassifier → TrajectoryCluster → Orchestrator

**Tech Stack:**
- LiteLLM, DSPy, MLflow, scikit-learn, numpy/pandas

---

## Project Structure

```
ai_analysis/
├── config.py              # LLM/MLflow configuration
├── llm/
│   ├── base.py            # Base LLM client interface
│   ├── litellm_client.py  # LiteLLM implementation
│   └── prompts.py         # Prompt templates (DSPy)
├── classifiers/
│   ├── commit_classifier.py   # Main classifier
│   └── keyword_extractor.py   # Keyword extraction
├── clustering/
│   ├── trajectory_cluster.py  # Trajectory clustering
│   └── semantic_matcher.py    # Semantic similarity
├── tracking/
│   └── mlflow_tracker.py       # MLflow integration
├── models/
│   ├── commit.py          # ClassifiedCommit model
│   └── trajectory.py      # Trajectory model
└── orchestrator.py        # Main analyzer orchestrator
```

---

## Core Requirements

### 1. Configuration Module (`config.py`)
- Environment variables: `LLM_PROVIDER`, `MODEL_NAME`, `API_KEY`, `MLFLOW_ENABLED`
- Singleton pattern with `get_config()`
- Default model: `gpt-4o-mini`
- MLflow tracking: `sqlite:///mlflow.db`

### 2. Data Models
- `ClassifiedCommit`: commit_hash, subject, author, timestamp, diff, commit_type (enum), confidence, keywords
- `Trajectory`: trajectory_id, trajectory_type, title, description, commit_hashes, authors, timestamps
- Enums: `CommitType` (feature, bug_fix, refactor, docs, test, unknown), `TrajectoryType`

### 3. LLM Abstraction Layer
- `BaseLLMClient`: Abstract interface with `classify_commit()`, `extract_keywords()`, `cluster_commits()`
- `LiteLLMClient`: Implementation using litellm.completion
  - JSON response parsing with error handling
  - Diff truncation for token efficiency
- Prompt templates for classification, keyword extraction, clustering

### 4. Commit Classification
- `CommitClassifier.classify()`: Classify single commit via LLM
- `CommitClassifier.classify_batch()`: Batch classification with optional progress
- Returns: `List[ClassifiedCommit]` with type, confidence, keywords
- Classification types: feature, bug_fix, refactor, docs, test

### 5. Keyword Extraction
- `KeywordExtractor.extract()`: Extract keywords from text using LLM
- `KeywordExtractor.extract_from_commits()`: Aggregate and rank keywords from multiple commits
- Returns: Top N keywords by frequency

### 6. Trajectory Clustering
- `TrajectoryClusterer.cluster()`: Group related commits into trajectories
- Filter by trajectory_type (feature, bug_fix, all)
- Uses LLM semantic analysis to group commits
- Returns: `List[Trajectory]` with titles, descriptions, commit lists

### 7. Semantic Matching (Helper)
- `SemanticMatcher.compute_similarity()`: String similarity using SequenceMatcher
- `SemanticMatcher.are_related()`: Check if two commits are related (subject + keywords)
- `SemanticMatcher.find_related_commits()`: Find all commits related to a target

### 8. MLflow Tracking
- `MLflowTracker.log_classification()`: Log commit classification results
- `MLflowTracker.log_clustering()`: Log clustering metrics
- `MLflowTracker.start_run()` / `end_run()`: Run lifecycle
- `@track_llm_call` decorator for automatic tracking

### 9. Main Orchestrator
- `AIAnalyzer.classify_commits()`: Classify raw commits
- `AIAnalyzer.cluster_trajectories()`: Cluster commits into trajectories
- `AIAnalyzer.analyze_repository()`: Full pipeline (classify + cluster)
- `AIAnalyzer._compute_statistics()`: Type distribution, avg confidence, trajectory counts

### 10. Prompt Templates (DSPy-style)
- `ClassificationPrompt`: Commit classification with guidelines
- `KeywordExtractionPrompt`: Keyword extraction with technical term focus
- `ClusteringPrompt`: Trajectory grouping with temporal consideration
- `TrajectorySummaryPrompt`: Generate trajectory summaries

---

## Requirements Updates

```
# AI/LLM Dependencies
litellm>=1.40.0
dspy-ai>=2.4.0
mlflow>=2.10.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

---

## Environment Variables (`.env.example`)

```bash
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
API_KEY=your_api_key_here
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=500
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

---

## Usage Example

```python
from ai_analysis import AIAnalyzer

analyzer = AIAnalyzer()
result = analyzer.analyze_repository(raw_commits)

# Access results
for traj in result["trajectories"]:
    print(f"{traj['title']}: {traj['commit_count']} commits")
```
