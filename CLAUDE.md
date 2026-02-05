CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project Overview

SWE 轨迹生成平台 - A platform for extracting and analyzing Software Engineering (SWE) trajectories from real open-source repositories (e.g., vLLM, SGLang). It analyzes Git commit histories to automatically identify, classify, and aggregate Feature development and Bug Fix trajectories, providing high-quality datasets for SWE Agent evaluation.

Architecture

Technology Stack
- Frontend: Dash (Plotly), Dash Bootstrap Components
- Backend: FastAPI, Uvicorn, GitPython
- AI/LLM: LiteLLM (for GPT-4/Claude 3.5), DSPy framework for agent development
- Tracking: MLflow (sqlite or remote server)
- Deployment: Docker-compose
Key Design Principles
- Frontend-Backend Separation: Dash frontend, FastAPI backend
- Visual Style: Hand-drawn/Sketch UI style (RoughViz or custom CSS like PaperCSS/Wired Elements)
- Traceability: All LLM calls must go through LiteLLM with MLflow lifecycle tracking
Core Components

GitDataFetcher (app.py:42)
Handles Git repository interaction using subprocess. Key responsibilities:
- Clone repositories and scan branches
- Extract commit metadata (hash, author, timestamp, subject)
- Extract diff content for each commit
- Persist commits as individual Markdown files (structure: {repo_name}/{branch}/{commit_hash}.md)
GitCommit (app.py:16)
Data model representing a single commit with methods to convert to Markdown format.

Data Storage Structure

API Endpoints (Planned)

- POST /api/fetch - Trigger Git clone and log scan
- POST /api/analyze - Trigger LLM classification and clustering
- GET /api/status/{task_id} - Query background task progress
- GET /api/trajectories - Get analyzed feature summaries
- GET /api/export/{id} - Export specific trajectory as Markdown bundle
AI Analysis Module (To Be Implemented)

The AI module should:
- Use LiteLLM for unified LLM calls across different providers
- Implement keyword extraction from commit messages
- Classify commits into: Feature, Bug Fix, Refactor, Docs, Test
- Trajectory Clustering: Link related commits into complete trajectories using LLM semantic analysis
- Track all prompts, token usage, and response times via MLflow
Trajectory Clustering Logic
Related commits (same feature or bug fix process) should be grouped into trajectories using LLM analysis of commit relationships.

Development Commands

Running the Application
The Dash app runs in debug mode by default.

Testing Git Operations
The app uses subprocess to call git commands directly. Ensure git is installed and accessible in PATH.

Non-Functional Requirements
- Performance: Initial scan of 1000 commits should complete in <30 seconds
- Concurrency: Support multiple users (use Celery or BackgroundTasks)
- Extensibility: Plugin-based analysis logic for future trajectory metrics