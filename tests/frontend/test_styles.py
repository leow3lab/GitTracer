"""Tests for sketch styles."""

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
    assert ".sketch-border" in content or ".hand-drawn" in content or ".sketch-card" in content


def test_sketch_css_has_paper_background():
    """Verify CSS has paper-like background"""
    with open(CSS_PATH) as f:
        content = f.read()
    assert any(x in content for x in ["#f9f7f1", "#faf8f5", "paper", "parchment"])
