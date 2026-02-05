"""Tests for frontend layout components."""

import pytest


def test_frontend_package_exists():
    """Verify frontend package structure is created"""
    from frontend import layouts, callbacks, styles
    assert hasattr(layouts, 'components')
    assert hasattr(callbacks, '__file__')
    assert hasattr(styles, '__file__')
