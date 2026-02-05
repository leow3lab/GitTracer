"""Tests for frontend layout components."""

import pytest


def test_frontend_package_exists():
    """Verify frontend package structure is created"""
    from frontend import layouts, callbacks, styles
    assert hasattr(layouts, 'components')
    assert hasattr(callbacks, '__file__')
    assert hasattr(styles, '__file__')


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
