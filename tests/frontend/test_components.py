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


def test_repo_config_component():
    """Verify repository configuration component"""
    from frontend.layouts.components.repo_config import create_repo_config
    component = create_repo_config()
    assert component is not None
    # Should have input fields for repo path, branch, and top_k
    assert hasattr(component, 'children')


def test_commit_table_component():
    """Verify commit table component"""
    from frontend.layouts.components.commit_table import create_commit_table
    sample_data = [
        {'idx': 1, 'commit': 'abc123', 'subject': 'Test commit', 'author': 'Test Author'}
    ]
    table = create_commit_table(sample_data)
    assert table is not None


def test_trajectory_view_component():
    """Verify trajectory view component"""
    from frontend.layouts.components.trajectory_view import create_trajectory_view
    view = create_trajectory_view([])
    assert view is not None
