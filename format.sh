#!/bin/bash
# 格式化代码脚本

echo "Formatting Python code with black..."

if command -v uv &> /dev/null; then
    uv run black frontend/ tests/ app.py
else
    .venv/bin/black frontend/ tests/ app.py
fi

echo "✓ Code formatted!"
