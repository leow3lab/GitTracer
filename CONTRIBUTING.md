# 开发规范 (Development Guidelines)

## 代码风格 (Code Style)

本项目使用 **Black** 进行 Python 代码格式化。

### 自动检查

每次 `git commit` 时，pre-commit hook 会自动运行 black 检查：
- 如果代码不符合规范，提交将被拒绝
- 请运行 `./format.sh` 或手动格式化后再提交

### 手动格式化

```bash
# 方式 1: 使用脚本
./format.sh

# 方式 2: 使用 uv
uv run black frontend/ tests/ app.py

# 方式 3: 使用 venv
.venv/bin/black frontend/ tests/ app.py
```

### 检查而不修改

```bash
# 只检查，不修改
.venv/bin/black --check frontend/ tests/ app.py

# 查看差异
.venv/bin/black --diff frontend/ tests/ app.py
```

## Black 配置

`pyproject.toml` 中的配置：
```toml
[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311', 'py312']
```

## Git Hooks

### Pre-commit Hook

自动检查代码风格：
```bash
.git/hooks/pre-commit
```

### 禁用 Hook (临时)

```bash
# 跳过 pre-commit hook
git commit --no-verify -m "message"
```

## 提交前检查清单

- [ ] 代码通过 black 格式化检查
- [ ] 所有测试通过 (`pytest tests/`)
- [ ] 代码符合项目规范
- [ ] 提交信息清晰明确

## 常用命令

```bash
# 格式化代码
./format.sh

# 运行测试
pytest tests/

# 检查代码风格
black --check frontend/ tests/

# Lint 检查
flake8 frontend/ tests/

# 类型检查
mypy frontend/
```
