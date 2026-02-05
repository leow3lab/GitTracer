# 开发日志（command-log）

## [2026-02-05 00:00]

- **实现功能**：补充 `GitDataFetcher` 支持“传入 URL 自动 clone/fetch 到本地缓存目录”的详细设计文档（仅文档），用于后续实现。
- **遇到的问题**：
  - 当前仓库没有 `docs/logs/command-log.md` 与 `docs/TODO.md`，与既定开发流程不一致。
  - 当前 `GitDataFetcher` 直接把 `repo_path` 当本地路径执行 `git log/show`，URL 会导致 `cwd` 目录不存在而失败。
- **解决方案**：
  - 新增设计文档 `docs/plans/2026-02-05-gitdatafetcher-auto-clone-from-url.md`，明确 URL 识别、缓存目录命名、clone/fetch/checkout 及错误信息策略，并给出 2-3 条验收测试用例方向。
  - 创建本日志与 `docs/TODO.md`，把该功能列为待实现任务。
- **下一步计划**：
  - 在获得源代码改动授权后，按设计将 `_ensure_local_repo` 等逻辑落到 `app.py`，并补充对应单测（若允许改测试代码）。

## [2026-02-05 00:10]

- **实现功能**：在 `app.py` 的 `GitDataFetcher` 中实现“repo_path 传入 Git URL 时自动 clone/fetch 到本地缓存目录（`./data/repos/...`）”逻辑，并尽力 checkout 到指定分支（不存在则回退到远端默认分支）。
- **遇到的问题**：
  - 需要在“repo 尚未存在时”执行 `git clone`（无 cwd），而后续 `git log/show` 仍需在本地目录下执行。
  - 分支名可能不存在（例如传了 `main`，仓库默认是 `master`）。
- **解决方案**：
  - 增加 `_is_git_url`、`_compute_cache_dir`、`_run_cmd`、`_default_branch`、`_ensure_local_repo`，在 `__init__` 中把 URL 输入转换为本地缓存目录。
  - checkout 失败时通过 `refs/remotes/origin/HEAD` 探测默认分支并回退。
- **下一步计划**：
  - 增补测试用例（尤其是 URL 识别、clone 复用缓存、分支回退路径）。

## [2026-02-05 00:20]

- **实现功能**：更新前端规划文档 `docs/plans/2025-02-04-frontend-dash-app.md` 中的仓库输入框文案，提示用户既可输入本地路径，也可输入 GitHub URL，并使用 `vllm`（`https://github.com/vllm-project/vllm`）作为示例；同时提醒 URL 模式需要注意网络访问（代理/VPN/防火墙）并提供基本排查建议。
- **遇到的问题**：原文案只描述“本地路径”，与当前后端已支持 URL clone/fetch 的能力不一致。
- **解决方案**：在 `Repository Configuration Component` 的示例代码片段中更新 Label/placeholder/help text。
- **下一步计划**：后续把提示信息同步到实际前端组件实现与校验逻辑里（如需）。

## [2026-02-05 15:10]

- **实现功能**：修复 Dash UI 报错 “nonexistent object in Output: `commit-table-container`”，通过在初始布局中预先渲染包含 `commit-table-container` 的 Tabs（`main-content-area` 默认 children 设为 tabs）。
- **遇到的问题**：`repo_callbacks` 的回调输出包含 `Output("commit-table-container", "children")`，但初始 layout 尚未渲染出该 id（依赖导航回调填充 `main-content-area`），导致前端 renderer 抛错。
- **解决方案**：在 `frontend/layouts/main_layout.py` 中创建 `initial_tabs`，让 `commit-table-container` 从首次渲染就存在，避免回调竞争/时序问题。
- **下一步计划**：将来可以进一步重构：由 `stored-commits` 驱动表格渲染，减少多个回调同时写同一块 UI 的耦合。

## [2026-02-05 15:20]

- **实现功能**：更新 UI 文案示例中的 vLLM 仓库 URL 为官方仓库 `https://github.com/vllm-project/vllm-ascend`，并同步到实际前端组件输入提示（placeholder/help text）与 `docs/plans/2025-02-04-frontend-dash-app.md`。
- **遇到的问题**：此前示例使用了 `vllm-project/vllm`，与期望示例不一致。
- **解决方案**：统一替换为 `vllm-project/vllm-ascend`，减少用户误解。

## [2026-02-05 15:30]

- **实现功能**：修复前端对 GitHub URL 的错误校验：`validate_repo_path()` 现在识别 Git URL 并直接放行，由 `GitDataFetcher` 负责 clone/fetch。
- **遇到的问题**：原逻辑使用 `os.path.exists()` 校验输入，导致 URL 被当作本地路径而报 “Path does not exist”。
- **解决方案**：在 `frontend/callbacks/repo_callbacks.py` 增加 `is_git_url()`（支持 https/ssh/git@... 等常见格式），URL 模式跳过本地 `.git` 校验。

## [2026-02-05 15:40]

- **实现功能**：修复 UI 提示 “GitDataFetcher not available”：将 `GitDataFetcher` 从 `app.py` 拆分到独立模块 `git_data.py`，前端回调直接从该模块导入，消除 `app.py` ↔ `frontend` 的循环导入。
- **遇到的问题**：`frontend/callbacks/repo_callbacks.py` 之前通过 `from app import GitDataFetcher` 导入，会触发 `app.py` 反向导入 `frontend.app`，形成循环依赖并导致 ImportError，从而使 `GitDataFetcher=None`。
- **解决方案**：新增 `git_data.py`（仅包含 `GitCommit`/`GitDataFetcher`），`app.py` 仅作为入口创建 Dash app；回调改为 `from git_data import GitDataFetcher`。

