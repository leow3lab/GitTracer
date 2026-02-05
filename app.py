mport subprocess
import json
import sys
import os
import time


import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime

--- 逻辑类（集成自您的原始代码） ---

class GitCommit:
    def init(self, data, idx):
        self.idx = idx
        self.commit_id = data.get("commit", "Unknown")
        self.author = data.get("author", "Unknown")
        self.email = data.get("email", "Unknown")
        self.timestamp = data.get("timestamp", 0)
        self.subject = data.get("subject", "No Subject")
        self.diff = data.get("diff", "")

    def to_md(self):
        dt_object = datetime.fromtimestamp(self.timestamp)
        formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        md_body = [
            f"### [{self.idx}] Commit: {self.commit_id}",
            f"Subject: {self.subject}  ",
            f"Author: {self.author} (<{self.email}>)  ",
            f"Date: {formatted_time}",
            "\n---",
            "\n#### Code Changes",
            "```diff",
            self.diff if self.diff.strip() else "No code changes in this commit.",
            "```"
        ]
        return "\n".join(md_body)

class GitDataFetcher:
    def init(self, repo_path, branch="main", top_k=100):
        self.repo_path = os.path.abspath(repo_path)
        self.branch = branch
        self.top_k = top_k

    def _run_git(self, args):
        return subprocess.check_output(
            ["git"] + args, 
            cwd=self.repo_path, 
            encoding='utf-8', 
            errors='ignore'
        )

    def fetch_commits(self):
        log_format = "%H|%an|%ae|%at|%s"
        try:
            log_output = self._run_git([
                "log", self.branch, f"--pretty=format:{log_format}", f"-n", str(self.top_k)
            ])
        except Exception as e:
            return None, str(e)

        commits_list = []
        lines = [l for l in log_output.split('\n') if l]
        
        for i, line in enumerate(lines):
            current_idx = i + 1
            parts = line.split('|')
            if len(parts) < 5: continue
            h, name, email, time_val, subj = parts[:5]
            
            # 快速获取 diff
            diff_content = self._run_git(["show", "--patch", "-m", "--pretty=format:", h])

            commits_list.append({
                "idx": current_idx,
                "commit": h,
                "author": name,
                "email": email,
                "timestamp": int(time_val),
                "subject": subj,
                "diff": diff_content
            })
        
        return commits_list, None

--- Dash UI 构建 ---

app = dash.Dash(name, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Git Repository Trajectory Analyzer", className="text-center my-4 text-primary"), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("仓库设置 (Repository Settings)"),
                dbc.CardBody([
                    dbc.Label("本地仓库路径:"),
                    dbc.Input(id="repo-path", placeholder="E.g. /path/to/your/repo", type="text", value=os.getcwd()),
                    dbc.Label("分支名称:", className="mt-2"),
                    dbc.Input(id="branch-name", value="main", type="text"),
                    dbc.Label("抓取数量 (Top K):", className="mt-2"),
                    dbc.Input(id="top-k", value=20, type="number", min=1),
                    dbc.Button("分析仓库", id="btn-analyze", color="primary", className="mt-3 w-100"),
                ])
            ], className="mb-4"),
            
            html.Div(id="status-message")
        ], width=4),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="提交列表 (Commit History)", tab_id="tab-list", children=[
                    html.Div(id="table-container", className="mt-3")
                ]),
                dbc.Tab(label="代码详情 (Code Detail)", tab_id="tab-detail", children=[
                    html.Div(id="detail-container", className="mt-3 p-3 border rounded bg-light", 
                             style={"maxHeight": "700px", "overflowY": "auto"})
                ]),
            ], id="tabs", active_tab="tab-list")
        ], width=8)
    ]),
    
    # 存储获取到的原始数据
    dcc.Store(id='stored-data')
], fluid=True)

--- 回调逻辑 ---

@app.callback(
    [Output("stored-data", "data"),
     Output("status-message", "children"),
     Output("table-container", "children")],
    [Input("btn-analyze", "n_clicks")],
    [State("repo-path", "state"), State("repo-path", "value"),
     State("branch-name", "value"), State("top-k", "value")]
)
def update_analysis(n_clicks, state, path, branch, top_k):
    if n_clicks is None:
        return None, "", "请输入路径并点击分析"
    
    if not os.path.exists(path):
        return None, dbc.Alert("路径不存在！", color="danger"), ""
    
    fetcher = GitDataFetcher(path, branch, top_k)
    data, error = fetcher.fetch_commits()
    
    if error:
        return None, dbc.Alert(f"错误: {error}", color="danger"), ""

    df = pd.DataFrame(data)
    # 创建表格
    table = dash_table.DataTable(
        id='commit-table',
        columns=[
            {"name": "ID", "id": "idx"},
            {"name": "Hash", "id": "commit"},
            {"name": "Subject", "id": "subject"},
            {"name": "Author", "id": "author"}
        ],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        row_selectable='single',
        page_size=10
    )
    
    return data, dbc.Alert(f"成功加载 {len(data)} 个提交", color="success"), table

@app.callback(
    [Output("detail-container", "children"), Output("tabs", "active_tab")],
    [Input("commit-table", "derived_virtual_selected_rows"), Input("stored-data", "data")],
    prevent_initial_call=True
)
def show_details(selected_rows, stored_data):
    if not selected_rows or not stored_data:
        return "在左侧选择一个提交以查看详情", "tab-list"
    
    row_idx = selected_rows[0]
    # 查找选中的数据项
    commit_data = stored_data[row_idx]
    commit_obj = GitCommit(commit_data, idx=commit_data['idx'])
    
    return dcc.Markdown(commit_obj.to_md(), dangerously_allow_html=True), "tab-detail"

if name == "main":
    app.run(debug=True)