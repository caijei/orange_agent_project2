# orange_agent_project

本科毕设项目：基于 Agent 的脐橙知识问答系统。项目包含 FastAPI 后端、React/Vite 前端和本地脐橙知识库数据。

## 目录结构

```text
orange_agent_project/
├── backend/          # FastAPI 后端、RAG、数据库、智能体服务
├── frontend/         # React + Vite 前端
├── docs/             # 原始 PDF、docx 等资料
├── docs_clean/       # 清洗后的知识库资料
└── README.md
```

## 环境准备

建议使用：

- Python 3.10
- Node.js 20.19+ 或 22.12+
- npm 10+

本项目开发和测试时使用的是 Python 3.10，建议接手同学也安装 Python 3.10，避免依赖版本不兼容。

## 1. 获取项目

```powershell
git clone <项目仓库地址>
cd orange_agent_project
```

如果不是通过 Git 获取项目，也可以直接解压项目压缩包，然后进入项目根目录。

## 2. 配置后端环境变量

复制示例配置文件：

```powershell
copy backend\.env.example backend\.env
```

打开 `backend/.env`，把下面这些值改成自己的：

```text
DASHSCOPE_API_KEY=你的阿里云百炼 DashScope API Key
TAVILY_API_KEY=你的 Tavily API Key，没有可先留空，联网搜索会不可用
OLLAMA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OLLAMA_MODEL=你的账户可用模型的名称
```

说明：

- `DASHSCOPE_API_KEY`：用于大模型、图片理解、Embedding、Rerank，通常必须配置。
- `TAVILY_API_KEY`：用于联网搜索功能，不配置时本地知识库问答仍可运行。
- `backend/.env` 包含私密信息，不要提交到 Git。
- OLLAMA_BASE_URL和OLLAMA_MODEL是调用大模型的地址和名称

## 3. 安装后端依赖

在项目根目录执行：

```powershell
python -m pip install -U pip
python -m pip install -r backend\requirements.txt
```

以上命令会直接把依赖安装到当前系统 Python 环境中。如果电脑上有多个 Python 版本，请先确认 `python --version` 显示的是 3.10。

说明：

- `python -m pip install -U pip`：升级 Python 的包安装工具 pip。
- `python -m pip install -r backend\requirements.txt`：安装后端依赖，例如 FastAPI、LangChain、ChromaDB、DashScope 等。

## 4. 启动后端服务

在项目根目录执行：

```powershell
cd backend
python api_server.py
```

看到类似下面的信息，说明后端启动成功：

```text
FastAPI v4 后端已启动：http://localhost:8888
```

后端默认地址是：

```text
http://localhost:8888
```

## 5. 安装前端依赖

重新打开一个终端，进入前端目录：

```powershell
cd frontend
npm ci
```

说明：

- `cd frontend`：进入前端项目目录。
- `npm ci`：按照 `frontend/package-lock.json` 精确安装前端依赖，依赖会安装到 `frontend/node_modules/`。
- 如果 `frontend/node_modules/` 已经存在，并且依赖没有变化，可以不用重复执行 `npm ci`。

## 6. 启动前端项目

确认当前终端在 `frontend` 目录下，然后执行：

```powershell
npm run dev
```

说明：

- `npm run dev`：启动 Vite 前端开发服务器。
- 启动成功后，终端会显示本地访问地址。

浏览器打开：

```text
http://localhost:5173
```

前端默认请求后端地址 `http://localhost:8888`，所以运行时需要保持后端终端不要关闭。

## 常见问题

### pip 安装失败

先确认 Python 版本：

```powershell
python --version
```

建议使用 Python 3.10。如果仍失败，可以升级 pip：

```powershell
python -m pip install -U pip setuptools wheel
```

### 前端启动失败

确认 Node.js 版本：

```powershell
node --version
npm --version
```

本项目的 Vite 版本要求 Node.js 20.19+ 或 22.12+。

### 页面提示网络错误

通常是后端没有启动，或端口不一致。请确认：

- 后端正在运行 `python api_server.py`
- 后端地址是 `http://localhost:8888`
- 前端地址是 `http://localhost:5173`

### 回答为空或模型报错

检查 `backend/.env` 中的 `DASHSCOPE_API_KEY` 是否正确，以及账号是否有对应模型调用权限。
