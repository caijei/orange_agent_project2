# 脐橙知识问答系统

> 本科毕业设计 · 基于 Agent 的脐橙知识问答系统

---

## 项目简介

本系统是一个基于大语言模型（LLM）和 LangChain Agent 框架构建的**脐橙领域知识问答系统**。系统整合了脐橙品种、栽培技术、病虫害防治、营养价值、市场行情和生长环境等多维度知识，支持用户以自然语言提问，并由 Agent 智能调用工具检索知识库，生成专业、准确的回答。

### 核心特性

| 特性 | 说明 |
|------|------|
| **Agent 架构** | 基于 LangChain 的 OpenAI Tools Agent，支持多工具调用和推理 |
| **RAG 检索增强** | 使用 FAISS 向量数据库进行语义检索，确保回答有据可查 |
| **多维知识库** | 涵盖品种、栽培、病虫害、营养、市场、气候六大领域 |
| **对话记忆** | 支持多轮对话，保持上下文一致性 |
| **多会话管理** | 支持多用户同时使用，会话隔离 |
| **Web 界面** | 简洁美观的中文 Web 问答界面 |

---

## 系统架构

```
用户输入问题
    │
    ▼
FastAPI 后端 (app.py)
    │
    ▼
LangChain Agent (agents/orange_agent.py)
    │
    ├── search_knowledge_base      ← FAISS 语义检索
    ├── get_variety_info           ← 品种信息查询
    ├── get_disease_pest_info      ← 病虫害信息查询
    ├── get_cultivation_tips       ← 栽培技术查询
    └── calculate_yield_estimate   ← 收益估算计算
    │
    ▼
OpenAI LLM / 兼容 API
    │
    ▼
生成最终回答 → 返回前端
```

---

## 知识库内容

| 文件 | 主题 | 主要内容 |
|------|------|---------|
| `varieties.txt` | 品种介绍 | 纽荷尔、朋娜、奈维林纳、清家、林娜等主要品种特性 |
| `cultivation.txt` | 栽培技术 | 建园、施肥、修剪、灌溉、套袋、采收全流程 |
| `diseases.txt` | 病虫害防治 | 黄龙病、溃疡病、炭疽病、红蜘蛛、木虱等及防治方法 |
| `nutrition.txt` | 营养价值 | 营养成分、健康功效、食用建议 |
| `market.txt` | 市场与产业 | 产区分布、价格行情、产业链、市场趋势 |
| `climate.txt` | 生长环境 | 温度、光照、水分、土壤需求及气象灾害防范 |

---

## 快速开始

### 环境要求

- Python 3.10+
- OpenAI API Key（或兼容 OpenAI 接口的其他 API）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/caijei/orange_agent_project2.git
cd orange_agent_project2

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填写您的 OpenAI API Key
```

### 配置说明（`.env`）

```dotenv
OPENAI_API_KEY=your_api_key_here     # 必填
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，用于国内代理或其他兼容API
MODEL_NAME=gpt-3.5-turbo             # 使用的模型
EMBEDDING_MODEL=text-embedding-ada-002
PORT=8000
```

### 启动服务

```bash
python app.py
```

浏览器访问 `http://localhost:8000` 即可使用 Web 问答界面。

---

## API 接口

### POST `/api/chat` — 问答接口

```json
// 请求体
{
  "question": "纽荷尔脐橙有哪些特点？",
  "session_id": "user123"
}

// 响应
{
  "answer": "纽荷尔脐橙是目前国内种植面积最广的品种...",
  "session_id": "user123"
}
```

### DELETE `/api/session/{session_id}` — 清除会话

### GET `/api/topics` — 获取知识领域列表

### GET `/health` — 健康检查

---

## 提问示例

- 纽荷尔脐橙和朋娜脐橙有什么区别？
- 脐橙黄龙病如何识别和防治？
- 脐橙套袋有什么好处？什么时候套袋合适？
- 赣南脐橙为什么品质好？
- 脐橙每天吃多少合适？有哪些营养价值？
- 帮我估算 20 亩脐橙，亩产 4000 公斤，价格 5 元/公斤的年收益

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| Agent 框架 | LangChain |
| 向量数据库 | FAISS |
| 嵌入模型 | OpenAI text-embedding-ada-002 |
| 大语言模型 | OpenAI GPT-3.5-turbo / GPT-4 |
| 前端 | 原生 HTML + CSS + JavaScript |

---

## 项目结构

```
orange_agent_project2/
├── app.py                    # FastAPI 主应用
├── config.py                 # 配置管理
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量示例
├── agents/
│   ├── orange_agent.py       # LangChain Agent 定义
│   └── tools.py              # Agent 工具集
├── knowledge/
│   ├── loader.py             # 知识库加载器
│   ├── vector_store.py       # FAISS 向量库管理
│   └── data/                 # 知识库文本数据
│       ├── varieties.txt     # 品种知识
│       ├── cultivation.txt   # 栽培技术
│       ├── diseases.txt      # 病虫害防治
│       ├── nutrition.txt     # 营养价值
│       ├── market.txt        # 市场信息
│       └── climate.txt       # 气候环境
├── templates/
│   └── index.html            # Web 界面模板
└── static/
    ├── css/style.css         # 样式
    └── js/app.js             # 前端逻辑
```

---

## 许可证

本项目为学术用途（本科毕业设计），仅供学习参考。
