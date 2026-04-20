"""LangChain agent for navel orange knowledge Q&A."""

from typing import List

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI

from agents.tools import (
    calculate_yield_estimate,
    get_cultivation_tips,
    get_disease_pest_info,
    get_variety_info,
    search_knowledge_base,
)
from config import config

SYSTEM_PROMPT = """你是一个专业的脐橙知识问答助手，专门解答关于脐橙（Navel Orange）的各类问题。

你具备以下方面的专业知识：
- 脐橙品种特性与选种建议
- 脐橙栽培技术（定植、施肥、修剪、灌溉、套袋等）
- 脐橙病虫害识别与防治（黄龙病、溃疡病、红蜘蛛、木虱等）
- 脐橙的营养价值与健康功效
- 脐橙市场行情与产业发展
- 脐橙生长环境与气候要求

**回答原则：**
1. 优先使用工具搜索知识库，确保回答有据可查。
2. 回答要专业、准确、实用，适合农业生产者和消费者。
3. 如果问题超出脐橙知识范围，请礼貌地说明并引导回脐橙相关话题。
4. 回答时请注意结构清晰，适当使用列表和分段。
5. 对于病虫害问题，请特别注意强调农药安全使用和环境保护。
6. 请用中文回答（除非用户用其他语言提问）。

当前系统：脐橙知识问答系统（基于Agent的农业知识问答）"""

TOOLS = [
    search_knowledge_base,
    get_variety_info,
    get_disease_pest_info,
    get_cultivation_tips,
    calculate_yield_estimate,
]


def create_orange_agent() -> AgentExecutor:
    """Create and return the navel orange Q&A agent executor."""
    llm = ChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm=llm, tools=TOOLS, prompt=prompt)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=config.MAX_ITERATIONS,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    return agent_executor
