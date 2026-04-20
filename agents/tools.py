"""LangChain tools for the navel orange knowledge agent."""

from typing import Optional

from langchain.tools import tool
from langchain_community.vectorstores import FAISS


_vector_store: Optional[FAISS] = None


def set_vector_store(vs: FAISS) -> None:
    """Register the global vector store used by search tools."""
    global _vector_store
    _vector_store = vs


@tool
def search_knowledge_base(query: str) -> str:
    """
    在脐橙知识库中搜索与问题相关的信息。
    当需要回答关于脐橙的任何问题时，请首先使用此工具检索相关知识。
    输入应为搜索关键词或问题描述。
    """
    if _vector_store is None:
        return "知识库尚未加载，请联系管理员。"
    docs = _vector_store.similarity_search(query, k=4)
    if not docs:
        return "未在知识库中找到相关信息。"
    results = []
    for i, doc in enumerate(docs, 1):
        topic = doc.metadata.get("topic", "")
        heading = doc.metadata.get("heading", "")
        label = f"[{topic}] {heading}".strip("[] ") if heading else f"[{topic}]"
        results.append(f"参考资料{i}（{label}）:\n{doc.page_content}")
    return "\n\n---\n\n".join(results)


@tool
def get_variety_info(variety_name: str) -> str:
    """
    获取特定脐橙品种的详细信息，包括外观特征、成熟期、口感品质和适种区域。
    输入品种名称，如"纽荷尔"、"朋娜"、"奈维林纳"等。
    """
    if _vector_store is None:
        return "知识库尚未加载，请联系管理员。"
    query = f"脐橙品种 {variety_name} 特征 成熟期 品质"
    docs = _vector_store.similarity_search(query, k=3)
    if not docs:
        return f'未找到关于"{variety_name}"品种的信息。'
    results = []
    for doc in docs:
        if variety_name in doc.page_content or "品种" in doc.metadata.get("topic", ""):
            results.append(doc.page_content)
    if not results:
        results = [docs[0].page_content]
    return "\n\n".join(results[:2])


@tool
def get_disease_pest_info(disease_or_pest: str) -> str:
    """
    获取脐橙特定病害或虫害的详细信息，包括症状、危害和防治方法。
    输入病虫害名称，如"黄龙病"、"溃疡病"、"红蜘蛛"、"木虱"等。
    """
    if _vector_store is None:
        return "知识库尚未加载，请联系管理员。"
    query = f"脐橙 {disease_or_pest} 症状 防治方法"
    docs = _vector_store.similarity_search(query, k=3)
    if not docs:
        return f'未找到关于"{disease_or_pest}"的信息。'
    return "\n\n---\n\n".join(doc.page_content for doc in docs[:2])


@tool
def get_cultivation_tips(stage: str) -> str:
    """
    获取脐橙特定生长阶段或栽培环节的技术要点。
    输入如"施肥"、"修剪"、"采收"、"定植"、"灌溉"、"套袋"等关键词。
    """
    if _vector_store is None:
        return "知识库尚未加载，请联系管理员。"
    query = f"脐橙 {stage} 技术要点 方法"
    docs = _vector_store.similarity_search(query, k=3)
    if not docs:
        return f'未找到关于"{stage}"的栽培技术信息。'
    return "\n\n---\n\n".join(doc.page_content for doc in docs[:2])


@tool
def calculate_yield_estimate(area_mu: float, yield_per_mu: float, price_per_kg: float) -> str:
    """
    根据种植面积、亩产量和单价估算脐橙种植收益。
    输入格式：面积（亩）, 亩产量（公斤/亩）, 单价（元/公斤）
    例如：calculate_yield_estimate(10, 4000, 4.0)
    """
    total_yield = area_mu * yield_per_mu
    total_revenue = total_yield * price_per_kg
    # Average cost estimate based on standard navel orange orchard management
    # (range: 3700-7000 yuan/mu depending on region and practices; midpoint used)
    AVERAGE_COST_PER_MU = 5000
    total_cost = area_mu * AVERAGE_COST_PER_MU
    profit = total_revenue - total_cost
    return (
        f"收益估算结果：\n"
        f"  种植面积：{area_mu} 亩\n"
        f"  亩产量：{yield_per_mu} 公斤\n"
        f"  总产量：{total_yield:.0f} 公斤\n"
        f"  销售单价：{price_per_kg} 元/公斤\n"
        f"  毛收入：{total_revenue:,.0f} 元\n"
        f"  估算成本（按5000元/亩）：{total_cost:,.0f} 元\n"
        f"  估算纯利润：{profit:,.0f} 元\n"
        f"  （注：实际利润受价格波动、管理水平等因素影响，此估算仅供参考）"
    )
