# main.py

import os
import re
from config import SITE_CONFIGS
from spiders import BaseSpider, URLListSpider
from utils import ensure_dir, sanitize_filename, save_text, save_json, md5_text


RAW_DIR = "data/raw_html"
MD_DIR = "data/markdown"
META_DIR = "data/metadata"


def guess_topic(title: str, markdown: str) -> str:
    text = f"{title}\n{markdown}"
    mapping = {
        "病虫害防治": ["病虫", "黄龙病", "防治", "病害", "虫害"],
        "土壤肥力与施肥": ["施肥", "肥料", "养分", "土壤"],
        "水肥一体化与灌溉": ["灌溉", "水肥一体化", "滴灌", "水分"],
        "气候环境与灾害": ["寒潮", "冻害", "干旱", "高温", "灾害", "气象"],
        "采后处理与贮藏加工": ["采后", "贮藏", "储运", "包装", "运输"],
        "品种资源与新品种": ["品种", "新品种", "种质", "砧木"],
        "栽培与生产管理": ["栽培", "修剪", "果园", "管理", "生产"],
        "检测方法": ["检测", "检验", "测定", "方法"],
        "果实品质与分布特征": ["品质", "糖度", "固形物", "分布", "产区"],
    }
    for topic, keywords in mapping.items():
        if any(k in text for k in keywords):
            return topic
    return "综合"


def detect_doc_type(site_key: str, title: str, markdown: str) -> str:
    text = f"{title}\n{markdown}"
    if "标准" in text or "GB/" in text or "GBT" in text:
        return "标准"
    if any(k in text for k in ["条例", "法规"]):
        return "法规"
    if any(k in text for k in ["规划", "方案", "实施意见", "通知"]):
        return "规划/政策"
    if site_key == "natesc":
        return "农事指导"
    return "资料"


def save_records(records: list[dict]):
    ensure_dir(RAW_DIR)
    ensure_dir(MD_DIR)
    ensure_dir(META_DIR)

    for rec in records:
        title = rec["title"] or "untitled"
        fid = md5_text(rec["url"])
        safe_title = sanitize_filename(title)

        raw_path = os.path.join(RAW_DIR, f"{fid}_{safe_title}.html")
        md_path = os.path.join(MD_DIR, f"{fid}_{safe_title}.md")
        meta_path = os.path.join(META_DIR, f"{fid}_{safe_title}.json")

        topic = guess_topic(title, rec["markdown"])
        doc_type = detect_doc_type(rec["site_key"], title, rec["markdown"])

        metadata = {
            "id": fid,
            "title": title,
            "url": rec["url"],
            "source_site": rec["source_site"],
            "site_key": rec["site_key"],
            "publish_date": rec.get("publish_date", ""),
            "topic": topic,
            "doc_type": doc_type,
        }

        # markdown 文件头
        md_text = (
            f"# {title}\n\n"
            f"- 来源站点: {rec['source_site']}\n"
            f"- URL: {rec['url']}\n"
            f"- 发布日期: {rec.get('publish_date', '')}\n"
            f"- 分类: {topic}\n"
            f"- 文档类型: {doc_type}\n\n"
            f"---\n\n"
            f"{rec['markdown']}\n"
        )

        save_text(raw_path, rec["raw_html"])
        save_text(md_path, md_text)
        save_json(meta_path, metadata)

        print(f"[SAVE] {md_path}")


def crawl_site(site_key: str):
    config = SITE_CONFIGS[site_key]
    spider = BaseSpider(site_key, config)
    records = spider.crawl()
    save_records(records)


def crawl_standard_detail_urls():
    # 标准页建议你先手工整理 URL 再抓
    detail_urls = [
        # 例子：这里替换成你实际收集到的标准详情页
        # "https://std.samr.gov.cn/gb/search/gbDetailed?id=71F772D7F6E9D3A7E05397BE0A0AB82A",
    ]
    if not detail_urls:
        print("[INFO] no standard detail urls configured.")
        return

    config = SITE_CONFIGS["standard"]
    spider = URLListSpider("standard", config)
    records = spider.crawl_from_detail_urls(detail_urls)
    save_records(records)


if __name__ == "__main__":
    # 先抓农技中心
    crawl_site("natesc")

    # 再抓赣州政府公开页
    crawl_site("ganzhou_gov")

    # 如果你已经整理好了标准详情页，就抓标准
    crawl_standard_detail_urls()