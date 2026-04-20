# cleaners.py

import re
from bs4 import BeautifulSoup
import html2text
from utils import normalize_whitespace


NOISE_PATTERNS = [
    r"上一篇.*",
    r"下一篇.*",
    r"打印本页.*",
    r"关闭窗口.*",
    r"责任编辑[:：].*",
    r"来源[:：].*",
    r"扫一扫在手机打开当前页.*",
    r"主办单位[:：].*",
    r"版权所有.*",
    r"网站地图.*",
    r"联系我们.*",
    r"ICP备.*",
]

def clean_html_noise(soup: BeautifulSoup):
    # 删除常见无关标签
    for tag in soup(["script", "style", "noscript", "iframe", "form", "button"]):
        tag.decompose()

    # 删除导航/页脚等常见 class/id
    bad_keywords = [
        "nav", "navbar", "footer", "header", "breadcrumb", "menu",
        "share", "pagination", "page", "related", "recommend", "copyright"
    ]

    all_tags = soup.find_all(True)
    for tag in all_tags:
        attr_text = " ".join([
            str(tag.get("id", "")),
            " ".join(tag.get("class", [])) if isinstance(tag.get("class"), list) else str(tag.get("class", "")),
        ]).lower()
        if any(k in attr_text for k in bad_keywords):
            # 只删明显噪声容器，避免误删正文
            if tag.name in ("div", "section", "aside", "ul", "nav", "footer", "header"):
                tag.decompose()


def html_to_markdown(html: str) -> str:
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.body_width = 0
    md = converter.handle(html)
    return md


def clean_text_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []
    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue
        if any(re.fullmatch(p, line) for p in NOISE_PATTERNS):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 去掉过多空行
    text = normalize_whitespace(text)
    return text


def clean_article_html_to_markdown(article_html: str) -> str:
    soup = BeautifulSoup(article_html, "lxml")
    clean_html_noise(soup)
    md = html_to_markdown(str(soup))
    md = clean_text_lines(md)
    return md


def extract_publish_date(text: str) -> str:
    patterns = [
        r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)",
        r"(\d{4}\.\d{1,2}\.\d{1,2})"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return ""