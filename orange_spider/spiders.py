# spiders.py

import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from utils import absolute_url, is_http_url
from cleaners import clean_article_html_to_markdown, extract_publish_date


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


class BaseSpider:
    def __init__(self, site_key: str, config: dict, timeout: int = 20):
        self.site_key = site_key
        self.config = config
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch(self, url: str) -> str:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return resp.text

    def _match_keywords(self, text: str) -> bool:
        keywords = self.config.get("allowed_keywords", [])
        if not keywords:
            return True
        return any(k in text for k in keywords)

    def extract_links_from_list_page(self, list_url: str):
        html = self.fetch(list_url)
        soup = BeautifulSoup(html, "lxml")
        base_url = self.config.get("base_url", list_url)

        results = []
        seen = set()

        # 这里用最宽松策略：先扫所有 a
        for a in soup.select("a[href]"):
            title = a.get_text(" ", strip=True)
            href = a.get("href", "").strip()

            if not href or not title:
                continue

            full_url = absolute_url(base_url, href)
            if not is_http_url(full_url):
                continue

            # 粗过滤：标题含关键词
            if not self._match_keywords(title):
                continue

            if full_url in seen:
                continue
            seen.add(full_url)

            results.append({
                "title": title,
                "url": full_url,
                "source_site": self.config.get("site_name", self.site_key),
            })

        return results

    def extract_main_content_html(self, detail_html: str) -> str:
        soup = BeautifulSoup(detail_html, "lxml")

        selectors = self.config.get("content_selector_candidates", [])
        for selector in selectors:
            node = soup.select_one(selector)
            if node and len(node.get_text("\n", strip=True)) > 80:
                return str(node)

        # 兜底：找文本最多的 div
        best_tag = None
        best_len = 0
        for tag in soup.find_all(["div", "article", "section"]):
            text_len = len(tag.get_text("\n", strip=True))
            if text_len > best_len:
                best_len = text_len
                best_tag = tag

        if best_tag:
            return str(best_tag)

        return str(soup.body) if soup.body else detail_html

    def extract_detail(self, item: dict):
        detail_html = self.fetch(item["url"])
        content_html = self.extract_main_content_html(detail_html)
        markdown = clean_article_html_to_markdown(content_html)

        publish_date = extract_publish_date(detail_html)

        return {
            "title": item["title"],
            "url": item["url"],
            "source_site": item["source_site"],
            "site_key": self.site_key,
            "publish_date": publish_date,
            "raw_html": detail_html,
            "content_html": content_html,
            "markdown": markdown,
        }

    def crawl(self):
        all_items = []
        for list_url in self.config.get("list_urls", []):
            try:
                items = self.extract_links_from_list_page(list_url)
                all_items.extend(items)
                time.sleep(random.uniform(1, 2))
            except Exception as e:
                print(f"[ERROR] list page failed: {list_url} -> {e}")

        # 去重
        dedup = {}
        for item in all_items:
            dedup[item["url"]] = item
        all_items = list(dedup.values())

        print(f"[INFO] {self.site_key} list items: {len(all_items)}")

        details = []
        for idx, item in enumerate(all_items, 1):
            try:
                detail = self.extract_detail(item)
                details.append(detail)
                print(f"[OK] {idx}/{len(all_items)} {item['title']}")
                time.sleep(random.uniform(1, 2))
            except Exception as e:
                print(f"[ERROR] detail failed: {item['url']} -> {e}")

        return details


class URLListSpider(BaseSpider):
    """
    用于你已经手工整理好详情页 URL 的情况
    """
    def crawl_from_detail_urls(self, detail_urls: list[str]):
        details = []
        for idx, url in enumerate(detail_urls, 1):
            try:
                item = {
                    "title": url,
                    "url": url,
                    "source_site": self.config.get("site_name", self.site_key),
                }
                detail = self.extract_detail(item)
                # 如果 title 还是 URL，就尝试从正文里补标题
                if detail["title"] == url:
                    detail["title"] = self._extract_title_from_html(detail["raw_html"]) or url
                details.append(detail)
                print(f"[OK] {idx}/{len(detail_urls)} {detail['title']}")
                time.sleep(random.uniform(1, 2))
            except Exception as e:
                print(f"[ERROR] detail failed: {url} -> {e}")
        return details

    def _extract_title_from_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        if soup.title:
            return soup.title.get_text(strip=True)
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""