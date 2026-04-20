# config.py

SITE_CONFIGS = {
    "natesc": {
        "site_name": "全国农业技术推广服务中心",
        "base_url": "https://www.natesc.org.cn",
        "list_urls": [
            # 这里先放一个栏目页，后续你可以继续加翻页链接
            "https://www.natesc.org.cn/GK/index?Cate1=%E5%85%AC%E5%BC%80&Cate2=%E5%86%9C%E4%BA%8B%E6%8C%87%E5%AF%BC&CategoryId=11a63552-05c9-475e-a504-0392e64ead0b"
        ],
        "allowed_keywords": ["柑橘", "脐橙"],
        "list_item_selector_candidates": [
            "a"
        ],
        "content_selector_candidates": [
            "div.article-content",
            "div.content",
            "div.TRS_Editor",
            "div#zoom",
            "div.zw",
            "div.article",
            "div.details"
        ]
    },

    "ganzhou_gov": {
        "site_name": "赣州政府及主产区政府",
        "base_url": "https://www.ganzhou.gov.cn",
        "list_urls": [
            # 可以先把搜索结果页、专题页、公开页放进来
            # 后面你可继续补
            "https://www.ganzhou.gov.cn/zfxxgk/c144129/list.shtml"
        ],
        "allowed_keywords": ["脐橙", "赣南脐橙", "黄龙病", "种苗", "采后处理", "实施方案", "规划"],
        "list_item_selector_candidates": [
            "a"
        ],
        "content_selector_candidates": [
            "div.TRS_Editor",
            "div.article-content",
            "div.content",
            "div#zoom",
            "div.zw",
            "div.article"
        ]
    },

    "law_plan": {
        "site_name": "法规规划类页面",
        "base_url": "",
        "list_urls": [
            # 这里建议后续你自己补具体法规或规划的列表页
        ],
        "allowed_keywords": ["赣南脐橙", "条例", "规划", "保护"],
        "list_item_selector_candidates": [
            "a"
        ],
        "content_selector_candidates": [
            "div.TRS_Editor",
            "div.article-content",
            "div.content",
            "div#zoom",
            "div.zw"
        ]
    },

    "standard": {
        "site_name": "标准信息页",
        "base_url": "https://std.samr.gov.cn",
        "list_urls": [
            # 标准站很多内容是搜索跳转，不一定适合直接列表抓
            # 这里建议你先手工收集标准详情页 URL，再喂给代码抓详情
        ],
        "allowed_keywords": ["脐橙", "赣南脐橙", "柑橘"],
        "list_item_selector_candidates": [
            "a"
        ],
        "content_selector_candidates": [
            "body"
        ]
    }
}