"""本地 MCP Fetch 工具服务端 — 在容器内执行网页抓取，利用容器网络访问国内法律站点

启动方式：
    python -m app.mcp.fetch_mcp_server [--port 8766]

与 ModelScope fetch 不同，此服务器运行在容器本地，可以访问：
- zh.wikipedia.org / en.wikipedia.org
- www.gov.cn / flk.npc.gov.cn / www.court.gov.cn
- 其他不封禁服务器 IP 的站点
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("本地Fetch工具服务")


def _fetch_url(url: str, timeout: int = 15) -> str:
    """使用 urllib 获取 URL 内容，自动处理 WAF cookie 挑战，返回纯文本"""
    import urllib.request
    import urllib.error
    import re

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    cookies = ""

    for attempt in range(2):
        req = urllib.request.Request(url, headers=headers)
        if cookies:
            req.add_header("Cookie", cookies)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read()
                content_type = resp.headers.get("Content-Type", "")
                final_url = resp.geturl()
                break
        except urllib.error.HTTPError as e:
            if e.code == 449:
                # WAF challenge: extract cookie from response and retry
                set_cookie = e.headers.get("Set-Cookie", "")
                if "https_waf_cookie" in set_cookie:
                    waf_match = re.search(r'https_waf_cookie=([^;]+)', set_cookie)
                    if waf_match:
                        cookies = f"https_waf_cookie={waf_match.group(1)}"
                        continue
            return f"HTTP {e.code}: {e.reason} (URL: {url})"
        except urllib.error.URLError as e:
            return f"连接失败: {e.reason} (URL: {url})"
        except Exception as e:
            return f"抓取异常: {type(e).__name__}: {e}"
    else:
        return f"WAF 验证失败，无法获取内容 (URL: {url})"

    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    # HTML → 纯文本
    if "text/html" in content_type:
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<head[^>]*>.*?</head>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"&#?\w+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

    # 截断过长内容（保留前 6000 字符）
    limit = 6000
    suffix = f"\n\n...[内容已截断，共 {len(text)} 字符]" if len(text) > limit else ""
    text = text[:limit]
    header = f"来源: {final_url}\n"
    return header + text + suffix


@mcp.tool()
def fetch(url: str) -> str:
    """抓取指定 URL 的网页内容并返回纯文本。用于获取法律法规、司法解释等实时信息。

    Args:
        url: 要抓取的网页 URL，如 https://www.gov.cn 或 https://zh.wikipedia.org/wiki/公司法
    """
    return _fetch_url(url)


def _search_wikipedia(query: str, lang: str = "zh", max_results: int = 5) -> str:
    """使用 Wikipedia OpenSearch API 搜索，过滤无关结果后返回"""
    import json
    import urllib.parse
    import urllib.request

    base = f"https://{lang}.wikipedia.org/w/api.php"
    params = urllib.parse.urlencode({
        "action": "opensearch",
        "search": query,
        "limit": max_results + 3,  # 多取几条，过滤后还有足够数量
        "namespace": "0",
        "format": "json",
    })
    url = f"{base}?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LegalRAG/1.0 (educational project; legal-research)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"Wikipedia 搜索异常: {type(e).__name__}: {e}"

    # data = [query, [titles], [descriptions], [urls]]
    titles = data[1] if len(data) > 1 else []
    descs = data[2] if len(data) > 2 else []
    urls = data[3] if len(data) > 3 else []

    if not titles:
        return ""

    # 相关性过滤：标题或摘要中必须包含查询关键词的至少一个
    import re as _re_wiki
    query_keywords = [kw for kw in _re_wiki.sub(r'[，。？；：、！\s]+', ' ', query).split() if len(kw) >= 2]
    if query_keywords:
        filtered = []
        for i, (title, desc, url) in enumerate(zip(titles, descs, urls)):
            combined = (title + ' ' + desc).lower()
            match_count = sum(1 for kw in query_keywords if kw.lower() in combined)
            if match_count > 0:
                filtered.append((title, desc, url))
        # 如果过滤后结果太少（<1条），返回空让上层切换到其他搜索引擎
        if len(filtered) < 1:
            return ""
        titles, descs, urls = zip(*filtered) if filtered else ([], [], [])

    titles = list(titles)[:max_results]
    descs = list(descs)[:max_results]
    urls = list(urls)[:max_results]

    lines = [f"Wikipedia '{query}' 搜索结果 ({len(titles)} 条):"]
    for i, (title, desc, url) in enumerate(zip(titles, descs, urls), 1):
        lines.append(f"\n{i}. {title}")
        lines.append(f"   URL: {url}")
        if desc:
            lines.append(f"   摘要: {desc[:300]}")

    return "\n".join(lines)


def _search_bing(query: str, max_results: int = 5) -> str:
    """使用 Bing 搜索，解析 HTML 提取结果"""
    import re
    import urllib.request
    import urllib.error
    import urllib.parse

    encoded = urllib.parse.quote(query)
    search_url = f"https://www.bing.com/search?form=QBLH&q={encoded}"
    req = urllib.request.Request(
        search_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"Bing 搜索失败: HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"Bing 搜索失败: 连接错误 ({e.reason})"
    except Exception as e:
        return f"Bing 搜索异常: {type(e).__name__}: {e}"

    # Extract result blocks — Bing wraps each result in <li class="b_algo">
    results = []
    blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', html, re.DOTALL)

    for block in blocks:
        # Extract title and URL from <a> tag
        link_m = re.search(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
        if not link_m:
            continue

        url = link_m.group(1)
        title = re.sub(r'<[^>]+>', '', link_m.group(2)).strip()

        # Skip non-result URLs
        if not title or url.startswith("https://www.bing.com"):
            continue

        # Extract snippet from the block (text outside <a> tags, in <p> or <div>)
        snippet = ""
        p_m = re.search(r'<(?:p|div)[^>]*class="[^"]*b_line[^"]*"[^>]*>(.*?)</(?:p|div)>', block, re.DOTALL)
        if not p_m:
            p_m = re.search(r'<(?:p|div)[^>]*>(.*?)</(?:p|div)>', block, re.DOTALL)
        if p_m:
            snippet = re.sub(r'<[^>]+>', '', p_m.group(1)).strip()
        if not snippet:
            # Fallback: extract any text from the block
            snippet = re.sub(r'<[^>]+>', ' ', block).strip()
            snippet = re.sub(r'\s+', ' ', snippet)[:300]

        results.append((title, url, snippet[:300]))
        if len(results) >= max_results:
            break

    if not results:
        return ""

    lines = [f"Bing 搜索 '{query}' 结果 ({len(results)} 条):"]
    for i, (title, url, snippet) in enumerate(results, 1):
        lines.append(f"\n{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   摘要: {snippet}")

    return "\n".join(lines)


def _search_duckduckgo(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo Lite 搜索互联网（备用方案，可能触发验证码）"""
    import re
    import urllib.request
    import urllib.error
    from urllib.parse import unquote

    encoded = urllib.request.quote(query)
    search_url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    req = urllib.request.Request(
        search_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LegalRAG/1.0; +legal-rag-bot)",
            "Accept": "text/html",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"搜索失败: HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"搜索失败: 连接错误 ({e.reason})"
    except Exception as e:
        return f"搜索异常: {type(e).__name__}: {e}"

    # 检查是否触发了验证码
    if "captcha" in html.lower() or "challenge" in html.lower():
        return ""  # 静默回退，由主搜索逻辑处理

    results = []
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    for row in rows:
        link_m = re.search(r'<a\s+rel="nofollow"\s+href="([^"]+)"[^>]*>(.*?)</a>', row, re.DOTALL)
        if not link_m:
            continue
        title = re.sub(r"<[^>]+>", "", link_m.group(2)).strip()
        if not title:
            continue

        ddg_url = link_m.group(1)
        uddg_m = re.search(r"uddg=([^&]+)", ddg_url)
        real_url = unquote(uddg_m.group(1)) if uddg_m else ddg_url
        if real_url.startswith("/"):
            real_url = "https:" + real_url

        tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        snippets = []
        for td in tds[2:]:
            text = re.sub(r"<[^>]+>", "", td).strip()
            if text and text != title:
                snippets.append(text)
        snippet = " ".join(snippets)[:300]

        results.append((title, real_url, snippet))
        if len(results) >= max_results:
            break

    if not results:
        return ""

    lines = [f"搜索 '{query}' 结果 ({len(results)} 条):"]
    for i, (title, url, snippet) in enumerate(results, 1):
        lines.append(f"\n{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   摘要: {snippet}")

    return "\n".join(lines)


def _search_web(query: str, max_results: int = 5) -> str:
    """综合搜索：Wikipedia API → Bing → DuckDuckGo，含关键词自动简化"""
    # 关键词简化：提取前2-3个核心词
    import re as _re
    keywords = _re.sub(r'[，。？；：、！\s]+', ' ', query).strip()
    # 去掉常见修饰词
    stop_words = ['最新', '规定', '修订', '解读', '如何', '什么', '请问', '哪些', '怎么', '多少']
    parts = [p for p in keywords.split() if p not in stop_words]
    simple_query = ' '.join(parts[:3]) if len(parts) > 1 else query

    queries_to_try = [query]
    if simple_query != query:
        queries_to_try.append(simple_query)
    if len(parts) > 1:
        queries_to_try.append(parts[0])  # 单核心词回退

    for q in queries_to_try:
        # 策略1: Wikipedia 中文
        result = _search_wikipedia(q, "zh", max_results)
        if result and "异常" not in result:
            return result

        # 策略2: Wikipedia 英文
        result = _search_wikipedia(q, "en", max_results)
        if result and "异常" not in result:
            return result

    # 策略3: Bing 搜索 (注意：依赖 JS 渲染，容器 IP 可能被限制)
    result = _search_bing(query, max_results)
    if result and "失败" not in result and "异常" not in result:
        return result

    # 策略4: DuckDuckGo Lite (备用，可能触发验证码)
    result = _search_duckduckgo(query, max_results)
    if result:
        return result

    return '未找到与 "' + query + '" 相关的结果。请尝试更换简短的核心关键词（如 公司法 而非 公司法注册资本最新修订）。'


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """搜索互联网，返回相关网页的标题、摘要和URL。用于查找法律法规、司法解释等实时信息。

    工作流程: 先使用本工具搜索关键词获取结果列表，再对高相关结果使用 fetch 工具深度抓取全文。

    Args:
        query: 搜索关键词，如 "公司法 注册资本 2024修订"
        max_results: 返回结果数量，默认5条，最多10条
    """
    return _search_web(query, max_results=int(min(max_results, 10)))


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="本地 Fetch MCP 服务端 (SSE)")
    parser.add_argument("--port", type=int, default=8766, help="监听端口 (默认 8766)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址 (默认 127.0.0.1)")
    args = parser.parse_args()
    app = mcp.sse_app()
    print(f"[fetch-mcp] 启动本地 Fetch MCP 服务: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
