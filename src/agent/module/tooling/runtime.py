from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
import re
from xml.etree import ElementTree
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import Any
from bs4 import BeautifulSoup

from src.agent.module.tooling.registry import ToolRegistry


class ToolRuntimeError(RuntimeError):
    pass


class ToolRuntime:
    """Runtime executor for real tool calls used by tool-aware workers."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(self, tool_name: str, tool_input: dict[str, Any], timeout_sec: int = 30) -> dict[str, Any]:
        if not self.registry.has_tool(tool_name):
            raise ToolRuntimeError(f"unknown tool in runtime: {tool_name}")

        if not isinstance(tool_input, dict):
            raise ToolRuntimeError("tool_input must be a dictionary")

        if tool_name in ("duckduckgo_search", "web_search"):
            query = tool_input.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolRuntimeError("duckduckgo_search requires non-empty 'query' string")
            max_results = tool_input.get("max_results", 5)
            if not isinstance(max_results, int) or max_results <= 0:
                max_results = 5
            return self._duckduckgo_search(query=query.strip(), max_results=max_results, timeout_sec=timeout_sec)

        if tool_name == "wikipedia_search":
            query = tool_input.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolRuntimeError("wikipedia_search requires non-empty 'query' string")
            max_results = tool_input.get("max_results", 5)
            if not isinstance(max_results, int) or max_results <= 0:
                max_results = 5
            return self._wikipedia_search(query=query.strip(), max_results=max_results, timeout_sec=timeout_sec)

        if tool_name == "arxiv_search":
            query = tool_input.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolRuntimeError("arxiv_search requires non-empty 'query' string")
            max_results = tool_input.get("max_results", 5)
            if not isinstance(max_results, int) or max_results <= 0:
                max_results = 5
            return self._arxiv_search(query=query.strip(), max_results=max_results, timeout_sec=timeout_sec)

        if tool_name == "fetch_url":
            url = tool_input.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ToolRuntimeError("fetch_url requires non-empty 'url' string")
            max_chars = tool_input.get("max_chars", 5000)
            if not isinstance(max_chars, int) or max_chars <= 0:
                max_chars = 5000
            return self._fetch_url(url=url.strip(), max_chars=max_chars, timeout_sec=timeout_sec)

        if tool_name == "current_datetime":
            now_utc = datetime.now(timezone.utc)
            return {
                "tool": "current_datetime",
                "utc_iso": now_utc.isoformat(),
                "unix_ts": int(now_utc.timestamp()),
            }

        if tool_name == "calculator":
            expression = tool_input.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                raise ToolRuntimeError("calculator requires non-empty 'expression' string")
            value = _safe_eval_arithmetic(expression)
            return {"tool": "calculator", "expression": expression, "value": value}

        if tool_name == "final_answer":
            content = tool_input.get("content")
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            if not isinstance(content, str):
                content = str(content)
            return {"tool": "final_answer", "answer": content}

        raise ToolRuntimeError(f"unsupported tool execution path: {tool_name}")

    def _duckduckgo_search(self, query: str, max_results: int, timeout_sec: int) -> dict[str, Any]:
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1",
        }
        url = f"https://api.duckduckgo.com/?{urlencode(params)}"

        try:
            content_type, raw = _http_get(url, timeout_sec=timeout_sec)
        except Exception as exc:
            raise ToolRuntimeError(f"duckduckgo request failed: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolRuntimeError(f"duckduckgo response is not valid json: {exc}") from exc

        results: list[dict[str, Any]] = []
        abstract_text = payload.get("AbstractText")
        abstract_url = payload.get("AbstractURL")
        heading = payload.get("Heading") or query

        if abstract_text:
            results.append(
                {
                    "title": heading,
                    "snippet": abstract_text,
                    "url": abstract_url,
                    "source": "duckduckgo_abstract",
                }
            )

        related = payload.get("RelatedTopics", [])
        flattened_topics = _flatten_related_topics(related)
        for topic in flattened_topics:
            if len(results) >= max_results:
                break
            text = topic.get("Text")
            first_url = topic.get("FirstURL")
            if not text:
                continue
            results.append(
                {
                    "title": text.split(" - ")[0],
                    "snippet": text,
                    "url": first_url,
                    "source": "duckduckgo_related",
                }
            )

        return {
            "tool": "duckduckgo_search",
            "query": query,
            "result_count": len(results),
            "results": results[:max_results],
        }

    def _wikipedia_search(self, query: str, max_results: int, timeout_sec: int) -> dict[str, Any]:
        params = {
            "action": "opensearch",
            "search": query,
            "limit": str(max_results),
            "namespace": "0",
            "format": "json",
        }
        url = f"https://en.wikipedia.org/w/api.php?{urlencode(params)}"

        try:
            content_type, raw = _http_get(url, timeout_sec=timeout_sec)
        except Exception as exc:
            raise ToolRuntimeError(f"wikipedia request failed: {exc}") from exc

        try:
            payload = json.loads(raw)
            _, titles, descriptions, links = payload
        except Exception as exc:
            raise ToolRuntimeError(f"wikipedia response parse failed: {exc}") from exc

        results = []
        for title, description, link in zip(titles, descriptions, links):
            results.append(
                {
                    "title": title,
                    "snippet": description,
                    "url": link,
                    "source": "wikipedia",
                }
            )

        return {
            "tool": "wikipedia_search",
            "query": query,
            "result_count": len(results),
            "results": results,
        }

    def _arxiv_search(self, query: str, max_results: int, timeout_sec: int) -> dict[str, Any]:
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(max_results),
        }
        url = f"https://export.arxiv.org/api/query?{urlencode(params)}"

        try:
            content_type, raw = _http_get(url, timeout_sec=timeout_sec)
        except Exception as exc:
            raise ToolRuntimeError(f"arxiv request failed: {exc}") from exc

        try:
            root = ElementTree.fromstring(raw)
        except Exception as exc:
            raise ToolRuntimeError(f"arxiv xml parse failed: {exc}") from exc

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
            if not title:
                continue
            results.append(
                {
                    "title": title,
                    "snippet": summary[:500],
                    "url": link,
                    "published": published,
                    "source": "arxiv",
                }
            )

        return {
            "tool": "arxiv_search",
            "query": query,
            "result_count": len(results),
            "results": results[:max_results],
        }

    def _fetch_url(self, url: str, max_chars: int, timeout_sec: int) -> dict[str, Any]:
        try:
            content_type, decoded = _http_get(url, timeout_sec=timeout_sec)
        except Exception as exc:
            raise ToolRuntimeError(f"fetch_url request failed: {exc}") from exc

        title = ""
        text = decoded

        if "text/html" in content_type or "<html" in decoded.lower():
            soup = BeautifulSoup(decoded, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            text = soup.get_text(" ", strip=True)
        else:
            text = decoded

        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars]

        return {
            "tool": "fetch_url",
            "url": url,
            "content_type": content_type,
            "title": title,
            "text": text,
            "text_length": len(text),
        }


def _flatten_related_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for topic in topics:
        if "Topics" in topic and isinstance(topic["Topics"], list):
            flattened.extend(_flatten_related_topics(topic["Topics"]))
        elif isinstance(topic, dict):
            flattened.append(topic)
    return flattened


def _safe_eval_arithmetic(expression: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Mod,
        ast.Constant,
        ast.Load,
        ast.FloorDiv,
    )
    parsed = ast.parse(expression, mode="eval")
    for node in ast.walk(parsed):
        if not isinstance(node, allowed_nodes):
            raise ToolRuntimeError(f"unsupported expression element: {type(node).__name__}")
    return float(eval(compile(parsed, filename="<calc>", mode="eval"), {"__builtins__": {}}, {}))


def _http_get(url: str, timeout_sec: int) -> tuple[str, str]:
    req = Request(
        url,
        headers={
            "User-Agent": "PlanOverGraphToolRuntime/1.0 (+https://github.com/tuandung222/Plan-over-Graph)",
            "Accept": "*/*",
        },
    )
    with urlopen(req, timeout=timeout_sec) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read().decode("utf-8", errors="replace")
    return content_type, raw
