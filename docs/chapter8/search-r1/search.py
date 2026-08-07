"""提供 DeepSeek Search、Wikipedia 和知乎搜索三个后端。"""

import gzip
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
from deepseek_search import search as deepseek_search
from deepseek_search.config import resolve_api_key
from dotenv import load_dotenv


SEARCH_BACKENDS = ("deepseek", "wikipedia", "zhihu")
SEARCH_ENDPOINT = "https://developer.zhihu.com/api/v1/content/global_search"
WIKIPEDIA_SEARCH_ENDPOINT = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = (
    "agentic-rl-lab/0.1 "
    "(https://github.com/KMnO4-zx/agentic-rl-lab)"
)
DEFAULT_SEARCH_CONCURRENCY = {"deepseek": 16, "wikipedia": 3, "zhihu": 1}
DEFAULT_SEARCH_TIMEOUT = {"deepseek": 60.0, "wikipedia": 15.0, "zhihu": 15.0}
EVIDENCE_PATTERN = re.compile(
    r"^\[(?:\d+)\]\s*Source:\s*(.*?)\s*\nEvidence:\s*(.*?)"
    r"(?=\n\s*\n\[(?:\d+)\]\s*Source:|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class SearchItem:
    """保存一条搜索证据及其可选来源信息。"""

    title: str
    content: str
    source: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """保存一次搜索的证据或错误信息。"""

    ok: bool
    items: list[SearchItem]
    latency: float
    status: int | None = None
    error: str | None = None


@dataclass
class SearchStats:
    """累计并发搜索请求的运行指标。"""

    backend: str
    requests: int = 0
    successes: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    errors: int = 0
    latency_total: float = 0.0
    credential_failovers: int = 0
    web_search_requests: int = 0
    result_count: int = 0
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def metrics(self) -> dict[str, float]:
        """把累计计数转换成便于 SwanLab 记录的单次请求均值。"""
        with self._lock:
            denominator = max(self.requests, 1)
            metrics = {
                "search/success_rate": self.successes / denominator,
                "search/timeout_rate": self.timeouts / denominator,
                "search/429_rate": self.rate_limits / denominator,
                "search/error_rate": self.errors / denominator,
                "search/latency": self.latency_total / denominator,
                "search/results": self.result_count / denominator,
            }
            if self.backend == "deepseek":
                metrics.update(
                    {
                        "search/web_search_requests": (
                            self.web_search_requests / denominator
                        ),
                        "search/input_tokens": self.input_tokens / denominator,
                        "search/cache_read_input_tokens": (
                            self.cache_read_input_tokens / denominator
                        ),
                        "search/output_tokens": self.output_tokens / denominator,
                    }
                )
            if self.backend == "zhihu":
                metrics["search/credential_failover_rate"] = (
                    self.credential_failovers / denominator
                )
            return metrics


@dataclass
class DeepSeekSearchClient:
    """调用受约束的 Evidence 模式，并把返回文本转成工具证据。"""

    api_key: str | None = field(default=None, repr=False)
    model: str = "deepseek-v4-flash"
    timeout: float = DEFAULT_SEARCH_TIMEOUT["deepseek"]
    max_retries: int = 1
    retry_delay: float = 1.0
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="deepseek")
    )

    def __post_init__(self) -> None:
        # 提前检查登录状态，避免 rollout 开始后才发现所有搜索都无法鉴权。
        self.api_key = resolve_api_key(self.api_key)

    @classmethod
    def from_env(
        cls, env_path: str | Path | None = None, **kwargs: Any
    ) -> "DeepSeekSearchClient":
        """优先读取项目 .env，也兼容 deepseek-search login 保存的密钥。"""
        if env_path:
            load_dotenv(env_path)
        return cls(**kwargs)

    def search(self, query: str) -> SearchResult:
        """执行一次 Evidence 搜索；超时、429 和 5xx 最多有限重试。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        saw_timeout = False
        saw_rate_limit = False
        attempt = 0
        while True:
            try:
                response = deepseek_search(
                    query,
                    api_key=self.api_key,
                    model=self.model,
                    timeout=self.timeout,
                    mode="evidence",
                )
                latency = time.perf_counter() - started
                items = parse_evidence(response.evidence)
                usage = response.usage
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += latency
                    self.stats.web_search_requests += response.total_search_requests
                    self.stats.result_count += response.result_count
                    self.stats.input_tokens += int(usage.get("input_tokens", 0))
                    self.stats.cache_read_input_tokens += int(
                        usage.get("cache_read_input_tokens", 0)
                    )
                    self.stats.output_tokens += int(usage.get("output_tokens", 0))
                return SearchResult(ok=True, items=items, latency=latency, status=200)
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                retryable = status == 429 or status >= 500
                with self.stats._lock:
                    if status == 429 and not saw_rate_limit:
                        self.stats.rate_limits += 1
                        saw_rate_limit = True
                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {status}", status)
            except httpx.TimeoutException:
                with self.stats._lock:
                    if not saw_timeout:
                        self.stats.timeouts += 1
                        saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except httpx.HTTPError as error:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, type(error).__name__)
            except RuntimeError as error:
                return self._error_result(started, str(error))

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换为不会泄露密钥的工具结果。"""
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


@dataclass
class WikipediaSearchClient:
    """使用 Wikimedia Action API 搜索英文 Wikipedia 并返回页面正文片段。"""

    timeout: float = DEFAULT_SEARCH_TIMEOUT["wikipedia"]
    max_retries: int = 2
    retry_delay: float = 1.0
    min_request_interval: float = 0.31
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="wikipedia")
    )
    _rate_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _next_request_time: float = field(default=0.0, init=False, repr=False)

    def search(self, query: str) -> SearchResult:
        """搜索 Top 3 页面；所有并发调用共享约 200 RPM 的启动限速。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        saw_timeout = False
        saw_rate_limit = False
        attempt = 0
        while True:
            self._wait_for_rate_slot()
            try:
                result = self._request(query, started)
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += result.latency
                    self.stats.result_count += len(result.items)
                return result
            except urllib.error.HTTPError as error:
                if error.code == 429 and not saw_rate_limit:
                    with self.stats._lock:
                        self.stats.rate_limits += 1
                    saw_rate_limit = True
                retryable = error.code == 429 or error.code >= 500
                if retryable and attempt < self.max_retries:
                    retry_after = error.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else self.retry_delay * (2**attempt)
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {error.code}", error.code)
            except (TimeoutError, socket.timeout):
                if not saw_timeout:
                    with self.stats._lock:
                        self.stats.timeouts += 1
                    saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except urllib.error.URLError as error:
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    if not saw_timeout:
                        with self.stats._lock:
                            self.stats.timeouts += 1
                        saw_timeout = True
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        attempt += 1
                        continue
                    return self._error_result(started, "request timeout")
                return self._error_result(started, type(error).__name__)
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                return self._error_result(started, type(error).__name__)

    def _wait_for_rate_slot(self) -> None:
        """序列化请求启动时间，避免超过 Wikimedia 的识别客户端分钟限额。"""
        with self._rate_lock:
            now = time.monotonic()
            delay = self._next_request_time - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_request_time = now + self.min_request_interval

    def _request(self, query: str, started: float) -> SearchResult:
        """一次请求同时执行全文搜索并取得前三个页面的纯文本摘要。"""
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 3,
                "prop": "extracts|info",
                "explaintext": 1,
                "exintro": 1,
                "exchars": 1200,
                "inprop": "url",
                "redirects": 1,
                "utf8": 1,
            }
        )
        request = urllib.request.Request(
            f"{WIKIPEDIA_SEARCH_ENDPOINT}?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": WIKIPEDIA_USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            payload = json.loads(body.decode("utf-8"))
            pages = payload.get("query", {}).get("pages", [])
            if not isinstance(pages, list):
                raise TypeError("Wikipedia 搜索响应 pages 不是列表")
            if any(not isinstance(page, dict) for page in pages):
                raise TypeError("Wikipedia 搜索响应 page 不是对象")
            ordered_pages = sorted(
                pages,
                key=lambda page: int(page.get("index", 1_000_000)),
            )
            items = [
                SearchItem(
                    title=str(page.get("title") or "Untitled").strip(),
                    content=str(page.get("extract") or "").strip(),
                    source="Wikipedia",
                    url=str(page.get("fullurl") or "").strip(),
                )
                for page in ordered_pages
                if str(page.get("extract") or "").strip()
            ]
            return SearchResult(
                ok=True,
                items=items,
                latency=time.perf_counter() - started,
                status=response.status,
            )

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换成 rollout 可观察、不中断训练的搜索结果。"""
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


@dataclass
class ZhihuSearchClient:
    """轮转使用多组凭证，并通过有限重试执行知乎搜索。"""

    access_secrets: str | list[str]
    timeout: float = DEFAULT_SEARCH_TIMEOUT["zhihu"]
    max_retries: int = 2
    retry_delay: float = 1.0
    stats: SearchStats = field(default_factory=lambda: SearchStats(backend="zhihu"))
    _next_secret_index: int = field(default=0, init=False, repr=False)
    _rate_limited_secret_indices: set[int] = field(
        default_factory=set, init=False, repr=False
    )
    _credential_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """清洗、去重凭证，并兼容直接传入单个字符串。"""
        raw_secrets = (
            [self.access_secrets]
            if isinstance(self.access_secrets, str)
            else self.access_secrets
        )
        secrets = list(
            dict.fromkeys(secret.strip() for secret in raw_secrets if secret.strip())
        )
        if not secrets:
            raise ValueError("至少需要一个知乎搜索 key")
        self.access_secrets = secrets

    @classmethod
    def from_env(
        cls, env_path: str | Path | None = None, **kwargs: Any
    ) -> "ZhihuSearchClient":
        """从逗号或换行分隔的环境变量读取一组搜索凭证。"""
        if env_path:
            load_dotenv(env_path)
        raw_secrets = (
            os.getenv("ZHIHU_SEARCH_KEYS")
            or os.getenv("ZHIHU_SEARCH_KEY")
            or os.getenv("ZHIHU_ACCESS_SECRET")
        )
        if not raw_secrets:
            raise ValueError(
                "请设置 ZHIHU_SEARCH_KEYS、ZHIHU_SEARCH_KEY 或 ZHIHU_ACCESS_SECRET"
            )
        secrets = [
            item.strip() for item in re.split(r"[,\n]", raw_secrets) if item.strip()
        ]
        return cls(access_secrets=secrets, **kwargs)

    def search(self, query: str) -> SearchResult:
        """轮转 key 搜索一个 query；429 切换 key，超时和 5xx 有限重试。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        saw_timeout = False
        saw_rate_limit = False
        credential = self._next_credential()
        if credential is None:
            return self._error_result(
                started, "all search keys are rate limited", 429
            )
        secret_index, access_secret = credential
        attempt = 0
        while True:
            try:
                result = self._request(query, started, access_secret)
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += result.latency
                    self.stats.result_count += len(result.items)
                return result
            except urllib.error.HTTPError as error:
                if error.code == 429 and not saw_rate_limit:
                    with self.stats._lock:
                        self.stats.rate_limits += 1
                    saw_rate_limit = True
                if error.code == 429:
                    self._disable_credential(secret_index)
                    credential = self._next_credential()
                    if credential is None:
                        return self._error_result(
                            started, "all search keys are rate limited", error.code
                        )
                    secret_index, access_secret = credential
                    with self.stats._lock:
                        self.stats.credential_failovers += 1
                    continue
                if error.code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {error.code}", error.code)
            except (TimeoutError, socket.timeout):
                if not saw_timeout:
                    with self.stats._lock:
                        self.stats.timeouts += 1
                    saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except urllib.error.URLError as error:
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    if not saw_timeout:
                        with self.stats._lock:
                            self.stats.timeouts += 1
                        saw_timeout = True
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        attempt += 1
                        continue
                    return self._error_result(started, "request timeout")
                return self._error_result(started, type(error).__name__)
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                return self._error_result(started, type(error).__name__)

    def _next_credential(self) -> tuple[int, str] | None:
        """按 round-robin 顺序取下一组尚未被 429 停用的凭证。"""
        secrets = cast(list[str], self.access_secrets)
        with self._credential_lock:
            for _ in range(len(secrets)):
                index = self._next_secret_index
                self._next_secret_index = (self._next_secret_index + 1) % len(secrets)
                if index not in self._rate_limited_secret_indices:
                    return index, secrets[index]
        return None

    def _disable_credential(self, index: int) -> None:
        """把返回 429 的 key 标记为本次运行不可再用。"""
        with self._credential_lock:
            self._rate_limited_secret_indices.add(index)

    def _request(
        self, query: str, started: float, access_secret: str
    ) -> SearchResult:
        """发出一次知乎 API 请求并解析真实响应结构。"""
        params = urllib.parse.urlencode(
            {"Query": query, "Count": 3, "SearchDB": "all"}
        )
        request = urllib.request.Request(
            f"{SEARCH_ENDPOINT}?{params}",
            headers={
                "Authorization": f"Bearer {access_secret}",
                "X-Request-Timestamp": str(int(time.time())),
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            items = [self._parse_item(item) for item in payload["Data"]["Items"]]
            return SearchResult(
                ok=True,
                items=items,
                latency=time.perf_counter() - started,
                status=response.status,
            )

    def _parse_item(self, item: dict[str, Any]) -> SearchItem:
        """从一条 API 结果中保留标题、摘要、来源和链接。"""
        source_parts = [str(item.get("ContentType") or "Zhihu")]
        if item.get("AuthorName"):
            source_parts.append(str(item["AuthorName"]))
        return SearchItem(
            title=str(item.get("Title") or "Untitled").strip(),
            content=str(item.get("ContentText") or "").strip()[:1200],
            url=str(item.get("Url") or "").strip(),
            source=" / ".join(source_parts),
        )

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换为不会泄露密钥的工具结果。"""
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


SearchClient = DeepSeekSearchClient | WikipediaSearchClient | ZhihuSearchClient


def resolve_search_concurrency(backend: str, value: int | None) -> int:
    """返回用户设置或当前后端的默认搜索并发。"""
    if backend not in SEARCH_BACKENDS:
        raise ValueError(f"不支持的搜索后端: {backend}")
    concurrency = DEFAULT_SEARCH_CONCURRENCY[backend] if value is None else value
    if concurrency < 1:
        raise ValueError("search_concurrency 必须大于等于 1")
    return concurrency


def resolve_search_timeout(backend: str, value: float | None) -> float:
    """返回用户设置或当前后端的默认请求超时。"""
    if backend not in SEARCH_BACKENDS:
        raise ValueError(f"不支持的搜索后端: {backend}")
    timeout = DEFAULT_SEARCH_TIMEOUT[backend] if value is None else value
    if timeout <= 0:
        raise ValueError("search_timeout 必须大于 0")
    return timeout


def create_search_client(
    backend: str,
    env_path: str | Path | None = None,
    *,
    model: str = "deepseek-v4-flash",
    timeout: float | None = None,
) -> SearchClient:
    """按名称创建搜索后端；三个后端共用同一套 rollout 接口。"""
    resolved_timeout = resolve_search_timeout(backend, timeout)
    if backend == "deepseek":
        return DeepSeekSearchClient.from_env(
            env_path,
            model=model,
            timeout=resolved_timeout,
        )
    if backend == "wikipedia":
        return WikipediaSearchClient(timeout=resolved_timeout)
    if backend == "zhihu":
        return ZhihuSearchClient.from_env(
            env_path,
            timeout=resolved_timeout,
        )
    raise ValueError(f"不支持的搜索后端: {backend}")


def parse_evidence(evidence: str | None) -> list[SearchItem]:
    """把 Evidence 模式的编号文本拆成可独立截断的证据条目。"""
    if not evidence:
        return []
    items = [
        SearchItem(title=title.strip(), content=content.strip())
        for title, content in EVIDENCE_PATTERN.findall(evidence.strip())
        if title.strip() and content.strip()
    ]
    if items:
        return items
    return [SearchItem(title="DeepSeek Web Search", content=evidence.strip())]


def format_item(item: SearchItem, index: int) -> str:
    """按后端提供的信息格式化一条完整工具证据。"""
    if item.source is None and item.url is None:
        return f"[{index}] Source: {item.title}\n    Evidence: {item.content}"
    lines = [
        f"[{index}] Title: {item.title}",
        f"    Content: {item.content}",
    ]
    if item.source:
        lines.append(f"    Source: {item.source}")
    if item.url:
        lines.append(f"    URL: {item.url}")
    return "\n".join(lines)
