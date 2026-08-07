"""本地 Python 代码执行沙箱：ReTool rollout 的 code_interpreter 工具后端。

设计原理
========

执行模型：模型在 <tool_call> 中给出的 Python 代码，用
`python -B -c <bootstrap>` 起全新子进程执行。list 形式 argv 不经过 shell，
代码经 repr() 内嵌进引导串——不落脚本文件、无注入面；每次调用一个全新
进程，天然满足官方"每次执行独立无状态"的语义。

四道保险：

1. 限并发：threading.BoundedSemaphore（默认 8）。不用 asyncio.Semaphore，
   因为它会绑定到单个事件循环，而训练循环每个 step/round 都会新建 loop，
   跨 loop 复用会直接报错。
2. 限时间：wall-clock 超时（默认 30s，对齐官方 recipe）后 os.killpg 杀死
   整个进程组（start_new_session=True 才能整组杀，防止 fork 出孤儿进程）；
   另有子进程引导串里的 RLIMIT_CPU 作为内核级 CPU 时间兜底。
3. 限输出：stdout/stderr 写入匿名临时文件而不是 PIPE，防止 print 洪水撑爆
   父进程内存；读回时只取文件尾部并按 4096 字符截断——执行结果和
   traceback 的关键信息都在末尾。
4. 限线程：子进程环境把 OMP/OpenBLAS/MKL/VECLIB 线程数压到 1，防止多个
   并发沙箱各自拉起一整组 BLAS 线程把机器打满。

结果哲学：成功回 stdout（没有 print 则为空串，官方语义）；失败回原始
stderr（完整 traceback）；超时回超时说明。报错不修饰、原样回喂——
论文中"自我纠错"行为正是从看到错误反馈再改代码中涌现的。
唯一处理：tool 文本中的模板特殊标记（`<|im_end|>`、`</tool_response>` 等）
会替换成无害写法，防止污染 observation 结构。

正式训练前应验证正常输出、auto-print、语法错误、运行时异常、超时杀进程、
输出截断、numpy 导入、空输出语义与异步并发。
"""

import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any


DEFAULT_TIMEOUT = 30.0  # 单次代码调用允许的最大 wall-clock 秒数。
MAX_OUTPUT_CHARS = 4096  # stdout/stderr 各自允许保留的最大字符数。
READ_CAP_BYTES = MAX_OUTPUT_CHARS * 4  # 从输出文件读取的最大字节数（预留多字节字符）。

# 子进程引导代码：先限制 CPU 时间，再 exec 模型代码。
# 用 repr 内嵌代码，避免任何转义问题；__name__ 保持 "__main__" 语义。
_BOOTSTRAP = (
    "import resource\n"
    "resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))\n"
    "exec(compile({code}, '<sandbox>', 'exec'))\n"
)

# 子进程环境：保留基础环境，但把常见数值库的线程数压到 1，
# 防止多个并发沙箱各自拉起一整组 BLAS 线程把机器打满。
_CHILD_ENV_OVERRIDES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class ExecResult:
    """保存一次代码执行的结果。"""

    ok: bool  # 进程正常结束且 returncode 为 0。
    stdout: str  # 截断后的标准输出。
    stderr: str  # 截断后的标准错误。
    returncode: int | None  # 进程退出码；未正常退出时为 None。
    timed_out: bool  # 是否因 wall-clock 超时被杀死。
    latency: float  # 从启动到回收的耗时（秒）。


@dataclass
class SandboxStats:
    """累计沙箱调用的运行指标。"""

    calls: int = 0
    successes: int = 0
    errors: int = 0  # 进程返回非零退出码（语法错误、异常等）。
    timeouts: int = 0
    latency_total: float = 0.0

    def metrics(self) -> dict[str, float]:
        """把累计计数转换成便于 SwanLab 记录的比例。"""
        denominator = max(self.calls, 1)
        return {
            "sandbox/success_rate": self.successes / denominator,
            "sandbox/error_rate": self.errors / denominator,
            "sandbox/timeout_rate": self.timeouts / denominator,
            "sandbox/latency": self.latency_total / denominator,
        }


def ensure_trailing_print(code: str) -> str:
    """给最后一行非空行补 print()，与官方 recipe 的 auto-print 技巧一致。"""
    lines = code.split("\n")
    for index in range(len(lines) - 1, -1, -1):
        if lines[index] == "":
            continue
        if not lines[index].startswith("print"):
            lines[index] = f"print({lines[index]})"
        break
    return "\n".join(lines)


def _read_capped(file: Any, limit: int = READ_CAP_BYTES) -> str:
    """只从输出文件尾部读取有限字节，避免 print 洪水撑爆内存。"""
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(max(0, size - limit))
    return file.read().decode("utf-8", errors="replace")


def _truncate_tail(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """按尾部保留输出；模型的关键结果通常打在末尾。"""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"[... truncated {omitted} chars ...]\n{text[-limit:]}"


# 工具返回文本里的模板特殊标记会污染 observation 结构
# （chat template 渲染消息内容时不做转义），统一替换成无害写法。
_SANITIZE_REPLACEMENTS = {
    "<|im_start|>": "< im_start >",
    "<|im_end|>": "< im_end >",
    "<tool_response>": "< tool_response >",
    "</tool_response>": "< /tool_response>",
}


def sanitize_tool_content(text: str) -> str:
    """把 tool 返回文本中的模板特殊标记替换成无害写法。"""
    for marker, replacement in _SANITIZE_REPLACEMENTS.items():
        text = text.replace(marker, replacement)
    return text


class LocalPythonSandbox:
    """用本机 Python 解释器执行模型生成的代码，带超时和资源限制。"""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_workers: int = 8,
        python: str = sys.executable,
    ) -> None:
        """保存执行参数，并创建并发信号量与统计器。"""
        self.timeout = timeout
        self.python = python
        self.stats = SandboxStats()
        # 用线程信号量而不是 asyncio.Semaphore：后者会绑定到单个事件循环，
        # 而训练循环每个 step、每个 round 都会新建 loop，跨 loop 复用会报错。
        self._semaphore = threading.BoundedSemaphore(max_workers)

    def run_code(self, code: str) -> ExecResult:
        """同步执行一段代码，返回截断后的输出与执行状态。"""
        started = time.perf_counter()
        self.stats.calls += 1
        bootstrap = _BOOTSTRAP.format(cpu=int(self.timeout) + 5, code=repr(code))
        env = {**os.environ, **_CHILD_ENV_OVERRIDES}
        with self._semaphore:
            # 输出写入匿名临时文件而不是 PIPE，防止子进程输出洪水撑爆父进程内存。
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                process = subprocess.Popen(
                    [self.python, "-B", "-c", bootstrap],
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=env,
                    start_new_session=True,  # 独立进程组，超时才能整组杀死
                )
                timed_out = False
                try:
                    returncode = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = None
                    process.wait()  # 回收僵尸进程
                stdout = _truncate_tail(_read_capped(stdout_file))
                stderr = _truncate_tail(_read_capped(stderr_file))

        latency = time.perf_counter() - started
        ok = not timed_out and returncode == 0
        if timed_out:
            self.stats.timeouts += 1
        elif ok:
            self.stats.successes += 1
        else:
            self.stats.errors += 1
        self.stats.latency_total += latency
        return ExecResult(ok, stdout, stderr, returncode, timed_out, latency)

    async def arun_code(self, code: str) -> ExecResult:
        """在线程池中异步执行代码（并发由 run_code 内的线程信号量限流）。"""
        return await asyncio.to_thread(self.run_code, code)

    def format_tool_content(self, result: ExecResult) -> str:
        """把执行结果格式化成喂给模型的 tool 消息内容（特殊标记已无害化）。"""
        if result.timed_out:
            return f"Error: execution timed out after {self.timeout:.0f} seconds."
        if result.ok:
            return sanitize_tool_content(result.stdout)  # 无 print 时为空字符串（官方语义）
        if result.stderr:
            return sanitize_tool_content(result.stderr)
        return f"Error: process exited with code {result.returncode}."
