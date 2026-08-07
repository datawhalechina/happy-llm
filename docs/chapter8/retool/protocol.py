"""定义 Qwen3.5 使用的代码解释器（code_interpreter）工具协议。

模型用 Qwen 原生 <tool_call> 格式调用
code_interpreter（参数就一个 code 字符串），沙箱执行结果以 role="tool" 消息返回；
最终答案用 \boxed{} 给出，且不允许同一轮里既调工具又给答案。
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        # 工具描述参照官方复现指南中实测"能涨点"的写法：
        # 强调 print 输出、每次执行独立无状态。
        "description": (
            "A Python code execution environment that allows you to:\n"
            "- Run Python code for calculations, data analysis, and other computational tasks\n"
            "- Get results through the `print()` function output\n"
            "- Execute code in a fresh process (each execution starts with no Python state)\n\n"
            "Important notes:\n"
            "- Results are captured from `print()` statements\n"
            "- Returns empty string if no output is printed\n"
            "- Each execution is independent (no state persistence between runs)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to be executed.",
                }
            },
            "required": ["code"],
        },
    },
}

SYSTEM_PROMPT = """You solve math problems step by step with help from a Python code interpreter.
Use the code_interpreter tool when calculation, symbolic manipulation, or enumeration helps you solve the problem accurately and quickly.

How to use the code_interpreter tool:
- Call code_interpreter with a `code` string containing Python code. Call it at most once per assistant turn, then wait for the execution result.
- Results are captured from what your code prints with print(). Always print the values you want to see.
- Each execution is independent: no variables, files, or state carry over between calls. Redefine everything you need in each piece of code.
- Code must finish within a few seconds and use little memory. Do not read or write files. If you enumerate or brute-force, keep the search space small.
- If the execution returns an error, analyze it and retry with corrected code when useful.

When you have the final answer, end with exactly one line in this format:
\\boxed{<your final answer>}
Do not call the tool and give the final answer in the same turn."""

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=code_interpreter>\s*<parameter=code>\s*(.*?)\s*"
    r"</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedAssistant:
    """保存一次 assistant 输出的协议解析结果。"""

    kind: str  # 解析类型："tool"、"answer" 或 "invalid"
    content: str  # 普通文本：工具调用前的推理，或完整答案/非法输出
    code: str | None = None  # kind 为 "tool" 时提取出的待执行 Python 代码


def initial_messages(question: str) -> list[dict[str, Any]]:
    """为一道数学题创建初始对话。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """渲染消息并把 tokenizer 的不同返回类型统一成一维 token 列表。"""
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[CODE_TOOL],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """用模型原生 chat template 构建带工具定义的生成 prompt。"""
    return _render_chat(tokenizer, messages, add_generation_prompt=True)


def _encoded_text_tokens(tokenizer: Any, text: str) -> list[int]:
    """把普通文本编码结果统一成一维 token 列表。"""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _suffix_prefix_overlap(tokens: list[int], suffix: list[int]) -> int:
    """返回 tokens 末尾与 suffix 开头的最长重叠长度。"""
    for length in range(min(len(tokens), len(suffix)), 0, -1):
        if tokens[-length:] == suffix[:length]:
            return length
    return 0


def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    """用真实采样 token 接上 assistant 结束符和新的 tool observation。

    与 search-r1 相同：不重渲染整段历史（token-in token-out），
    只拼接"assistant 结束符 + tool observation"的增量 token，
    避免 text↔token 重编码不可逆导致的训练错位（官方实测的坑）。

    增量片段只与 tool 消息有关、与 assistant 内容无关，因此 canonical 计算
    一律使用占位内容：采样文本可能含 </think> 等会被模板重构为
    reasoning/content 两段的内容，用真实文本
    定位结束边界不可靠。
    """
    canonical_prompt = build_prompt(tokenizer, messages_before_assistant)
    placeholder_message = {"role": "assistant", "content": "x"}
    messages_with_assistant = [*messages_before_assistant, placeholder_message]
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )
    placeholder_tokens = _encoded_text_tokens(tokenizer, "x")
    canonical_action = [*canonical_prompt, *placeholder_tokens]
    if canonical_assistant_end[: len(canonical_action)] != canonical_action:
        raise ValueError("chat template 无法定位 assistant 结束边界（占位内容也不匹配）")
    assistant_closing_tokens = canonical_assistant_end[len(canonical_action) :]

    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    if canonical_next_prompt[: len(canonical_assistant_end)] != canonical_assistant_end:
        raise ValueError("加入 tool observation 后 chat template 改写了历史消息")
    observation_tokens = canonical_next_prompt[len(canonical_assistant_end) :]

    # sampler 可能已经返回部分或全部 assistant 结束符，只补尚未包含的部分。
    overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]


def parse_assistant(text: str) -> ParsedAssistant:
    """把 assistant 文本识别成代码调用、最终回答或非法输出。

    合法工具调用要求：恰好一个 <tool_call>，且其后不再有其他内容。
    代码里允许出现任意字符（Python 本就有 <、>、引号），只校验非空。
    """
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        kind = "invalid" if "<tool_call>" in text else "answer"
        return ParsedAssistant(kind=kind, content=text.strip())
    if len(matches) != 1 or text[matches[0].end() :].strip():
        return ParsedAssistant(kind="invalid", content=text.strip())
    code = matches[0].group(1).strip()
    if not code:
        return ParsedAssistant(kind="invalid", content=text.strip())
    content = text[: matches[0].start()].strip()
    return ParsedAssistant(kind="tool", content=content, code=code)


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """构造一条结构化代码执行结果消息。"""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "code_interpreter",
        "content": content,
    }


def stop_sequences(tokenizer: Any) -> list[str]:
    """返回模型结束一轮 assistant 输出时使用的停止字符串。"""
    eos_token = getattr(tokenizer, "eos_token", None)
    return [eos_token] if eos_token else []
