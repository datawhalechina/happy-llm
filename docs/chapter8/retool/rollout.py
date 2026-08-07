"""执行 ReTool 的多轮代码交织（code-interlaced）轨迹。

整体结构与本章 Search-R1 的 rollout 一致：
- 首轮每个问题用 num_samples=group_size 一次分叉出整组轨迹
- 后续每条未结束轨迹单独采样，直到给出最终答案或耗尽预算
- 下一轮 prompt 用真实采样 token 连续构造（token-in token-out）
- 同 round 内多条轨迹的代码调用用 asyncio 并发执行（代码执行最长 30s，
  不能像 search-r1 的 HTTP 搜索那样串行处理）
"""

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import pytrio as trio

from data import MathExample
from protocol import (
    build_next_prompt,
    build_prompt,
    initial_messages,
    parse_assistant,
    stop_sequences,
    tool_message,
)
from reward import score_answer
from sandbox import ExecResult, LocalPythonSandbox


@dataclass(frozen=True)
class RolloutConfig:
    """保存轨迹长度和采样参数。"""

    group_size: int = 8  # 同一问题一次采样的轨迹数。
    max_code_calls: int = 4  # 每条轨迹最多调用代码解释器的次数。
    max_assistant_turns: int = 6  # 每条轨迹最多生成的 assistant 轮数。
    max_trajectory_tokens: int = 8192  # 整条轨迹允许的最大 token 数。
    max_assistant_tokens: int = 1024  # assistant 单轮最多生成的 token 数。
    max_tool_response_tokens: int = 512  # 单次代码执行结果允许的最大 token 数。
    temperature: float = 1.0  # 控制采样随机性。
    top_p: float = 1.0  # 核采样的累积概率上限。
    seed: int = 42  # 采样使用的随机种子。


@dataclass
class AssistantTurn:
    """保存一个可训练的 assistant 动作。"""

    prompt_tokens: list[int]  # 本轮生成前的完整输入 token。
    completion_tokens: list[int]  # 本轮 assistant 生成的 token。
    logprobs: list[float]  # 采样时每个生成 token 的旧策略 logprob。
    text: str  # 本轮生成 token 解码后的文本。


@dataclass
class Trajectory:
    """保存一条完整代码交织轨迹及其训练信号。"""

    example: MathExample  # 当前轨迹对应的问题和参考答案。
    group_index: int  # 当前轨迹在同题采样组中的编号。
    messages: list[dict[str, Any]]  # 当前轨迹的完整多轮消息。
    # 工具返回后供下一轮直接使用的连续 token。
    next_prompt_tokens: list[int] | None = None
    question_index: int = 0  # 当前问题在 rollout batch 中的编号。
    turns: list[AssistantTurn] = field(default_factory=list)  # 已生成的 assistant 轮次。
    code_calls: int = 0  # 已经调用代码解释器的次数。
    final_text: str = ""  # 轨迹结束时的最终 assistant 输出。
    reward: float = -1.0  # 最终答案获得的奖励（官方 ±1 规则）。
    advantage: float = 0.0  # 减去同题组平均奖励后的 advantage。
    valid_format: bool = False  # 最终答案是否含合法 \boxed{}。
    correct: bool = False  # 最终答案是否通过数学等价判定。
    done: bool = False  # 当前轨迹是否已经结束。


@dataclass(frozen=True)
class SampleRequest:
    """描述一批共享 prompt 的采样请求。"""

    trajectory_index: int  # 采样结果需要写回的轨迹编号。
    prompt_tokens: list[int]  # 本次采样的模型输入 token。
    num_samples: int  # 共需要采样的候选数量。
    max_tokens: int  # 每个候选最多生成的 token 数。
    seed: int  # 本次采样使用的随机种子。


@dataclass(frozen=True)
class PendingExecution:
    """保存一次等待执行的代码调用及其上下文。"""

    trajectory: Trajectory  # 发起调用的轨迹（执行结果要写回它）。
    code: str  # 待执行的 Python 代码。
    call_id: str  # 本次调用在消息历史里的 id。
    messages_before_assistant: list[dict[str, Any]]  # assistant 输出前的消息历史。
    assistant_text: str  # 发起调用的 assistant 输出文本。
    prompt_tokens: list[int]  # 本轮生成前的输入 token。
    completion_tokens: list[int]  # 本轮生成的 token。


async def sample_requests_async(
    sampling_client: Any,
    requests: list[SampleRequest],
    config: RolloutConfig,
    tokenizer: Any,
) -> list[Any]:
    """只并发执行 PyTRIO 的 sample_async 请求。"""
    tasks = []
    for request in requests:
        params = trio.SamplingParams(
            max_tokens=request.max_tokens,
            seed=request.seed,
            stop=stop_sequences(tokenizer),
            temperature=config.temperature,
            top_p=config.top_p,
        )
        tasks.append(
            sampling_client.sample_async(
                prompt=trio.ModelInput.from_ints(request.prompt_tokens),
                num_samples=request.num_samples,
                sampling_params=params,
                return_text=True,
            )
        )
    return list(await asyncio.gather(*tasks))


async def execute_pending_async(
    pendings: list[PendingExecution],
    sandbox: LocalPythonSandbox,
) -> list[ExecResult]:
    """并发执行同一 round 内所有轨迹的代码调用（受沙箱并发池限流）。"""
    return list(await asyncio.gather(*(sandbox.arun_code(p.code) for p in pendings)))


def token_count(tokenizer: Any, text: str) -> int:
    """统计一段普通文本的 token 数。"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def fit_tool_content(
    tokenizer: Any,
    content: str,
    pending: PendingExecution,
    config: RolloutConfig,
) -> tuple[str, list[int]] | None:
    """在工具输出和总轨迹预算内截断执行结果；保留尾部（结果/报错关键在末尾）。"""
    while True:
        if token_count(tokenizer, content) <= config.max_tool_response_tokens:
            next_tool_message = tool_message(pending.call_id, content)
            next_prompt = build_next_prompt(
                tokenizer,
                pending.messages_before_assistant,
                pending.prompt_tokens,
                pending.completion_tokens,
                next_tool_message,
            )
            if len(next_prompt) <= config.max_trajectory_tokens:
                return content, next_prompt
        if len(content) <= 64:
            return None
        content = "[... truncated ...]\n" + content[-int(len(content) * 0.7) :]


def make_request(
    tokenizer: Any,
    trajectory: Trajectory,
    trajectory_index: int,
    num_samples: int,
    seed: int,
    config: RolloutConfig,
) -> SampleRequest | None:
    """按当前真实 prompt 长度创建动态采样请求。"""
    prompt_tokens = (
        trajectory.next_prompt_tokens
        if trajectory.next_prompt_tokens is not None
        else build_prompt(tokenizer, trajectory.messages)
    )
    max_tokens = min(
        config.max_assistant_tokens,
        config.max_trajectory_tokens - len(prompt_tokens),
    )
    if max_tokens <= 0:
        trajectory.done = True
        return None
    return SampleRequest(trajectory_index, prompt_tokens, num_samples, max_tokens, seed)


def read_sequence(sequence: Any, tokenizer: Any) -> tuple[list[int], list[float], str]:
    """读取采样序列并校验 token 与旧策略 logprob 对齐。"""
    tokens = [int(token) for token in sequence.tokens]
    logprobs = [float(value) for value in sequence.logprobs]
    if len(tokens) != len(logprobs):
        raise ValueError("采样 token 与 logprob 长度不一致")
    text = sequence.text
    if text is None:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokens, logprobs, text


def begin_advance(
    trajectory: Trajectory,
    prompt_tokens: list[int],
    sequence: Any,
    tokenizer: Any,
    config: RolloutConfig,
) -> PendingExecution | None:
    """消费一次 assistant 输出；需要执行代码时返回待执行请求，否则结束轨迹。"""
    tokens, logprobs, text = read_sequence(sequence, tokenizer)
    # chat template 会 strip assistant 内容；消息历史、解析与结束边界计算与之保持一致。
    text = text.strip()
    trajectory.turns.append(AssistantTurn(prompt_tokens, tokens, logprobs, text))
    parsed = parse_assistant(text)

    can_code = (
        parsed.kind == "tool"
        and trajectory.code_calls < config.max_code_calls
        and len(trajectory.turns) < config.max_assistant_turns
    )
    if not can_code:
        trajectory.messages.append({"role": "assistant", "content": text})
        trajectory.final_text = text
        trajectory.done = True
        return None

    call_id = (
        f"code-{trajectory.question_index}-{trajectory.group_index}-"
        f"{trajectory.code_calls + 1}"
    )
    messages_before_assistant = list(trajectory.messages)
    # messages 用于协议和结果记录；下一轮模型输入则由真实采样 token 连续构造。
    trajectory.messages.append({"role": "assistant", "content": text})
    return PendingExecution(
        trajectory,
        parsed.code or "",
        call_id,
        messages_before_assistant,
        text,
        prompt_tokens,
        tokens,
    )


def finish_advance(
    pending: PendingExecution,
    result: ExecResult,
    tokenizer: Any,
    sandbox: LocalPythonSandbox,
    config: RolloutConfig,
) -> None:
    """写入代码执行结果并构造下一轮 prompt；预算放不下则结束轨迹。"""
    trajectory = pending.trajectory
    content = sandbox.format_tool_content(result)
    fitted = fit_tool_content(tokenizer, content, pending, config)
    if fitted is None:
        trajectory.final_text = pending.assistant_text
        trajectory.done = True
        return
    content, next_prompt_tokens = fitted
    trajectory.messages.append(tool_message(pending.call_id, content))
    trajectory.next_prompt_tokens = next_prompt_tokens
    trajectory.code_calls += 1


def score_trajectory(trajectory: Trajectory) -> None:
    """给完成的轨迹写入格式与数学等价奖励。"""
    result = score_answer(trajectory.final_text, trajectory.example.answer)
    trajectory.reward = result.reward
    trajectory.valid_format = result.valid_format
    trajectory.correct = result.correct


def assign_group_advantages(trajectories: list[Trajectory]) -> None:
    """按同题 group 的平均 reward 计算中心化 advantage（原地写入）。

    整组 advantage 全零的 degenerate 计数统一由 train.py 的
    degenerate_group_count 统计，这里不重复计算。
    """
    groups: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.question_index, []).append(trajectory)
    for group in groups.values():
        mean_reward = sum(item.reward for item in group) / len(group)
        for item in group:
            item.advantage = item.reward - mean_reward


async def advance_round(
    responses: list[Any],
    requests: list[SampleRequest],
    trajectories: list[Trajectory],
    tokenizer: Any,
    sandbox: LocalPythonSandbox,
    config: RolloutConfig,
    progress_callback: Callable[[int], None] | None,
) -> None:
    """消费一轮采样结果：先解析出所有代码调用并发执行，再统一写回结果。"""
    pendings: list[PendingExecution] = []
    for request, response in zip(requests, responses, strict=True):
        trajectory = trajectories[request.trajectory_index]
        pending = begin_advance(
            trajectory,
            request.prompt_tokens,
            response.sequences[0],
            tokenizer,
            config,
        )
        if pending is not None:
            pendings.append(pending)
        elif trajectory.done and progress_callback is not None:
            progress_callback(1)
    if not pendings:
        return
    results = await execute_pending_async(pendings, sandbox)
    for pending, result in zip(pendings, results, strict=True):
        finish_advance(pending, result, tokenizer, sandbox, config)
        if pending.trajectory.done and progress_callback is not None:
            progress_callback(1)


def rollout_batch(
    sampling_client: Any,
    tokenizer: Any,
    sandbox: LocalPythonSandbox,
    examples: list[MathExample],
    config: RolloutConfig,
    progress_callback: Callable[[int], None] | None = None,
) -> list[Trajectory]:
    """同步入口：整个 rollout 在一个事件循环内驱动（采样与代码执行并发）。

    只能在同步上下文调用；已处于事件循环中时（如 eval.py）请改用
    rollout_batch_async 并 await。
    """
    return asyncio.run(
        rollout_batch_async(
            sampling_client,
            tokenizer,
            sandbox,
            examples,
            config,
            progress_callback,
        )
    )


async def rollout_batch_async(
    sampling_client: Any,
    tokenizer: Any,
    sandbox: LocalPythonSandbox,
    examples: list[MathExample],
    config: RolloutConfig,
    progress_callback: Callable[[int], None] | None = None,
) -> list[Trajectory]:
    """异步控制多轮状态机，并在轨迹结束时可选上报完成数量。"""
    # 每个问题先创建一条共享初始 prompt 的根轨迹。
    roots = [
        Trajectory(
            example=example,
            group_index=0,
            messages=initial_messages(example.question),
            question_index=question_index,
        )
        for question_index, example in enumerate(examples)
    ]

    # 首轮每个问题只发一个请求，用 num_samples=group_size 生成多个分支。
    first_requests: list[SampleRequest] = []
    for index, trajectory in enumerate(roots):
        request = make_request(
            tokenizer,
            trajectory,
            index,
            config.group_size,
            config.seed + index,
            config,
        )
        if request:
            first_requests.append(request)

    trajectories: list[Trajectory] = []
    if first_requests:
        # 不同问题的首轮模型采样并发执行。
        responses = await sample_requests_async(
            sampling_client, first_requests, config, tokenizer
        )
        first_pendings: list[PendingExecution] = []
        for request, response in zip(first_requests, responses, strict=True):
            root = roots[request.trajectory_index]
            if len(response.sequences) != config.group_size:
                raise ValueError("首轮采样数量与 group_size 不一致")
            for group_index, sequence in enumerate(response.sequences):
                # 深拷贝保证每个分支拥有独立的 messages 和执行历史。
                branch = copy.deepcopy(root)
                branch.group_index = group_index
                trajectories.append(branch)
                pending = begin_advance(
                    branch,
                    request.prompt_tokens,
                    sequence,
                    tokenizer,
                    config,
                )
                if pending is not None:
                    first_pendings.append(pending)
                elif branch.done and progress_callback is not None:
                    progress_callback(1)
        if first_pendings:
            results = await execute_pending_async(first_pendings, sandbox)
            for pending, result in zip(first_pendings, results, strict=True):
                finish_advance(pending, result, tokenizer, sandbox, config)
                if pending.trajectory.done and progress_callback is not None:
                    progress_callback(1)

    # 首轮分叉后 prompt 已经不同，后续每条未结束轨迹单独采样一个结果。
    while any(not trajectory.done for trajectory in trajectories):
        requests: list[SampleRequest] = []
        for index, trajectory in enumerate(trajectories):
            if trajectory.done:
                continue
            request = make_request(
                tokenizer,
                trajectory,
                index,
                1,
                config.seed + index + len(trajectory.turns) * 10_000,
                config,
            )
            if request:
                requests.append(request)
            elif trajectory.done and progress_callback is not None:
                # make_request 会在轨迹 token 预算耗尽时直接将其标记为结束。
                progress_callback(1)
        if not requests:
            break
        # 各轨迹 prompt 不同，但仍通过多个 sample_async 并发采样。
        responses = await sample_requests_async(sampling_client, requests, config, tokenizer)
        for request, response in zip(requests, responses, strict=True):
            if len(response.sequences) != 1:
                raise ValueError("后续轮次每条轨迹必须只采样一个分支")
        await advance_round(
            responses,
            requests,
            trajectories,
            tokenizer,
            sandbox,
            config,
            progress_callback,
        )

    # 所有轨迹结束后先计算奖励，再按同题 group 计算中心化 advantage。
    for trajectory in trajectories:
        score_trajectory(trajectory)
    assign_group_advantages(trajectories)
    return trajectories
