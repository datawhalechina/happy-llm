"""使用 PyTRIO 在 AIME 2025 上评测 ReTool（retool / text-only 两种模式）。

retool 模式（多轮代码工具 rollout，复用 rollout.py 状态机）：
uv run python eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --output eval-results/aime25-retool-base.jsonl

评测训练好的 sampler weights 时加：
uv run python eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --model-path trio://<your_sampler_weights_path> \
    --output eval-results/aime25-retool-stepxx.jsonl

脚本报告 Average@N（论文 pass@1 的估计方式）、Pass@N、boxed format rate
和代码调用统计。评测集复用 04-opsd/datasets/aime_2025（30 题，答案均为整数）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
import pytrio as trio
from tqdm import tqdm

from data import MathExample
from rollout import RolloutConfig, rollout_batch_async
from sandbox import LocalPythonSandbox


trio.configure(sampling_timeout=18000,)  # 5 小时，足够多轮评测 30 道题

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR.parent / "04-opsd" / "datasets" / "aime_2025"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "eval-results" / "aime25.jsonl"
EXPECTED_ROWS = 30


def parse_args() -> argparse.Namespace:
    """解析并校验评测参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["retool", "text-only"],
        default="retool",
        help="retool=多轮代码工具 rollout；text-only=禁用工具的单轮推理（对照组）",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="04-opsd 保存的 AIME25 数据目录",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="PyTRIO sampling client 的基础模型",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="save_weights_for_sampler 返回的 trio:// 路径；留空评测 base model",
    )
    parser.add_argument("--val-n", type=int, default=12, help="每道题生成多少条轨迹")
    parser.add_argument(
        "--limit", type=int, default=0, help="只评测前 N 题；0 表示全部 30 题"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="retool 模式每次 rollout_batch 处理的题目数（逐 chunk 落盘）",
    )
    parser.add_argument("--concurrency", type=int, default=15, help="text-only 模式并发题目数")
    parser.add_argument(
        "--max-tokens", type=int, default=8192, help="text-only 模式单答案最大 token 数"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7, help="对齐论文评测配置")
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-code-calls", type=int, default=4)
    parser.add_argument("--max-assistant-turns", type=int, default=6)
    parser.add_argument("--max-trajectory-tokens", type=int, default=8192)
    parser.add_argument("--max-assistant-tokens", type=int, default=1024)
    parser.add_argument("--max-tool-response-tokens", type=int, default=512)
    parser.add_argument("--sandbox-timeout", type=float, default=30.0)
    parser.add_argument("--sandbox-workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="逐题 JSONL 输出；最后一行为 summary",
    )
    args = parser.parse_args()

    for name in ("val_n", "concurrency", "max_tokens", "chunk_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    return args


def load_aime25(path: Path, limit: int) -> Dataset:
    """读取本地 AIME25，验证结构后按需截取少量题目。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 AIME25 数据：{path}\n"
            "请先运行：uv run python 04-opsd/00-datasets.py --only aime25"
        )
    loaded = load_from_disk(str(path))
    dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    if not isinstance(dataset, Dataset):
        raise TypeError(f"期望 Dataset，实际得到 {type(dataset)!r}")
    missing = sorted({"problem", "answer"} - set(dataset.column_names))
    if missing:
        raise ValueError(
            f"AIME25 缺少字段 {missing}，实际字段为 {dataset.column_names}"
        )
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"AIME25 应有 {EXPECTED_ROWS} 题，实际为 {len(dataset)} 题")
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def extract_last_boxed(text: str) -> str | None:
    """提取最后一个花括号完整闭合的 ``\\boxed{...}``，支持嵌套花括号。"""
    end = len(text)
    while True:
        start = text.rfind("\\boxed", 0, end)
        if start < 0:
            return None
        left = text.find("{", start)
        if left < 0:
            end = start
            continue
        depth = 0
        for index in range(left, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[left + 1 : index].strip()
        end = start


def normalize_aime_answer(answer: str | None) -> str | None:
    """将 AIME 答案规范为无前导零的整数字符串。"""
    if answer is None:
        return None
    cleaned = answer.strip().replace(",", "").replace("$", "")
    cleaned = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", cleaned)
    match = re.fullmatch(r"\s*([+-]?\d+)\s*", cleaned)
    if match is None:
        return None
    return str(int(match.group(1)))


def build_text_prompt_ids(tokenizer: Any, problem: str) -> list[int]:
    """text-only 模式：与 04-opsd 评测相同的单轮推理 prompt。"""
    content = (
        f"{problem.strip()}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("AIME25 prompt token 为空")
    return prompt_ids


async def evaluate_problem_text_only(
    index: int,
    row: dict[str, Any],
    sampling_client: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """text-only 模式：并发采样一道题的 N 个单轮答案。"""
    problem = str(row["problem"]).strip()
    ground_truth = normalize_aime_answer(str(row["answer"]))
    if ground_truth is None:
        raise ValueError(f"AIME25 第 {index} 题 ground truth 不是整数: {row['answer']!r}")

    prompt_ids = build_text_prompt_ids(tokenizer, problem)
    params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        seed=args.seed + index,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=list(
            dict.fromkeys(
                token for token in (tokenizer.eos_token, "<|im_end|>") if token
            )
        ),
    )
    async with semaphore:
        response = await sampling_client.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_ids),
            num_samples=args.val_n,
            sampling_params=params,
            return_text=True,
        )
    if len(response.sequences) != args.val_n:
        raise RuntimeError(
            f"AIME25 第 {index} 题请求 {args.val_n} 条 completion，"
            f"实际返回 {len(response.sequences)} 条"
        )

    generations = []
    for sequence in response.sequences:
        text = sequence.text
        if text is None:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
        boxed = extract_last_boxed(text)
        predicted = normalize_aime_answer(boxed)
        generations.append(
            {
                "predicted_answer": predicted,
                "boxed_answer": boxed,
                "correct": predicted == ground_truth,
                "formatted": boxed is not None,
                "code_calls": 0,
                "turns": 1,
                "completion_tokens": len(sequence.tokens),
                "text": text,
            }
        )

    return {
        "type": "problem",
        "problem_id": int(row.get("id", index)),
        "problem": problem,
        "ground_truth": ground_truth,
        "val_n": args.val_n,
        "num_correct": sum(int(item["correct"]) for item in generations),
        "pass_at_n": any(item["correct"] for item in generations),
        "generations": generations,
    }


async def evaluate_chunk_retool(
    chunk: list[tuple[int, dict[str, Any]]],
    sampling_client: Any,
    tokenizer: Any,
    sandbox: LocalPythonSandbox,
    args: argparse.Namespace,
    progress: tqdm,
) -> list[dict[str, Any]]:
    """retool 模式：对一组题目跑多轮代码工具 rollout 并汇总逐题结果。"""
    examples = [
        MathExample(
            id=str(row.get("id", index)),
            question=str(row["problem"]).strip(),
            answer=str(row["answer"]),
            data_source="aime_2025",
        )
        for index, row in chunk
    ]
    config = RolloutConfig(
        group_size=args.val_n,
        max_code_calls=args.max_code_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    trajectories = await rollout_batch_async(
        sampling_client,
        tokenizer,
        sandbox,
        examples,
        config,
        progress_callback=progress.update,
    )

    results: list[dict[str, Any]] = []
    for question_index, (index, row) in enumerate(chunk):
        group = [
            trajectory
            for trajectory in trajectories
            if trajectory.question_index == question_index
        ]
        ground_truth = normalize_aime_answer(str(row["answer"]))
        generations = [
            {
                "predicted_answer": normalize_aime_answer(
                    extract_last_boxed(trajectory.final_text)
                ),
                "boxed_answer": extract_last_boxed(trajectory.final_text),
                "correct": trajectory.correct,
                "formatted": trajectory.valid_format,
                "code_calls": trajectory.code_calls,
                "turns": len(trajectory.turns),
                "completion_tokens": sum(
                    len(turn.completion_tokens) for turn in trajectory.turns
                ),
                "text": trajectory.final_text,
            }
            for trajectory in group
        ]
        results.append(
            {
                "type": "problem",
                "problem_id": int(row.get("id", index)),
                "problem": str(row["problem"]).strip(),
                "ground_truth": ground_truth,
                "val_n": args.val_n,
                "num_correct": sum(int(item["correct"]) for item in generations),
                "pass_at_n": any(item["correct"] for item in generations),
                "generations": generations,
            }
        )
    return results


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """聚合逐题结果，生成 Average@N、Pass@N、格式率和代码调用统计。"""
    total_problems = len(results)
    generations = [
        generation for item in results for generation in item["generations"]
    ]
    total_generations = len(generations)
    total_correct = sum(item["num_correct"] for item in results)
    total_formatted = sum(int(generation["formatted"]) for generation in generations)
    pass_count = sum(int(item["pass_at_n"]) for item in results)
    return {
        "type": "summary",
        "dataset": "yentinglin/aime_2025",
        "mode": args.mode,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "val_n": args.val_n,
        "problems": total_problems,
        "generations": total_generations,
        "average_at_n": total_correct / total_generations if total_generations else 0.0,
        "pass_at_n": pass_count / total_problems if total_problems else 0.0,
        "format_rate": total_formatted / total_generations if total_generations else 0.0,
        "mean_code_calls": (
            sum(float(generation["code_calls"]) for generation in generations)
            / total_generations
            if total_generations
            else 0.0
        ),
        "mean_turns": (
            sum(float(generation["turns"]) for generation in generations)
            / total_generations
            if total_generations
            else 0.0
        ),
        "correct_generations": total_correct,
        "passed_problems": pass_count,
    }


def write_results(
    results: list[dict[str, Any]], output: Path, *, append: bool
) -> None:
    """把逐题结果追加写入 JSONL 文件。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if append else "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


async def evaluate(args: argparse.Namespace) -> None:
    """创建 PyTRIO 采样客户端，按模式完成评测和结果落盘。"""
    dataset = load_aime25(args.dataset_path, args.limit)
    service_client = trio.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    tokenizer = sampling_client.get_tokenizer()

    results: list[dict[str, Any]] = []
    if args.mode == "text-only":
        semaphore = asyncio.Semaphore(args.concurrency)
        with tqdm(total=len(dataset), desc="AIME25 text-only", unit="problem") as progress:

            async def evaluate_and_track(index: int, row: dict[str, Any]) -> dict[str, Any]:
                result = await evaluate_problem_text_only(
                    index, row, sampling_client, tokenizer, args, semaphore
                )
                progress.update(1)
                return result

            results = list(
                await asyncio.gather(
                    *(evaluate_and_track(index, row) for index, row in enumerate(dataset))
                )
            )
        write_results(results, args.output, append=False)
    else:
        sandbox = LocalPythonSandbox(
            timeout=args.sandbox_timeout,
            max_workers=args.sandbox_workers,
        )
        indexed_rows = list(enumerate(dataset))
        chunks = [
            indexed_rows[offset : offset + args.chunk_size]
            for offset in range(0, len(indexed_rows), args.chunk_size)
        ]
        with tqdm(
            total=len(dataset) * args.val_n,
            desc="AIME25 retool",
            unit="trajectory",
        ) as progress:
            for chunk_number, chunk in enumerate(chunks):
                chunk_results = await evaluate_chunk_retool(
                    chunk, sampling_client, tokenizer, sandbox, args, progress
                )
                results.extend(chunk_results)
                # 逐 chunk 落盘，长评测中途失败也保留已完成部分。
                write_results(chunk_results, args.output, append=chunk_number > 0)
                tqdm.write(
                    f"chunk {chunk_number + 1}/{len(chunks)}: "
                    f"correct={sum(r['num_correct'] for r in chunk_results)}"
                    f"/{sum(len(r['generations']) for r in chunk_results)} "
                    f"sandbox={sandbox.stats.metrics()}"
                )

    summary = summarize(results, args)
    with args.output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(
        f"AIME25 [{args.mode}] Average@{args.val_n}: {summary['average_at_n']:.2%} | "
        f"Pass@{args.val_n}: {summary['pass_at_n']:.2%} | "
        f"Format: {summary['format_rate']:.2%} | "
        f"CodeCalls: {summary['mean_code_calls']:.2f}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))
