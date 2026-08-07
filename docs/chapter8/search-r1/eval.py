"""用同一套搜索环境评测 Base Model 或 Search-R1 checkpoint。

从 Happy-LLM 仓库根目录评测 Base Model：
python docs/chapter8/search-r1/eval.py \
    --batch-size 16 \
    --output docs/chapter8/search-r1/eval_result/eval_results_base.jsonl

评测训练后的 checkpoint：
python docs/chapter8/search-r1/eval.py \
    --batch-size 16 \
    --model-path 'trio://runxxxxxxxxxx' \
    --output docs/chapter8/search-r1/eval_result/eval_results_rl_step_20.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytrio as trio
from tqdm import tqdm

from data import SearchExample, load_examples
from rollout import RolloutConfig, Trajectory, rollout_batch
from search import (
    SEARCH_BACKENDS,
    SearchClient,
    create_search_client,
    resolve_search_concurrency,
    resolve_search_timeout,
)


EVAL_DATA_PATH = Path(__file__).resolve().parent / "datasets" / "dev.jsonl"
EVAL_RESULT_DIR = Path(__file__).resolve().parent / "eval_result"
EXPECTED_EVAL_SIZE = 70


def mean(values: list[float]) -> float:
    """计算列表均值，空列表返回零。"""
    return sum(values) / len(values) if values else 0.0


def chunks(examples: list[SearchExample], size: int) -> list[list[SearchExample]]:
    """把评测问题切成适合并发采样的小批次。"""
    return [examples[index : index + size] for index in range(0, len(examples), size)]


def trajectory_record(trajectory: Trajectory) -> dict[str, Any]:
    """把一条评测轨迹转成可复查的 JSON 记录。"""
    return {
        "type": "trajectory",
        "id": trajectory.example.id,
        "question": trajectory.example.question,
        "answers": trajectory.example.answers,
        "data_source": trajectory.example.data_source,
        "final_text": trajectory.final_text,
        "reward": trajectory.reward,
        "exact_match": trajectory.exact_match,
        "valid_format": trajectory.valid_format,
        "search_calls": trajectory.search_calls,
        "assistant_turns": len(trajectory.turns),
    }


def evaluation_metrics(
    trajectories: list[Trajectory], search_client: SearchClient
) -> dict[str, float]:
    """计算各数据源 EM、宏平均和搜索辅助指标。"""
    by_source: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        by_source[trajectory.example.data_source].append(trajectory)
    metrics: dict[str, float] = {}
    source_scores: list[float] = []
    for source, items in sorted(by_source.items()):
        score = mean([float(item.exact_match) for item in items])
        metrics[f"em/{source}"] = score
        source_scores.append(score)
    metrics.update(
        {
            "em/macro": mean(source_scores),
            "format/rate": mean([float(item.valid_format) for item in trajectories]),
            "rollout/search_calls": mean(
                [float(item.search_calls) for item in trajectories]
            ),
            "rollout/no_search_rate": mean(
                [float(item.search_calls == 0) for item in trajectories]
            ),
            "rollout/turns": mean([float(len(item.turns)) for item in trajectories]),
        }
    )
    metrics.update(search_client.stats.metrics())
    return metrics


def summary_record(
    args: argparse.Namespace,
    trajectories: list[Trajectory],
    metrics: dict[str, float],
) -> dict[str, Any]:
    """把本次评测配置和最终汇总指标转成 JSONL 的末行记录。"""
    return {
        "type": "summary",
        "base_model": args.base_model,
        "model_path": args.model_path,
        "search_backend": args.search_backend,
        "search_concurrency": args.search_concurrency,
        "search_timeout": args.search_timeout,
        "search_model": (
            args.search_model if args.search_backend == "deepseek" else None
        ),
        "evaluated_examples": len(trajectories),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    """解析评测模型、输出和 rollout 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多评测多少道题；0 表示评测固定评测集中的全部 70 道题",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="每批同时推进的评测问题数；模型采样和不同轨迹的搜索均可并发",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="创建 PyTRIO sampling client 使用的基础模型",
    )
    parser.add_argument(
        "--model-path",
        help="PyTRIO sampler weights 路径；留空即评测未训练的 base model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_RESULT_DIR / "eval_results.jsonl",
        help="逐题评测结果的 JSONL 输出路径；已有文件会被覆盖",
    )
    parser.add_argument(
        "--max-search-calls",
        type=int,
        default=4,
        help="每条评测轨迹最多调用搜索工具的次数",
    )
    parser.add_argument(
        "--search-backend",
        choices=SEARCH_BACKENDS,
        default="deepseek",
        help="搜索后端：deepseek 付费高并发，wikipedia 免费免密钥，zhihu 免费但有额度限制",
    )
    parser.add_argument(
        "--search-concurrency",
        type=int,
        default=None,
        help="不同轨迹之间的搜索并发；默认 deepseek=16、wikipedia=3、zhihu=1",
    )
    parser.add_argument(
        "--search-model",
        default="deepseek-v4-flash",
        help="DeepSeek Search 使用的模型；wikipedia 和 zhihu 后端会忽略此参数",
    )
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=None,
        help="单次搜索的超时秒数；默认 deepseek=60、zhihu=15",
    )
    parser.add_argument(
        "--max-assistant-turns",
        type=int,
        default=6,
        help="每条评测轨迹最多生成的 assistant 回合数",
    )
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=8192,
        help="整条评测轨迹允许使用的最大 token 数",
    )
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=1024,
        help="单个 assistant 回合最多生成的 token 数",
    )
    parser.add_argument(
        "--max-tool-response-tokens",
        type=int,
        default=1024,
        help="单次搜索结果最多保留的 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="评测采样温度；默认 0.0，尽量使用确定性生成",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="评测核采样的累积概率阈值",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="评测 rollout 使用的随机种子",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """在固定 70 题上运行 base 和 checkpoint 共用的 evaluator。"""
    args.search_concurrency = resolve_search_concurrency(
        args.search_backend, args.search_concurrency
    )
    args.search_timeout = resolve_search_timeout(
        args.search_backend, args.search_timeout
    )

    # 先校验磁盘上的固定评测集仍是完整 70 题，再按 limit 截取少量题做试跑。
    examples = load_examples(EVAL_DATA_PATH)
    if len(examples) != EXPECTED_EVAL_SIZE:
        raise ValueError(
            f"固定评测集应包含 {EXPECTED_EVAL_SIZE} 条，实际为 {len(examples)} 条；"
            "请重新运行 prepare_data.py"
        )
    if args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("评测数据为空，请先运行 prepare_data.py")

    # model_path 留空时评测 base model；传入 sampler weights 时评测训练后模型。
    service_client = trio.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=args.base_model,
        model_path=args.model_path,
    )

    # 评测沿用训练时相同的 tokenizer、搜索客户端和多轮轨迹限制。
    tokenizer = sampling_client.get_tokenizer()
    search_client = create_search_client(
        args.search_backend,
        Path(__file__).resolve().parent / ".env",
        model=args.search_model,
        timeout=args.search_timeout,
    )
    config = RolloutConfig(
        # 每道题只生成一条确定的评测轨迹，不需要训练阶段的 GRPO group。
        group_size=1,
        max_search_calls=args.max_search_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        search_concurrency=args.search_concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # 自动创建 eval_result 等父目录；每次运行都会覆盖指定的 JSONL 文件。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectories: list[Trajectory] = []
    batches = chunks(examples, args.batch_size)
    with args.output.open("w", encoding="utf-8") as file:
        # 外层显示整个评测集进度，内层显示当前 batch 内已经结束的轨迹数。
        with tqdm(
            total=len(examples),
            desc="Total eval",
            unit="question",
            position=0,
        ) as total_progress:
            for batch_index, batch in enumerate(batches, 1):
                with tqdm(
                    total=len(batch),
                    desc=f"Batch {batch_index}/{len(batches)}",
                    unit="question",
                    position=1,
                    leave=False,
                ) as batch_progress:

                    def update_progress(completed: int) -> None:
                        """同时推进当前 batch 和整个评测集的进度条。"""
                        batch_progress.update(completed)
                        total_progress.update(completed)

                    batch_trajectories = rollout_batch(
                        sampling_client,
                        tokenizer,
                        search_client,
                        batch,
                        config,
                        progress_callback=update_progress,
                    )
                trajectories.extend(batch_trajectories)

                # 每完成一批就立即落盘逐题结果，便于中途检查已完成的轨迹。
                for trajectory in batch_trajectories:
                    file.write(
                        json.dumps(trajectory_record(trajectory), ensure_ascii=False)
                        + "\n"
                    )
                total_progress.set_postfix(completed=len(trajectories))

        # 最后一行保存本次评测的模型信息和全部汇总指标，便于单文件复查。
        metrics = evaluation_metrics(trajectories, search_client)
        file.write(
            json.dumps(
                summary_record(args, trajectories, metrics),
                ensure_ascii=False,
            )
            + "\n"
        )

    # 终端也打印同一份汇总指标，方便运行结束后立即查看 EM 等结果。
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved predictions: {args.output}")


if __name__ == "__main__":
    main(parse_args())
