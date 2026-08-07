"""异步版 OPD demo：ModelScope DeepMath-103K + PyTRIO + SwanLab。

核心逻辑是 on-policy distillation：
student 先采样，teacher 对 student 采样轨迹算 logprob，
reverse_kl = student_logprob - teacher_logprob，再用 -reverse_kl 做 advantage。

小成本试跑：
python docs/chapter8/opd/02-demo-async.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 4 \
    --max-tokens 512 \
    --sample-size 100 \
    --swanlab-mode disabled
"""

from __future__ import annotations

import argparse
import asyncio
import random
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm


trio.configure(
    timeout=600,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets" / "DeepMath-103K"
DEEPMATH_SHARDS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTRIO 异步版 OPD / DeepMath")
    parser.add_argument("--dataset-repo", default="AI-ModelScope/DeepMath-103K")
    parser.add_argument("--dataset-revision", default="master")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--num-shards", type=int, default=1, help="完整 DeepMath 为 10 个分片")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--sample-size", type=int, default=1000, help="随机抽样数量；<=0 表示全用")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--teacher-base-model", default="Qwen/Qwen3.6-27B", help="默认使用 27B teacher")
    parser.add_argument("--teacher-model-path", default=None)

    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--question-suffix",
        default="Please solve the problem step by step and put the final answer in \\boxed{}.",
    )
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--sampler-refresh-steps", type=int, default=1)
    parser.add_argument("--save-weights-name", default="opd-deepmath-qwen35-4b-async")

    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swanlab-project", default="happy-llm-chapter8-opd")
    parser.add_argument("--swanlab-name", default="opd-deepmath-qwen35-4b-async")
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
    )
    args = parser.parse_args()

    # DeepMath-103K 在 ModelScope 上是 10 个 parquet 分片，这里允许只下载前 N 个分片试跑。
    if not 1 <= args.num_shards <= DEEPMATH_SHARDS:
        raise ValueError(f"--num-shards must be between 1 and {DEEPMATH_SHARDS}")

    # 这些参数都会直接参与训练循环或取模逻辑，不能为 0 或负数。
    for name in ("steps", "batch_size", "group_size", "sampler_refresh_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    return args


def shard_name(index: int) -> str:
    # ModelScope 上的 DeepMath-103K 使用 HuggingFace parquet 分片命名格式。
    return f"data/train-{index:05d}-of-{DEEPMATH_SHARDS:05d}.parquet"


def modelscope_file_url(repo: str, revision: str, file_path: str) -> str:
    # repo 形如 "AI-ModelScope/DeepMath-103K"，拆成 namespace 和数据集名。
    namespace, name = repo.split("/", 1)

    # ModelScope 的 repo 文件接口要求把文件路径放在 FilePath 参数里。
    query = urllib.parse.urlencode(
        {
            "Source": "SDK",
            "Revision": revision,
            "FilePath": file_path,
            "View": "False",
        }
    )
    return f"https://www.modelscope.cn/api/v1/datasets/{namespace}/{name}/repo?{query}"


def download_if_needed(url: str, local_path: Path, force: bool) -> None:
    # 已经下载过且没有要求强制重下时，直接复用本地分片。
    if local_path.exists() and not force:
        return

    # 先写入 .tmp，下载完成后再替换，避免中断时留下半截 parquet。
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    print(f"Downloading {url}\n  -> {local_path}")

    # urllib 足够处理这个单文件下载场景，避免额外依赖 requests。
    request = urllib.request.Request(url, headers={"User-Agent": "trio-case-opd-demo"})
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    tmp_path.replace(local_path)


def load_deepmath(args: argparse.Namespace) -> Dataset:
    """从 ModelScope 下载 parquet 到本地，然后用 datasets 库读取本地文件。"""
    shard_paths = []

    # 只下载前 num_shards 个分片，方便先用小数据量做联调。
    for index in range(args.num_shards):
        remote_path = shard_name(index)
        local_path = args.dataset_dir / remote_path

        # 数据真正来自 ModelScope；后面的 load_dataset 只是读取这些本地 parquet。
        download_if_needed(
            modelscope_file_url(args.dataset_repo, args.dataset_revision, remote_path),
            local_path,
            args.force_download,
        )
        shard_paths.append(str(local_path))

    # 使用 HuggingFace datasets 的 parquet reader 读本地文件，不会从 HF Hub 拉数据。
    dataset = load_dataset(
        "parquet",
        data_files=shard_paths,
        split="train",
        cache_dir=str(args.dataset_dir / ".datasets_cache"),
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)!r}")

    # OPD 这里只需要 prompt；DeepMath 里对应字段是 question。
    if "question" not in dataset.column_names:
        raise ValueError(f"DeepMath must contain 'question', got {dataset.column_names}")

    # 先固定 seed 打乱，再截取 sample_size，保证每次试跑可复现。
    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_size > 0:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    return dataset


def build_prompt(tokenizer: Any, question: str, suffix: str, enable_thinking: bool) -> list[int]:
    """把 DeepMath question 渲染成 chat prompt token。"""
    content = question.strip() if not suffix else f"{question.strip()}\n\n{suffix}"
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokens are empty")
    return prompt_ids


async def completion_teacher_logprobs_async(
    teacher_client: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> list[float]:
    """teacher 对 student 实际生成 completion 的逐 token logprob。"""
    # teacher 需要看到和 student rollout 完全一致的上下文：
    # prompt 是题目，completion 是 student 已经生成出来的答案 token。
    all_ids = prompt_ids + completion_ids

    # compute_logprobs_async 在当前 PyTRIO SDK 里 await 后直接返回 list[float | None]。
    # 第一个 token 通常没有前文，所以返回值里可能有 None；我们只取 completion 区间。
    all_logprobs = await teacher_client.compute_logprobs_async(
        trio.ModelInput.from_ints(all_ids)
    )

    # completion 的第一个 token 是在完整 prompt 后被预测出来的，
    # 因此从 len(prompt_ids) 开始到结尾就是 completion 的 logprobs。
    completion_logprobs = all_logprobs[len(prompt_ids) :]

    # completion 区间必须和 student 生成 token 一一对应，不能有 None，
    # 否则 reverse KL 的 token 对齐就不可靠。
    if len(completion_logprobs) != len(completion_ids) or any(v is None for v in completion_logprobs):
        raise ValueError("Invalid teacher logprobs for completion tokens")
    return [float(v) for v in completion_logprobs]


def build_opd_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs: list[float],
    advantages: list[float] | np.ndarray,
) -> trio.Datum:
    """PyTRIO importance_sampling 需要右移后的 input/target/logprobs/advantages。"""
    # 自回归训练需要右移：用当前位置 input token 预测下一个 target token。
    # prompt 内部的预测不是 OPD 训练目标，所以这部分 advantage 置 0。
    prompt_loss_len = len(prompt_ids) - 1

    # input 由完整 prompt 加 completion[:-1] 组成：
    # 最后一个 input 位置用来预测 completion 的最后一个 token。
    input_ids = prompt_ids + completion_ids[:-1]

    # target 的 prompt 区间用 0 占位；从最后一个 prompt token 位置开始预测 completion。
    target_ids = [0] * prompt_loss_len + completion_ids

    # old_logprobs 是 student rollout 时每个 completion token 的旧策略 logprob。
    # prompt 区间不训练，因此同样用 0.0 占位。
    padded_logprobs = [0.0] * prompt_loss_len + old_logprobs

    # advantages 是 -kl_penalty_coef * reverse_kl，只对 completion token 生效。
    padded_advantages = [0.0] * prompt_loss_len + list(advantages)

    # PyTRIO 的 importance_sampling 要求四个序列长度严格一致。
    if not (len(input_ids) == len(target_ids) == len(padded_logprobs) == len(padded_advantages)):
        raise ValueError("OPD datum fields must have the same length")

    # loss_fn_inputs 的三个字段对应 importance_sampling schema：
    # target_tokens 是要预测的 token，logprobs 是采样时旧策略概率，advantages 是训练信号。
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_ids, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )


async def run_prompt_rollout(
    student_sampler: Any,
    teacher_client: Any,
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    sampling_params: trio.SamplingParams,
) -> tuple[list[trio.Datum], list[float], list[int]]:
    """异步处理单道题：student 采样，再让 teacher 给同一批 completion 打 logprob。"""
    # DeepMath 这里只使用 question 字段构造 prompt。
    prompt_ids = build_prompt(
        tokenizer,
        row["question"],
        args.question_suffix,
        args.enable_thinking,
    )

    # 当前 student 采样出的 completion 就是 OPD 的 on-policy 轨迹。
    result = await student_sampler.sample_async(
        prompt=trio.ModelInput.from_ints(prompt_ids),
        num_samples=args.group_size,
        sampling_params=sampling_params,
        return_text=False,
    )

    sequences = [seq for seq in result.sequences if seq.tokens]

    # teacher 不是重新生成答案，而是并发给 student 已生成 token 打 logprob。
    teacher_tasks = [
        completion_teacher_logprobs_async(teacher_client, prompt_ids, seq.tokens)
        for seq in sequences
    ]
    teacher_logprobs_list = await asyncio.gather(*teacher_tasks) if teacher_tasks else []

    # datums 会送进 PyTRIO forward_backward；
    # reverse_kls 和 completion_token_counts 只用于本 step 的日志统计。
    datums: list[trio.Datum] = []
    reverse_kls: list[float] = []
    completion_token_counts: list[int] = []

    for seq, teacher_lps in zip(sequences, teacher_logprobs_list, strict=True):
        completion_ids = seq.tokens
        student_lps = [float(value) for value in seq.logprobs]

        if len(student_lps) != len(completion_ids):
            raise ValueError(
                f"Student token/logprob length mismatch: {len(completion_ids)} != {len(student_lps)}"
            )

        # OPD 信号：reverse_kl = student_logprob - teacher_logprob。
        reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
        advantages = -args.kl_penalty_coef * reverse_kl

        # importance_sampling 使用旧策略 logprobs 和 KL advantage 更新 student。
        datums.append(build_opd_datum(prompt_ids, completion_ids, student_lps, advantages))
        reverse_kls.extend(reverse_kl.tolist())
        completion_token_counts.append(len(completion_ids))

    return datums, reverse_kls, completion_token_counts


def start_swanlab(args: argparse.Namespace, dataset_size: int) -> Any | None:
    """SwanLab 只做实验记录，不影响训练逻辑。"""
    if not args.swanlab:
        return None
    config = vars(args).copy()
    config["dataset_dir"] = str(args.dataset_dir)
    config["dataset_size"] = dataset_size
    return swanlab.init(
        project=args.swanlab_project,
        name=args.swanlab_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["TRIO", "OPD", "DeepMath", "ModelScope", "async"],
        log_dir=str(SCRIPT_DIR / "swanlog"),
    )


async def main(args: argparse.Namespace) -> None:
    # 固定本地随机性：数据 shuffle 和部分采样参数都会用到 seed。
    random.seed(args.seed)
    np.random.seed(args.seed)

    # DeepMath 在这里作为 prompt pool 使用，只取 question 字段。
    dataset = load_deepmath(args)
    print(f"Loaded {len(dataset)} DeepMath prompts")

    service_client = trio.ServiceClient()
    swanlab_run = None

    try:
        print("Creating PyTRIO clients...")
        # student 是一个 LoRA training client，后续 forward/backward 和 optim 都作用在它上面。
        training_client = await service_client.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )
        tokenizer = training_client.get_tokenizer()

        # teacher 只负责给 student 采样轨迹打 logprob，不参与优化。
        teacher_client = await service_client.create_sampling_client_async(
            base_model=args.teacher_base_model or args.base_model,
            model_path=args.teacher_model_path,
        )

        # student rollout 的采样参数；这些回答会成为 OPD 的 on-policy 轨迹。
        sampling_params = trio.SamplingParams(
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop=[x for x in [tokenizer.eos_token, "<|im_end|>"] if x],
        )

        # PyTRIO 的优化器参数；每步 forward_backward 后调用一次 optim_step。
        adam_params = trio.AdamParams(
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
        )

        swanlab_run = start_swanlab(args, len(dataset))
        student_sampler = None

        for step in range(args.steps):
            step_start = time.time()
            if student_sampler is None or step % args.sampler_refresh_steps == 0:
                # 严格 on-policy：用当前 student 权重采样。
                student_sampler = await training_client.save_weights_and_get_sampling_client_async()

            # dataset 已经提前随机打乱；这里按 step 循环取 batch，超出后回绕。
            indices = [(step * args.batch_size + i) % len(dataset) for i in range(args.batch_size)]
            batch_rows = dataset.select(indices)

            # batch 内每道题的 rollout 彼此独立，可以并发请求远端 sampler。
            rollout_tasks = [
                run_prompt_rollout(
                    student_sampler=student_sampler,
                    teacher_client=teacher_client,
                    tokenizer=tokenizer,
                    row=row,
                    args=args,
                    sampling_params=sampling_params,
                )
                for row in batch_rows
            ]
            with tqdm(total=len(rollout_tasks), desc=f"OPD async step {step}", unit="prompt") as progress_bar:

                async def run_and_track(rollout_task: Any) -> tuple[list[trio.Datum], list[float], list[int]]:
                    result = await rollout_task
                    progress_bar.update(1)
                    return result

                rollout_results = await asyncio.gather(
                    *(run_and_track(rollout_task) for rollout_task in rollout_tasks)
                )

            # 一个 step 内会把 batch_size * group_size 条 completion 转成训练 datum。
            datums: list[trio.Datum] = []
            reverse_kls: list[float] = []
            completion_token_counts: list[int] = []
            for rollout_datums, rollout_reverse_kls, rollout_completion_counts in rollout_results:
                datums.extend(rollout_datums)
                reverse_kls.extend(rollout_reverse_kls)
                completion_token_counts.extend(rollout_completion_counts)

            if not datums:
                raise RuntimeError("No OPD datums were built")

            # 异步版 PyTRIO：先异步提交远程前向/反向和优化器更新，再 await 对应 future。
            fwd_bwd_future = await training_client.forward_backward_async(
                datums,
                loss_fn="importance_sampling",
            )
            optim_future = await training_client.optim_step_async(adam_params)
            fwd_bwd_result = await fwd_bwd_future
            await optim_future

            step_elapsed_time = time.time() - step_start
            completion_tokens_total = int(sum(completion_token_counts))

            # 记录最关键的 OPD 和训练指标，方便在 SwanLab 上看趋势。
            metrics = {
                "data/datums": len(datums),
                "data/completion_tokens_mean": float(np.mean(completion_token_counts)),
                "data/completion_tokens_total": completion_tokens_total,
                "data/completion_tokens_per_second": completion_tokens_total / step_elapsed_time,
                "opd/reverse_kl_mean": float(np.mean(reverse_kls)),
                "opd/reverse_kl_std": float(np.std(reverse_kls)),
                "train/learning_rate": args.learning_rate,
                "train/step": step,
                "time/step_elapsed_time": step_elapsed_time,
            }
            metrics.update({f"trainer/{k}": float(v) for k, v in dict(fwd_bwd_result.metrics).items()})
            if swanlab_run is not None:
                swanlab.log(metrics, step=step)
            tqdm.write(
                f"step {step:03d}/{args.steps} | datums {len(datums)} | "
                f"completion tokens mean {metrics['data/completion_tokens_mean']:.1f} | "
                f"tokens/s {metrics['data/completion_tokens_per_second']:.1f} | "
                f"reverse_kl {metrics['opd/reverse_kl_mean']:.4f} | "
                f"time {metrics['time/step_elapsed_time']:.2f}s"
            )

        # 保存最终 LoRA 权重，后续可以用这个 path 创建 sampler 做推理。
        save_future = await training_client.save_weights_for_sampler_async(
            name=args.save_weights_name
        )
        save_result = await save_future
        print(f"Saved weights: {save_result.path}")
        if swanlab_run is not None:
            swanlab.log({"save/weights_path": swanlab.Text(save_result.path)}, step=args.steps)
    finally:
        # 无论中间是否报错，都尽量正常结束日志。
        if swanlab_run is not None:
            swanlab.finish()


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main(parse_args()))
    print("#" * 50)
    print("# all done")
    print(f"# train cost {time.time() - start:.2f}s")
    print("#" * 50)
