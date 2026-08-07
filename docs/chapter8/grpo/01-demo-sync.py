"""PyTRIO 同步版 GRPO demo。

这个脚本实现一个最小 GRPO 训练流程：
1. 从 GSM8K 取一批数学题；
2. 用当前 LoRA 权重保存出采样客户端；
3. 每道题同步采样 group_size 个答案；
4. 用 boxed answer reward 计算 group-relative advantage；
5. 用 PyTRIO 的 importance_sampling / PPO 内置 loss 做一次优化。

最小试跑：

python docs/chapter8/grpo/01-demo-sync.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
"""

import argparse
import re
import time
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset
import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm


QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
LOSS_FNS = ("importance_sampling", "ppo")
FEWSHOT_PREFIX = [
    {"role": "user", "content": "How many r's are in strawberry?" + QUESTION_SUFFIX},
    {
        "role": "assistant",
        "content": (
            "<think>\n\n</think>\n\n"
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. "
            "There are three r's. \\boxed{3}"
        ),
    },
]


@dataclass
class GRPOConfig:
    """命令行参数解析后的训练配置。"""

    base_model: str
    lora_rank: int
    steps: int
    all_data: bool
    batch_size: int
    group_size: int
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    learning_rate: float
    beta1: float
    beta2: float
    loss_fn: str
    swanlab_mode: str
    swanlab_project: str


@dataclass
class RolloutSample:
    """一条采样结果，以及构造 importance_sampling 所需的旧策略 logprobs。"""

    tokens: list[int]
    logprobs: list[float]
    text: str
    reward: float
    advantage: float


def parse_args() -> GRPOConfig:
    """把训练配置集中到命令行参数，避免依赖环境变量。"""
    parser = argparse.ArgumentParser(description="PyTRIO 同步版 GRPO / GSM8K demo")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B", help="PyTRIO 基础模型名")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="GRPO 优化步数；每步从 GSM8K 取 batch-size 道题做 rollout",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="使用 GSM8K train split 全量数据训练一遍；打开后忽略 --steps",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="每个 step 的 GSM8K 题目数")
    parser.add_argument("--group-size", type=int, default=4, help="每道题采样的 completion 数")
    parser.add_argument("--max-tokens", type=int, default=1024, help="每次采样最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样 temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="采样 top_p")
    parser.add_argument("--seed", type=int, default=42, help="本地随机种子")
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="Adam learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument(
        "--loss-fn",
        choices=LOSS_FNS,
        default="importance_sampling",
        help="训练 loss：importance_sampling / ppo",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "disabled"),
        default="online",
        help="SwanLab 记录模式：online 上传到云端，disabled 完全关闭日志",
    )
    parser.add_argument("--swanlab-project", default="happy-llm-chapter8-grpo", help="SwanLab project")
    args = parser.parse_args()

    return GRPOConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        steps=args.steps,
        all_data=args.all_data,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        loss_fn=args.loss_fn,
        swanlab_mode=args.swanlab_mode,
        swanlab_project=args.swanlab_project,
    )


def extract_boxed(text: str) -> str | None:
    """取最后一个 \\boxed{...} 作为模型最终答案。"""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    """GSM8K 答案只做轻量归一化，避免 1,000 和 1000 被判成不同。"""
    return text.replace(",", "").strip().rstrip(".")


def grade_answer(response: str, ground_truth: str) -> float:
    """boxed answer 与标准答案完全一致时给 1，否则给 0。"""
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0


def extract_gsm8k_answer(answer_text: str) -> str:
    """GSM8K 的最终答案位于 `####` 后面。"""
    match = re.search(r"####\s*(.+)", answer_text)
    if match is None:
        raise ValueError(f"No GSM8K final answer found: {answer_text!r}")
    return normalize_answer(match.group(1))


def build_prompt(tokenizer: Any, question: str) -> list[int]:
    """把 few-shot + 当前题目渲染成模型输入 tokens。"""
    messages = [
        *FEWSHOT_PREFIX,
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("Prompt tokens are empty")
    return prompt_tokens


def load_gsm8k_train() -> Dataset:
    """加载 GSM8K 训练集；首次运行会由 datasets 下载缓存。"""
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)!r}")
    return dataset


def get_stop_sequences(tokenizer: Any) -> list[str]:
    """采样停止符尽量贴近 chat template，同时避免 None 和重复项。"""
    candidates = [tokenizer.eos_token, "<|im_end|>"]
    return list(dict.fromkeys([token for token in candidates if token]))


def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    ground_truth: str,
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> list[RolloutSample]:
    """同步采样一个 prompt 的 group_size 个 completion，并计算组内 advantage。"""
    # 对同一个 prompt 一次采样 group_size 个回答；同步 API 返回 future，所以这里立刻 `.result()` 等待。
    result = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    ).result()

    # 先把本组内每个回答的 token、旧策略 logprob、文本和 reward 收集起来。
    # advantage 需要等整组 reward 都算完后，减去组内均值才能得到。
    rewards: list[float] = []
    raw_samples: list[tuple[list[int], list[float], str]] = []

    for sequence in result.sequences:
        # return_text=True 时通常会直接返回文本；如果没有文本，就用 tokenizer 从 token 解码。
        text = sequence.text
        if text is None:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)

        # PyTRIO 采样返回 completion token 的 logprobs；
        # 后续 importance_sampling loss 需要用这组 logprobs 作为“采样时旧策略”的概率。
        tokens = list(sequence.tokens)
        logprobs = [float(value) for value in sequence.logprobs]
        if len(tokens) != len(logprobs):
            raise ValueError(
                f"Generated token/logprob length mismatch: {len(tokens)} != {len(logprobs)}"
            )

        # reward 只看模型回答里最后一个 \boxed{}，和 GSM8K 标准答案一致则为 1，否则为 0。
        reward = grade_answer(text, ground_truth)
        rewards.append(reward)
        raw_samples.append((tokens, logprobs, text))

    # GRPO 的核心是组内相对优势：同一道题里，比平均 reward 高的回答得到正 advantage。
    mean_reward = sum(rewards) / len(rewards)
    return [
        RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            text=text,
            reward=reward,
            advantage=reward - mean_reward,
        )
        for (tokens, logprobs, text), reward in zip(raw_samples, rewards, strict=True)
    ]


def build_grpo_datum(prompt_tokens: list[int], sample: RolloutSample) -> trio.Datum:
    """把单条 completion 转成 PyTRIO importance_sampling 所需的 Datum。"""
    if not sample.tokens:
        raise ValueError("Cannot train on an empty completion")

    # 自回归对齐方式如下：
    # input = prompt + completion[:-1]
    # target 前 observation_len 个位置属于 prompt 内部预测，不训练，用 0 / 0.0 占位；
    # 从最后一个 prompt token 开始预测 completion 的每个 token。
    observation_len = len(prompt_tokens) - 1
    input_tokens = prompt_tokens + sample.tokens[:-1]
    target_tokens = [0] * observation_len + sample.tokens
    padded_logprobs = [0.0] * observation_len + sample.logprobs
    padded_advantages = [0.0] * observation_len + [sample.advantage] * len(sample.tokens)

    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError("GRPO datum fields must have the same token length")

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )


def get_num_steps(dataset: Dataset, config: GRPOConfig) -> int:
    """计算实际训练 step 数；all-data 模式会覆盖命令行里的 steps。"""
    if config.all_data:
        return (len(dataset) + config.batch_size - 1) // config.batch_size
    return config.steps


def model_slug(base_model: str) -> str:
    """把模型名压成适合 experiment / weights name 的短名称。"""
    name = base_model.rsplit("/", 1)[-1].lower()
    name = name.replace("qwen3.5", "qwen35")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def build_run_name(config: GRPOConfig, effective_steps: int) -> str:
    """SwanLab experiment name 和最终权重名共用同一个规则生成值。"""
    loss_slug = config.loss_fn.replace("_", "-")
    return (
        f"grpo-sync-{model_slug(config.base_model)}-gsm8k-"
        f"{loss_slug}-steps{effective_steps}"
    )


def pick_batch(dataset: Dataset, step: int, batch_size: int, all_data: bool) -> Dataset:
    """取当前 step 的 batch；all-data 模式下不回绕，确保每条样本最多用一次。"""
    start = step * batch_size
    if all_data:
        end = min(start + batch_size, len(dataset))
        indices = list(range(start, end))
    else:
        # 非 all-data 模式保留原来的回绕逻辑，允许 steps 超过数据集可切出的完整 batch 数。
        indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    return dataset.select(indices)


def init_swanlab_run(
    config: GRPOConfig,
    effective_steps: int,
    dataset_size: int,
    run_name: str,
) -> Any | None:
    """SwanLab 只记录关键 GRPO 指标，不影响主训练逻辑。"""
    if config.swanlab_mode == "disabled":
        return None
    return swanlab.init(
        project=config.swanlab_project,
        name=run_name,
        mode=config.swanlab_mode,
        config={
            "base_model": config.base_model,
            "run_name": run_name,
            "lora_rank": config.lora_rank,
            "steps": config.steps,
            "all_data": config.all_data,
            "effective_steps": effective_steps,
            "batch_size": config.batch_size,
            "dataset_size": dataset_size,
            "group_size": config.group_size,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "learning_rate": config.learning_rate,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "loss_fn": config.loss_fn,
            "swanlab_mode": config.swanlab_mode,
            "seed": config.seed,
            "weights_name": run_name,
        },
    )


def get_numeric_metric(loss_metrics: dict[str, Any], key: str) -> float | None:
    """按当前 PyTRIO 实际返回的 metric key 精确取值，不做模糊匹配。"""
    value = loss_metrics.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def count_train_tokens(datums: list[trio.Datum]) -> int:
    """统计非零 advantage 的 token 数，也就是实际参与 RL loss 的 completion token。"""
    total = 0
    for datum in datums:
        total += sum(
            1
            for value in datum.loss_fn_inputs["advantages"].data
            if float(value) != 0.0
        )
    return total


def main(config: GRPOConfig) -> None:
    np.random.seed(config.seed)

    print("Loading GSM8K dataset...")
    train_data = load_gsm8k_train()
    print(f"Loaded {len(train_data)} GSM8K training examples")
    effective_steps = get_num_steps(train_data, config)
    run_name = build_run_name(config, effective_steps)
    print(f"Run / weights name: {run_name}")
    if config.all_data:
        print(
            f"All-data mode: {effective_steps} steps will cover "
            f"{len(train_data)} examples once"
        )

    print("Creating PyTRIO clients...")
    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()

    sampling_params = trio.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=get_stop_sequences(tokenizer),
    )
    adam_params = trio.AdamParams(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
    )
    swanlab_run = init_swanlab_run(
        config=config,
        effective_steps=effective_steps,
        dataset_size=len(train_data),
        run_name=run_name,
    )

    try:
        for step in range(effective_steps):
            batch_rows = pick_batch(
                train_data,
                step,
                config.batch_size,
                config.all_data,
            )

            # 采样必须使用当前策略，所以每个 step 先保存临时匿名 LoRA 权重并创建 sampler。
            sampling_client = training_client.save_weights_and_get_sampling_client()

            datums: list[trio.Datum] = []
            prompt_mean_rewards: list[float] = []
            rollout_lengths: list[int] = []
            n_degenerate = 0

            for row in tqdm(batch_rows, desc=f"GRPO step {step}", unit="prompt"):
                prompt_tokens = build_prompt(tokenizer, row["question"])
                ground_truth = extract_gsm8k_answer(row["answer"])
                rollout_samples = run_rollout_group(
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    ground_truth=ground_truth,
                    sampling_params=sampling_params,
                    group_size=config.group_size,
                )

                rollout_lengths.extend(len(sample.tokens) for sample in rollout_samples)
                rewards = [sample.reward for sample in rollout_samples]
                prompt_mean_reward = sum(rewards) / len(rewards)
                prompt_mean_rewards.append(prompt_mean_reward)

                # 同一题 group 内 reward 完全一样时，advantage 全为 0，没有训练信号，直接跳过。
                if all(sample.advantage == 0.0 for sample in rollout_samples):
                    n_degenerate += 1
                    continue

                for sample in rollout_samples:
                    datums.append(build_grpo_datum(prompt_tokens, sample))

            if datums:
                # 两种 loss 都由 PyTRIO 内置实现，并复用同一套 GRPO Datum。
                fwd_bwd_future = training_client.forward_backward(
                    datums,
                    loss_fn=config.loss_fn,
                )

                # 同步版 PyTRIO：提交远程前向/反向和优化器更新后，显式 `.result()` 等待完成。
                optim_future = training_client.optim_step(adam_params)
                fwd_bwd_result = fwd_bwd_future.result()
                optim_future.result()
                loss_metrics = dict(fwd_bwd_result.metrics)
            else:
                loss_metrics = {}

            loss_mean = get_numeric_metric(loss_metrics, "loss_mean")
            train_tokens = count_train_tokens(datums)
            mean_reward = sum(prompt_mean_rewards) / len(prompt_mean_rewards)
            avg_gen_len = (
                sum(rollout_lengths) / len(rollout_lengths)
                if rollout_lengths
                else 0.0
            )
            # 退化 group 指同一道题的所有回答 reward 都一样，advantage 全为 0，没有相对优劣信号。
            # 这个比例越高，说明当前 batch 里真正用于 GRPO 学习的题目越少。
            frac_degenerate = n_degenerate / len(prompt_mean_rewards)
            if swanlab_run is not None:
                log_payload = {
                    "reward": mean_reward,
                    "frac_degenerate": frac_degenerate,
                    "rollout/avg_gen_len": avg_gen_len,
                    "datums": len(datums),
                    "train_tokens": train_tokens,
                    **{
                        f"trainer/{key}": value
                        for key, value in loss_metrics.items()
                        if key == "loss_mean"
                    },
                }
                if loss_mean is not None:
                    log_payload["loss"] = loss_mean
                    log_payload["loss_mean"] = loss_mean
                swanlab.log(log_payload, step=step)

            loss_mean_text = "n/a" if loss_mean is None else f"{loss_mean:.4f}"
            print(
                f"Step {step:2d} | reward: {mean_reward:.3f} | "
                f"degenerate: {frac_degenerate:.0%} | datums: {len(datums)} | "
                f"train_tokens: {train_tokens} | "
                f"avg_gen_len: {avg_gen_len:.1f} | "
                f"loss_mean: {loss_mean_text} | loss_fn: {config.loss_fn}"
            )

        print("Saving final LoRA weights for sampler...")
        final_weights = training_client.save_weights_for_sampler(
            name=run_name
        ).result()
        print(f"Saved weights name: {run_name}, path: {final_weights.path}")
    finally:
        if swanlab_run is not None:
            swanlab_run.finish()


if __name__ == "__main__":
    cli_config = parse_args()
    start_main_time = time.time()
    main(cli_config)
    end_main_time = time.time()
    print("#" * 50)
    print("# all done")
    print(f"# train cost {end_main_time - start_main_time:.2f}s")
    print("#" * 50)
