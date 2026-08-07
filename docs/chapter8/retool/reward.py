"""计算 ReTool 的结果奖励：抽取最后一个 \\boxed{} 并与参考答案做数学等价判定。

对齐官方 recipe 的 math_dapo.compute_score(strict_box_verify=True)：
只取回答末尾 300 个字符、提取最后一个 \\boxed{}，对 +1 / 错 −1（格式非法也算错）。
等价判定使用 math_verify（verl 生态实际使用的库）。
"""

from dataclasses import dataclass

from math_verify import parse, verify


ANSWER_WINDOW_CHARS = 300  # 官方实现只检查回答末尾 300 个字符。


@dataclass(frozen=True)
class RewardResult:
    """保存最终 reward 及其判定细节。"""

    reward: float  # +1.0（答案正确）或 -1.0（错误、含格式非法）。
    correct: bool  # 答案是否通过数学等价判定。
    valid_format: bool  # 是否找到合法的 \boxed{} 答案。
    answer: str | None  # 抽取到的答案原文，用于记录和调试。


def extract_last_boxed(text: str) -> str | None:
    """提取最后一个 \\boxed{...} 的内容，按花括号配平。"""
    marker = "\\boxed{"
    index = text.rfind(marker)
    if index < 0:
        return None
    start = index + len(marker)
    depth = 1
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position]
    return None


def answers_equivalent(prediction: str, reference: str) -> bool:
    """用 math_verify 判定预测答案是否与参考答案数学等价。"""
    try:
        return bool(verify(parse(f"${reference}$"), parse(f"${prediction}$")))
    except Exception:  # math_verify 对怪异输入可能抛异常，一律按不等价处理
        return False


def score_answer(text: str, reference: str) -> RewardResult:
    """按官方 ±1 规则给整条轨迹的最终文本打分。"""
    answer = extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])
    if answer is None:
        return RewardResult(-1.0, False, False, None)
    correct = answers_equivalent(answer.strip(), reference.strip())
    return RewardResult(1.0 if correct else -1.0, correct, True, answer)
