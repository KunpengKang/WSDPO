import json
import argparse
from distutils.util import strtobool
from utils.evaluate_metric import evaluate
import re


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str)
    parser.add_argument('--gold', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--is_cot', type=str, default="True", help="Whether the predictions are from a CoT model")
    parser.add_argument('--threshold', type=float, default=0.7, help="Threshold to label positive samples")
    return parser.parse_args()


def extract_prediction_from_cot(text, is_cot=True):
    """
    从 CoT 中提取 Definition。
    """
    if not is_cot:
        return text

    # 优先匹配 ```Definition  ... ```
    matches = re.findall(r"```(?i:definition)\b\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 回退：单反引号
    matches = re.findall(r"`(.*?)`", text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return text


def main():
    opt = parse_option()
    is_cot = bool(strtobool(opt.is_cot))

    # ---------------------------
    # Load predictions
    # ---------------------------
    raw_predictions = json.load(open(opt.pred, "r", encoding="utf-8"))

    # 提取纯定义
    all_grouped_pred_definitions = [
        [extract_prediction_from_cot(txt, is_cot) for txt in group]
        for group in raw_predictions
    ]

    # ---------------------------
    # Load gold
    # ---------------------------
    data = json.load(open(opt.gold, "r", encoding="utf-8"))
    data = data[:25000]

    questions = [
        f"Please determine the correct definition of the target word in the context.\nContext: {item['context']}\nTarget word: {item['target']}"
        for item in data
    ]

    gold_definitions = [
        "\n".join(item.get("correct_definitions_in_context", []))
        for item in data
    ]
    print(len(all_grouped_pred_definitions))
    assert len(all_grouped_pred_definitions) == len(gold_definitions)

    print(f"[INFO] Loaded {len(questions)} samples")
    print(f"[INFO] Total predictions: {sum(len(g) for g in all_grouped_pred_definitions)}")

    # ---------------------------
    # ⭐ 扁平化所有预测，统一 evaluate
    # ---------------------------
    flat_pred = []
    flat_gold = []
    lengths = []

    for grouped, gold in zip(all_grouped_pred_definitions, gold_definitions):
        lengths.append(len(grouped))
        flat_pred.extend(grouped)
        flat_gold.extend([gold] * len(grouped))

    print(f"[INFO] Running evaluate on {len(flat_pred)} predictions in ONE batch...")

    flat_scores = evaluate(flat_pred, flat_gold)  # 一次性算出所有 scores

    # ---------------------------
    # 把 flat scores 切分回每个问题
    # ---------------------------
    index = 0
    split_scores = []
    for L in lengths:
        split_scores.append(flat_scores[index:index + L])
        index += L

    # ---------------------------
    # 组装输出 JSON
    # ---------------------------
    evaluation_results = []

    for qid, (grouped, scores, question, gold) in enumerate(
            zip(all_grouped_pred_definitions, split_scores, questions, gold_definitions)
    ):
        evaluation_results.append({
            "question_id": qid,
            "question": question,
            "ground_truth": gold,
            "pred_defs": grouped,
            "ex_scores": scores
        })

    # ---------------------------
    # Save
    # ---------------------------
    with open(opt.output, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved evaluation to {opt.output}")


if __name__ == "__main__":
    main()
