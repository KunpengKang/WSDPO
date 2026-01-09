import argparse
import os
import json
import time
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--sampling_output', type=str)
    parser.add_argument('--sample_strategy', type=str, default='default')  # ['default', 'greedy']
    parser.add_argument('--sample_budget', type=int, default=12)

    parser.add_argument('--max_model_len', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=1024)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_option()
    opt.llm_path = os.path.abspath(opt.llm_path)

    print(f"== start sampling on {opt.llm_path} ==")

    # —— 直接读取数据集 —— 
    data = json.load(open(opt.dataset_path, "r", encoding="utf-8"))

    # 直接读取 input 字段作为 prompt
    prompts = [item["input"] for item in data]

    print(f"Loaded {len(prompts)} prompts")

    # —— 初始化 vLLM ——
    llm = LLM(
        model=opt.llm_path,
        max_model_len=opt.max_model_len,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        swap_space=64,
        enforce_eager=True,
    )

    # —— 采样策略 ——
    temperature = 0.0 if opt.sample_strategy == "greedy" else 1.0
    sample_budget = 1 if opt.sample_strategy == "greedy" else opt.sample_budget

    sampling_params = SamplingParams(
        temperature=temperature,
        n=sample_budget,
        top_k=32,
        max_tokens=opt.max_new_tokens,
        stop=['<|EOT|>', '<|eot_id|>'],
    )

    # —— 生成 —— 
    start = time.time()
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    all_pred = []
    for output in tqdm(outputs):
            # 一个 prompt → 多个 sample
            all_pred.append([o.text for o in output.outputs])

    # —— 保存 —— 
    with open(opt.sampling_output, "w", encoding="utf-8") as f:
        json.dump(all_pred, f, indent=2, ensure_ascii=False)

    end = time.time()
    print(f"Done. Avg time per example: {(end-start)/len(prompts):.4f}s")
