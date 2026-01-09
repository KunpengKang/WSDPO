import json
import argparse
import os
import numpy as np
from tqdm import tqdm
import re

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=str, default="../result/eval_sft_sample_filtered.json",
                        help="Evaluation results with question_id, pred_defs, ex_scores")
    parser.add_argument('--sample', type=str, default="../result/outputs/COT_SFT_filtered-checkpoint-11360merge_sampling_default_semcor.json",
                        help="COT text corresponding to each pred_sql")
    parser.add_argument('--output', type=str, default="../result/wsd_dpo_filtered_llama.json",
                        help="Where to save the DPO dataset")
    return parser.parse_args()


def has_numbered_structure(text, min_steps=2):
    """ 判断 CoT 是否包含至少 min_steps 个编号步骤 """
    text = text.strip()
    pattern = r'([0-9]+)[\.\、\)]'
    steps = re.findall(pattern, text)
    unique_steps = set(steps)
    return len(unique_steps) >= min_steps


def select_pair(item, sampling_outcomes, 
                score_threshold=0.85, 
                window=0.03,
                min_score_diff=0.1,
                n=1):  # 新增参数：要生成的正负样本对数量
    """
    返回：list of tuples，每个tuple包含 (chosen_cot, rejected_cot, chosen_has_structure)
          如果不符合条件返回 None
    """
    preds = item['pred_defs']
    scores = item['ex_scores']
    
    # 基本校验：预测结果数量需要至少能选出n个正样本和n个负样本
    if len(preds) == 0 or len(scores) == 0 or len(scores) < 2*n:
        return None

    scores = np.array(scores, dtype=float)
    max_score = float(scores.max())
    min_score = float(scores.min())
    
    # 1. 丢掉整体质量太差的样本
    if max_score < score_threshold:
        return None
    
    # 检查最高分和最低分的差异是否足够大
    if max_score - min_score < min_score_diff:
        return None

    question_id = item['question_id']
    cot_list = sampling_outcomes[question_id]

    # 2. 排序：从高到低和从低到高
    sorted_indices_desc = np.argsort(-scores)  # 分数降序索引
    sorted_indices_asc = np.argsort(scores)    # 分数升序索引
    
    # 3. 选出前n个高分作为正样本候选，前n个低分作为负样本候选
    top_n_indices = sorted_indices_desc[:n]
    bottom_n_indices = sorted_indices_asc[:n]
    
    pair_results = []
    structured_count = 0
    
    # 4. 为每个高分样本匹配一个低分样本
    for i in range(n):
        chosen_idx = top_n_indices[i]
        rejected_idx = bottom_n_indices[i]
        
        # 优先找结构化的正样本（仅对第一个样本应用window筛选逻辑保持原逻辑）
        if i == 0:
            # 对第一个样本，沿用原逻辑找window内的结构化样本
            found_structured = False
            for idx in sorted_indices_desc:
                if scores[idx] < max_score - window:
                    break
                cot_text = cot_list[idx].strip()
                if has_numbered_structure(cot_text):
                    chosen_idx = idx
                    found_structured = True
                    break
            chosen_cot = cot_list[chosen_idx].strip()
            chosen_has_structure = found_structured or has_numbered_structure(chosen_cot)
        else:
            # 后续样本直接使用排序后的结果
            chosen_cot = cot_list[chosen_idx].strip()
            chosen_has_structure = has_numbered_structure(chosen_cot)
        
        rejected_cot = cot_list[rejected_idx].strip()
        
        if chosen_has_structure:
            structured_count += 1
            
        pair_results.append((chosen_cot, rejected_cot, chosen_has_structure))
    
    return pair_results

if __name__ == "__main__":
    opt = parse_option()
    np.random.seed(42)

    # load evaluation results
    eval_data = json.load(open(opt.eval, 'r', encoding='utf-8'))

    # load sampling outcomes (CoT text)
    sampling_outcomes = json.load(open(opt.sample, 'r', encoding='utf-8'))

    collection = []
    structured_count = 0  # 统计结构化正样本数量

    for item in tqdm(eval_data, total=len(eval_data)):
        # 调用修改后的select_pair函数，默认生成3对样本
        results = select_pair(item, sampling_outcomes, n=1)
        if results is None:
            continue

        # 遍历每一对样本，生成DPO实例
        for chosen_cot, rejected_cot, chosen_has_structure in results:
            if chosen_has_structure:
                structured_count += 1

            instance = {
                "question_id": f"{item['question_id']}_{len(collection)+1}",  # 为每对样本生成唯一ID
                "messages": [{"role": "user", "content": item['question']}],
                "chosen": {"role": "assistant", "content": chosen_cot},
                "rejected": {"role": "assistant", "content": rejected_cot}
            }
            collection.append(instance)

    # save DPO dataset
    os.makedirs(os.path.dirname(opt.output), exist_ok=True)
    with open(opt.output, 'w', encoding='utf-8') as f:
        json.dump(collection, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(collection)} DPO instances, saved to {opt.output}")
    print(f"其中具有结构化 CoT 的正样本数量：{structured_count}")
    print(f"占比：{structured_count / max(1, len(collection)):.2%}")
