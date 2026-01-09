#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter a JSONL/JSON file by definition-extraction + scoring.

1. 从 assistant 的 content 里用正则提取 ```Definition\n...``` 的内容
2. 把提取到的定义列表（pre）与 correct_definitions_in_context 的内容列表（real）一起喂给
   evaluate(pres, reals) -> List[float]
3. 只保留分数 > 0.8 的完整样本，写出到新文件
"""

import json
import re
from evaluate_metric import evaluate
import numpy as np

def extract_definition(content: str) -> str:
    """从assistant的content中提取```Definition```中的内容"""
    pattern = r'```Definition\s*(.*?)\s*```'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def process_json_file(input_path: str, output_path: str, threshold: float = 0.8):
    """
    处理JSON文件，筛选相似度大于threshold的条目
    
    参数:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        threshold: 相似度阈值
    """
    # 读取输入JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pre_list = []
    real_list = []
    filtered_data = []
    
    # 遍历所有JSON对象
    for item in data:
        if "messages" not in item:
            continue
            
        messages = item["messages"]
        pre_definition = ""
        real_definition = ""
        
        # 从messages中提取assistant和correct_definitions_in_context的内容
        for msg in messages:
            if msg["role"] == "assistant":
                pre_definition = extract_definition(msg["content"])
            elif msg["role"] == "correct_definitions_in_context":
                real_definition = msg["content"].strip()
        
        # 如果两个定义都存在，则添加到列表
        if pre_definition and real_definition:
            pre_list.append(pre_definition)
            real_list.append(real_definition)
            filtered_data.append(item)
    
    # 计算相似度分数
    scores = evaluate(pre_list, real_list)
    
    # 筛选分数大于0.8的条目
    final_data = []
    for i, (score, item) in enumerate(zip(scores, filtered_data)):
        if score > threshold:
            # 可选：将分数添加到数据中以便调试
            # item["similarity_score"] = score
            final_data.append(item)
    
    # 输出统计信息
    print(f"原始数据条目数: {len(data)}")
    print(f"包含有效定义的条目数: {len(filtered_data)}")
    print(f"相似度大于{threshold}的条目数: {len(final_data)}")
    print(f"平均相似度: {np.mean(scores):.3f}")
    print(f"相似度分布:")
    for thresh in [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]:
        count = sum(1 for s in scores if s > thresh)
        print(f"  >{thresh:.1f}: {count}条 ({count/len(scores)*100:.1f}%)")
    
    # 保存到新的JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存到: {output_path}")
    
    # 返回统计信息
    return {
        "total_original": len(data),
        "total_with_definitions": len(filtered_data),
        "total_filtered": len(final_data),
        "avg_score": float(np.mean(scores)),
        "scores": [float(s) for s in scores]  # 可选：返回所有分数
    }

def main():
    # 输入输出文件路径
    input_path = ".../data/syn_cot.json"
    
    # 构建输出文件路径
    import os
    dir_name = os.path.dirname(input_path)
    file_name = os.path.basename(input_path)
    name_without_ext = os.path.splitext(file_name)[0]
    output_path = os.path.join(dir_name, f"{name_without_ext}_filtered.json")
    
    # 处理文件
    stats = process_json_file(input_path, output_path, threshold=0.8)
    
    # 显示一些示例
    print("\n前5个条目的示例:")
    
    # 读取输出文件显示示例
    with open(output_path, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)
    
    for i in range(min(2, len(filtered_data))):
        print(f"\n示例 {i+1}:")
        item = filtered_data[i]
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                print(f"  Assistant定义: {extract_definition(msg['content'])[:100]}...")
            elif msg["role"] == "correct_definitions_in_context":
                print(f"  真实定义: {msg['content'][:100]}...")

if __name__ == "__main__":
    main()