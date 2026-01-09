

import json
import jsonlines

# 原文件是标准 JSON 数组
data = json.load(open(".../data/syn_cot_filtered.json", "r", encoding="utf-8"))

# 写成 jsonl（一行一个对象）
with jsonlines.open(".../data/syn_cot_filtered_clean.json", "w") as writer:
    for sample in data:
        clean_msg = [turn for turn in sample["messages"]
                     if turn["role"] in {"user", "assistant"}]
        writer.write({"messages": clean_msg})