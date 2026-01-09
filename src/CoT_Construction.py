import json
import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI   

# -------------------- 参数解析 --------------------
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--sample_budget', type=int, default=6)
    return parser.parse_args()

args = parse_option()

# -------------------- 全局变量 --------------------
model = 'Qwen/Qwen2.5-7B-Instruct'
system_message = open('../data/system.txt', 'r').read()
trainset = json.loads(open('../data/semcor.json').read()) #处理后的semcor路径

# 实时保存的路径（逐条写入 jsonl）
incremental_save_path = '../data/semcor_raw.jsonl' 

# -------------------- 请求函数 --------------------
def def_generate_item(item, model=model, n=6, t=1):
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=args.api_key,
    )

    prompt = (
        f"Context: {item['context']}\n"
        f"Target word: {item['target']}\n"
        f"Reference Definition: {item['correct_definitions_in_context'][0]}\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        n=n,
        temperature=t
    )
    # print(response)
    output = {
        'context': item['context'],
        'target': item['target'],
        'correct_definitions_in_context': item['correct_definitions_in_context'][0],  # 取第一个
        'generated': [choice.message.content for choice in response.choices]
    }
    return output

# -------------------- 并发请求 + 实时保存 --------------------
collection = []
failed_task_id = []
executor = ThreadPoolExecutor(max_workers=3)
all_task = {
    executor.submit(def_generate_item, trainset[i], model, args.sample_budget, 1.0): i
    for i in range(len(trainset))
}

# 打开 jsonl 文件，逐条 append
f_inc = open(incremental_save_path, 'a', encoding='utf-8')

for future in tqdm.tqdm(as_completed(all_task)):
    task_id = all_task[future]
    try:
        res = future.result(timeout=300)
        res['question_id'] = task_id
        print('====================')
        print('task_id:', task_id)
        print('count:', len(res['generated']))

        # 1) 放到内存里，后面还要做后处理
        collection.append(res)

        # 2) 实时写入到 jsonl，一行一个样本
        f_inc.write(json.dumps(res, ensure_ascii=False) + '\n')
        f_inc.flush()

    except Exception as e:
        print('failed')
        print('task_id:', task_id)
        print('error:', repr(e))
        failed_task_id.append(task_id)

executor.shutdown()
f_inc.close()  # 记得关文件
print('failed: ', failed_task_id)

# -------------------- 后处理 --------------------
tailored = []
for col in tqdm.tqdm(collection):
    gened = col['generated'].copy()
    d = col.copy()
    del d['generated']
    for sql in gened:
        d['output'] = sql
        tailored.append(d.copy())
tailored.sort(key=lambda x: x['question_id'])

# -------------------- 转对话格式 --------------------
collections = []
for i in range(len(tailored)):
    col = {'messages': []}
    user = f"Please determine the correct definition of the target word in the context.\nContext: {tailored[i]['context']}\nTarget word: {tailored[i]['target']}"
    assistant = tailored[i]['output']
    correct_definitions_in_context = tailored[i]['correct_definitions_in_context']
    col['messages'].append({'role': 'user', 'content': user})
    col['messages'].append({'role': 'assistant', 'content': assistant})
    col['messages'].append({'role': 'correct_definitions_in_context', 'content': correct_definitions_in_context})
    collections.append(col)
print('total conversation samples:', len(collections))

# -------------------- 保存最终数据 --------------------
json.dump(
    collections,
    open('../data/syn_cot.json', 'w'),
    indent=2,
    ensure_ascii=False
)
