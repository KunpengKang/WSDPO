import argparse, json, os

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="./data/dev_bird.json")
    parser.add_argument('--output_dir', type=str, default='./data/splits')
    parser.add_argument('--num_splits', type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_option()
    print('Start Splitting Dataset...')
    print(opt)

    os.makedirs(opt.output_dir, exist_ok=True)

    # 读取 + 重构
    with open(opt.input, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        print(len(raw))
    data = []
    for d in raw:
        new_sample = {
            "input": f"Please determine the correct definition of the target word in the context.\nContext: {d['context']}\nTarget word: {d['target']}",
            "output": d["correct_definitions_in_context"][0]
        }


        data.append(new_sample)

    # 均分
    split_size = (len(data) // opt.num_splits) + 1
    split_data = [data[i*split_size:(i+1)*split_size] for i in range(opt.num_splits)]

    for i, split in enumerate(split_data):
        print(f'Split {i} size: {len(split)}')

    prefix = os.path.basename(opt.input).rsplit('.', 1)[0]
    for i, split in enumerate(split_data):
        out_path = os.path.join(opt.output_dir, f"{prefix}_part{i}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(split, f, ensure_ascii=False, indent=2)

    print('Splitting completed.')