import json

def load_json_data(file_path, max_items=None):
    data = []
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                #  JSON 
                json_data = json.load(f)
                data = json_data if max_items is None else json_data[:max_items]
            elif file_path.endswith('.jsonl'):
                #  JSONL 
                for line in f:
                    data.append(json.loads(line))
                    if max_items and len(data) >= max_items:
                        break
            else:
                raise ValueError(f'Unsupported file format: {file_path}')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f'Error: {e}')
    return data

def main():
    file_path = '/home/wyd/zxw/raglab-exp/data/eval_datasets/StrategyQA/strategyqa_train.json'
    max_items = 500
    data = load_json_data(file_path, max_items)
    if data:
        print(f'Loaded {len(data)} items from {file_path}')
        print('First 3 items:')
        for item in data[:3]:
            print(item)
        # 写入新文件
        new_file_path = f"{file_path.rsplit('.', 1)[0]}_first_{max_items}_samples.jsonl"
        with open(new_file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f'Wrote {len(data)} items to {new_file_path}')
    else:
        print('No data loaded')

if __name__ == '__main__':
    main()
