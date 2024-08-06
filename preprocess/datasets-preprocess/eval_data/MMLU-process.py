import json
import os
import random

def load_json_data(file_path, max_items=None):
    data = []
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                #  JSON 
                json_data = json.load(f)
                data = random.sample(json_data, max_items) if max_items else json_data
            elif file_path.endswith('.jsonl'):
                #  JSONL 
                all_data = [json.loads(line) for line in f]
                if max_items:
                    data = random.sample(all_data, max_items)
                else:
                    data = all_data
            else:
                raise ValueError(f'Unsupported file format: {file_path}')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f'Error: {e}')
    return data

def main():
    file_path = '/home/wyd/zxw/raglab-exp/data/eval_datasets/MMLU/mmlu_all_test_data.jsonl'
    max_items = 500

    data = load_json_data(file_path, max_items)

    if data:
        print(f'Loaded {len(data)} items from {file_path}')
        print('First 5 items:')
        for item in data[:5]:
            print(item)
        
        file_name, file_ext = os.path.splitext(os.path.basename(file_path))
        new_file_name = f"{file_name}_random_{max_items}_samples.jsonl"
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

        with open(new_file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        print(f'Wrote {len(data)} items to {new_file_path}')
    else:
        print('No data loaded')

if __name__ == '__main__':
    main()