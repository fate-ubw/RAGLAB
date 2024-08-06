import json

with open('/workspace/raglab-exp/data/train_data/full_output_1005.jsonl', 'r') as f_in, open('/workspace/raglab-exp/data/train_data/full_output_1005-10samples.jsonl', 'w') as f_out:
    for i, line in enumerate(f_in):
        data = json.loads(line)
        f_out.write(json.dumps(data) + '\n')
        if i == 49:
            break