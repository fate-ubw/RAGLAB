import os
import csv
import json
from tqdm import tqdm
import pdb
folder_path = "/home/wyd/raglab-exp/data/eval_datasets/MMLU/data/test"

output_file = "/home/wyd/raglab-exp/data/eval_datasets/MMLU/mmlu_all_test_data.jsonl"

all_data = []

for filename in tqdm(os.listdir(folder_path)):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader: # list of sample
                tmp = {}
                tmp['question'] = row[0]
                tmp['answerKey'] = row[-1]
                tmp['choices'] = {'text':row[1:-1],'label':["A", "B", "C", "D"]}
                all_data.append(tmp) # list of dict

with open(output_file, "w") as jsonl_file:
    for data in all_data:
        json_data = json.dumps(data)
        jsonl_file.write(json_data + "\n")

print(f"success {output_file}")