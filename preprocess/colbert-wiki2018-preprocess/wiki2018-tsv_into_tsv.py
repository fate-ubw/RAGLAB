import os
import pdb
from tqdm import tqdm

target_dir = './data/retrieval/colbertv2.0_passages/wiki2018/'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

with open('./data/retrieval/colbertv2.0_passages/wiki2018/psgs_w100.tsv', 'r') as file:
    lines = file.readlines()
    samples = lines[1:]  # remove first line

wiki_data = []
for sample in tqdm(samples):
    wiki_data.append(str(int(sample.split('\t', 2)[0]) - 1) + '\t' + sample.split('\t', 2)[1] + '\t' + sample.split('\t', 2)[2])

with open(os.path.join(target_dir, 'wiki2018.tsv'), 'w') as file:
    file.writelines(wiki_data)
