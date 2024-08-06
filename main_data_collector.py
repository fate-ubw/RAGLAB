import os
import json
import jsonlines
import argparse
from tqdm import tqdm
import time
from typing import List, Dict, Any
import random
from raglab.data_collector import DatasetCollector, CriticModelDatasetCollector, DatasetCollectorParallel
import pdb

class InvalidCollector(Exception):
    pass

train_name = "train_20w.jsonl"
test_name = "test_20w.jsonl"


def collect_datasets(args, dataset_config):
    train_output_path = os.path.join(args.output, train_name)
    test_output_path = os.path.join(args.output, test_name)
    # Load existing data if files exist
    all_train_data = load_jsonl(train_output_path) if os.path.exists(train_output_path) else []
    all_test_data = load_jsonl(test_output_path) if os.path.exists(test_output_path) else []
    # Keep track of processed datasets
    processed_datasets = set(item['raw_dataset'] for item in all_train_data + all_test_data)
    pbar = tqdm(dataset_config.items(), desc="Collecting datasets")

    for dataset_name, config in dataset_config.items():
        if dataset_name in processed_datasets:
            print(f"Skipping {dataset_name} as it has already been processed.")
            continue
        pbar.set_description(f"Collecting data for {dataset_name}")

        print(f"Collecting data for {dataset_name}")
        args.dataset_name  = dataset_name
        if args.collector_method  == 'selfrag-critic':
            collector = CriticModelDatasetCollector(args)
        elif args.collector_method  == 'base':
            collector = DatasetCollector(args)
        elif args.collector_method == 'base_parallel':
            collector = DatasetCollectorParallel(args)

        train_data, test_data = collector.collect_data(
            split=config['split'],
            n=config['samples'],
            format=config['format'],
            test_ratio=config['test_ratio']
        )

        all_train_data.extend(train_data)
        all_test_data.extend(test_data)
        # Save data after each dataset
        save_jsonl(all_train_data, train_output_path)
        save_jsonl(all_test_data, test_output_path)

        pbar.set_postfix(train_samples=len(train_data), test_samples=len(test_data))
        print(f"Collected {len(train_data)} training samples and {len(test_data)} test samples for {dataset_name}")
    print(f"Total collected: {len(all_train_data)} training samples and {len(all_test_data)} test samples")

    return all_train_data, all_test_data

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(file_path) as reader:
        for item in reader:
            data.append(item)
    return data

def preprocess_item(item):
    if isinstance(item.get('output'), (dict, list)):
        item['output'] = json.dumps(item['output'])
    return item

def save_jsonl(data: List[Dict[str, Any]], output_file: str):
    with jsonlines.open(output_file, mode='w') as writer:
        processed_data = [preprocess_item(item) for item in data]
        writer.write_all(processed_data)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect and process dataset samples.")
    # collecter config
    parser.add_argument('--collector_method', type=str,choices=['base', 'selfrag-critic', 'base_parallel'], default='base', help='diff method of collect  trian data for sft')
    parser.add_argument("--base_path", default="/home/wyd/FlashRAG/dataset", help="Base path for dataset directories")
    parser.add_argument("--output", default="collected_data.jsonl", help="Output file name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # language model
    parser.add_argument('--llm_mode', type = str, default='HF_Model', choices=['HF_Model','Openai_api', 'Lora_Model', 'Unified_api'], help='flag of language or api')
    parser.add_argument("--llm_path", type = str, help = 'path to llm')
    parser.add_argument("--dtype", type=str, default= "half", help="all base model inference using half(fp16)")
    parser.add_argument('--generate_maxlength', type = int, default = 50, help = 'llm generate max length')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature of decoding algorithm')
    parser.add_argument('--top_p', type=float, default=1.0, help='top-p of decoding algorithm')
    parser.add_argument('--generation_stop', type=str, default='', help='early_stop is one of the setting of generate() function, early_stop to control the outputs of llm')

    # api config
    parser.add_argument('--llm_name', type=str, default='gpt-3.5-turbo', help='language model name of openai api')
    parser.add_argument('--llm_api', type=str, help='API language model name')
    parser.add_argument('--api_key', type=str, help='API key for accessing the model')
    parser.add_argument('--api_base', type=str, help='Base URL for the API')
    parser.add_argument('--api_key_path', type=str, help='path of .txt which save api_key for openai api')
    parser.add_argument('--api_logprobs', type=int, default = False, help='Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.')
    parser.add_argument('--api_top_logprobs', type=int, default=1, help='An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.')
    parser.add_argument('--local_proxy_url', type = str, default=None, help = 'the url for local proxy')
    # retrieval config
    parser.add_argument('--retrieval_name', type = str, default = 'colbert_api', choices = ['colbert_api'], help = 'name of retrieval model')
    parser.add_argument("--n_docs", type= int, default=10, help="Number of documents to retrieve per questions")
    return parser.parse_args()

def main():
    args = parse_arguments()
    random.seed(args.seed)
    dataset_config = {
        # flashrag format dataset
        "2wikimultihopqa": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "ambig_qa": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "arc": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "asqa": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "bamboogle": {"samples": 303, "split": "test", "format": "flashrag", "test_ratio": 0.01},
        "boolq": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "commense_qa": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "eli5": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "fermi": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "hellaswag": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "hotpotqa": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "mmlu": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "ms_marco": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "musique": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "narrative_qa": {"samples": 303, "split": "test", "format": "flashrag", "test_ratio": 0.01},
        "nq": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "openbookqa": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "piqa": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "popqa": {"samples": 303, "split": "test", "format": "flashrag", "test_ratio": 0.01},
        "siqa": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "squad": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "triviaqa": {"samples": 303, "split": "dev", "format": "flashrag", "test_ratio": 0.01},
        "web_questions": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        "wiki_qa": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},

        # # alpaca format
        "tulu_v2": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "baize":{"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "code_alpaca": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "cot": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "dolly": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "flan_v2": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "gpt4_alpaca": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "hard_coded": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "lima": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "oasst1": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "open_orca": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "science": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "self_instruct": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "sharegpt": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "stanford_alpaca": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "tulu_v1": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "unnatural_instructions": {"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},
        # "wizardlm":{"samples": 303, "split": "train", "format": "stanford_alpaca", "test_ratio": 0.01},

        # discard
        # "truthful_qa": {"samples": 303, "split": "dev", "format": "flashrag", "test_ratio": 0.01},
        # "wned": {"samples": 303, "split": "dev", "format": "flashrag", "test_ratio": 0.01},
        # "ay2": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        # "trex": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        # "zsre": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        # "wow": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        # "fever": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        # "wikiasp": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        # "curatedtrec": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
        # "quartz": {"samples": 303, "split": "train", "format": "flashrag", "test_ratio": 0.01},
    }
    
    start_time = time.time()

    train_data, test_data = collect_datasets(args, dataset_config)
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    # Save train data
    train_output_path = os.path.join(args.output, train_name)
    save_jsonl(train_data, train_output_path)
    print(f"Saved {len(train_data)} training samples to {train_output_path}")
    # Save test data
    test_output_path = os.path.join(args.output, test_name)
    save_jsonl(test_data, test_output_path)
    print(f"Saved {len(test_data)} test samples to {test_output_path}")
    # 
    print("Data collection and storage complete.")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()