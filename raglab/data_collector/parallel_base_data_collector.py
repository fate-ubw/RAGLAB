import json
import random
import os
import gzip
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from abc import ABC
from raglab.instruction_lab import ALGORITHM_INSTRUCTIONS, DATA_INSTRUCTIONS
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pdb

class DatasetCollectorParallel(ABC):

    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.base_path = args.base_path

    def read_file(self, file_path: str) -> List[Dict[str, Any]]:
        if file_path.endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'

        with open_func(file_path, mode, encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            elif file_path.endswith('.jsonl') or file_path.endswith('.jsonl.gz'):
                return [json.loads(line.strip()) for line in f]
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

    def sample_data(self, data: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        return random.sample(data, min(n, len(data)))

    def process_item(self, item: Dict[str, Any], idx: int, format:str) -> list[Dict[str, Any]]:
        dataset_type = self.find_dataset_type(self.dataset_name)
        item = self.preprocessed_data(item, dataset_type)
        if format == 'flashrag':
            task_instruction = self.find_dataset_instruction(self.dataset_name)
            if task_instruction == '':
                return [{
                            "instruction": item.get("question", ""),
                            "input": "",
                            "output": item.get("golden_answers", [""])[0],
                            "raw_dataset": self.dataset_name,
                            "raw_dataset_idx": item.get("id", idx)
                        }]
            else:
                return [{
                            "instruction": task_instruction,
                            "input": item.get("question", ""),
                            "output": item.get("golden_answers", [""])[0],
                            "raw_dataset": self.dataset_name,
                            "raw_dataset_idx": item.get("id", idx)
                    }]
        elif format == 'stanford_alpaca':
            item["raw_dataset"] = self.dataset_name
            item["raw_dataset_idx"] = item.get("id", idx)
            return [item]
        else:
            raise FormatNotFoundError("invalid format, Please check dataset_config in main_data_collector")

    def collect_data(self, split: str, n: int, format: str, test_ratio: float = 0.1, num_chunks: int = 16) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        file_path = os.path.join(self.base_path, self.dataset_name, f"{split}.json")
        if not os.path.exists(file_path):
            file_path = os.path.join(self.base_path, self.dataset_name, f"{split}.jsonl")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return [], []
        processed_train_data = []
        processed_test_data  = []
        data = self.read_file(file_path)
        sampled_data = self.sample_data(data, n)
        total_samples = len(sampled_data)
        if total_samples == 1:
            # If there's only one sample, use it for both train and test
            train_data = sampled_data
            test_data = sampled_data
        elif total_samples == 2:
            # If there are two samples, split them evenly
            train_data = sampled_data[:1]
            test_data = sampled_data[1:]
        else:
            # Ensure at least one sample in each set
            test_size = max(1, int(total_samples * test_ratio))
            train_size = total_samples - test_size
            train_data = sampled_data[:train_size]
            test_data = sampled_data[train_size:]
        # Process train and test data
        processed_train_data = self._process_data_parallel(train_data, format, num_chunks)
        if total_samples == 1:
            processed_test_data = processed_train_data
        else:
            processed_test_data = self._process_data_parallel(test_data, format, num_chunks)

        return processed_train_data, processed_test_data

    def _process_data_parallel(self, data: List[Dict[str, Any]], format: str, num_chunks: int) -> List[Dict[str, Any]]:
        # Split data into num_chunks
        chunks = np.array_split(data, num_chunks)
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = [executor.submit(self._process_chunk, (chunk_idx, chunk.tolist()), format) for chunk_idx, chunk in enumerate(chunks)]
            results = [future.result() for future in as_completed(futures)]

        # Flatten the results
        processed_data = [item for chunk_result in results for item in chunk_result]
        return processed_data

    def _process_chunk(self, chunk_info: Tuple[int, List[Dict[str, Any]]], format: str) -> List[Dict[str, Any]]:
        chunk_idx, chunk = chunk_info
        processed_chunk = []
        chunk_size = len(chunk)
        for idx, item in enumerate(chunk):
            processed_item = self.process_item(item, idx + chunk_idx * chunk_size, format)
            if processed_item is not None:
                processed_chunk.extend(processed_item)
        return processed_chunk

    def _process_data(self, data:List[Dict[str, Any]], format: str) -> List[Dict[str, Any]]:
        processed_data = []
        for idx, item in enumerate(tqdm(data, total=len(data), desc=f"Processing {self.dataset_name}")):
            processed_item = self.process_item(item, idx, format)
            if processed_item is not None:
                processed_data.extend(processed_item)
            # save train or test
        return processed_data

    def find_dataset_instruction(self, dataset_name:str) -> str:
        target_instruction = ''
        for instruction in DATA_INSTRUCTIONS:
            if instruction["dataset_name"].lower() == dataset_name:
                target_instruction = instruction["instruction"]
        return target_instruction

    def find_dataset_type(self, dataset_name:str) -> str:
        dataset_type = ''
        for instruction in DATA_INSTRUCTIONS:
            if instruction['dataset_name'].lower() == dataset_name:
                dataset_type = instruction['type']
        return dataset_type

    def preprocessed_data(self, data: dict, type: str) -> dict:
        # Mapping numbers to letters
        if type == "MultiChoice":
            answer_map = ['A', 'B', 'C', 'D']
            # Extracting and formatting choices
            choices = data['choices']
            formatted_choices = [f"{answer_map[i]}. {choice}" for i, choice in enumerate(choices)]
            # Merging choices into the question
            question_with_choices = data['question'] + "\n" + "\n".join(formatted_choices)
            # Converting golden_answers number to corresponding letter
            golden_answers = [formatted_choices[ans] for ans in data['golden_answers']]

            # Creating the processed data dictionary
            processed_data = {
                'id': data['id'],
                'question': question_with_choices,
                'golden_answers': golden_answers,
                'metadata': data.get('metadata', {})
            }
            return processed_data
        else:
            pass
        return data

class FormatNotFoundError(Exception):
    pass