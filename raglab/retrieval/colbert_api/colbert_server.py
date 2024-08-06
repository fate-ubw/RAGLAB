import argparse
import os
import math
from flask import Flask, render_template, request
from functools import lru_cache
# from dotenv import load_dotenv
from utils import over_write_args_from_file
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import pdb

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dbPath", type = str, help = 'path to index database. Index is index and embedding pairs')
    parser.add_argument('--text_dbPath', type = str, help='path to text database')
    parser.add_argument('--config',type = str, default = "")
    args = parser.parse_args()
    over_write_args_from_file(args, args.config)
    return args

class ColbertServer:
    def __init__(self, args):
        self.counter = 0
        self.index_dbPath = args.index_dbPath
        self.text_dbPath = args.text_dbPath
        self.retriever_modelPath = args.retriever_modelPath
        self.port = args.port
        self.app = Flask(__name__)
        self.setup_retrieve()
        self.app.add_url_rule("/api/search", view_func = self.api_search ,methods=["GET"])
        self.app.run("0.0.0.0", args.port)

    def setup_retrieve(self):
        index_name = os.path.basename(self.index_dbPath)
        with Run().context(RunConfig(experiment = self.index_dbPath)):
            self.searcher = Searcher(index = index_name, checkpoint = self.retriever_modelPath)

    def api_search(self):
        if request.method == "GET":
            self.counter += 1
            print(f"The {self.counter}th API call succeeded.")
            return self.api_search_query(request.args.get("query"), request.args.get("k"))
        else:
            return ('', 405)

    @lru_cache(maxsize=1000000)
    def api_search_query(self, query, k):
        if k == None: k = 10
        k = min(int(k), 100) 
        ids = self.searcher.search(query, k)
        passages = {}
        for passage_id, passage_rank, passage_score in zip(*ids):
            if '|' in self.searcher.collection[passage_id]:
                title, content = self.searcher.collection[passage_id].split('|',1) # max split is 1. The first place is title
                passages[passage_rank] = {'id':passage_id,'title':title.strip(),'text': content.strip(), 'score':passage_score}
            else:
                passages[passage_rank] = {'id':passage_id,'text': self.searcher.collection[passage_id], 'score':passage_score}
        return passages

if __name__ == "__main__":
    args = get_config()
    Server = ColbertServer(args)

