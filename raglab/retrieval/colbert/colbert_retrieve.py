import faiss
import os
import sys

from colbert.data import Collection
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from raglab.retrieval.retrieve import Retrieve
import pdb
RED = '\033[91m'
END = '\033[0m'

class ColbertRetrieve(Retrieve):
    def __init__(self, args):
        self.index_dbPath = args.index_dbPath
        self.text_dbPath = args.text_dbPath
        self.retriever_modelPath = args.retriever_modelPath
        self.nbits = args.nbits
        self.num_gpu = args.num_gpu
        self.doc_maxlen = args.doc_maxlen
        self.n_docs = args.n_docs

    def setup_retrieve(self):
        index_name = os.path.basename(self.index_dbPath)
        collection_path = self.text_dbPath
        collection = Collection(path = collection_path)
        try:
            with Run().context(RunConfig(experiment = self.index_dbPath)):
                self.searcher = Searcher(index = index_name)
        except:
            print(f'{RED}Warning!!! Your wiki database has issues. RagLab will enabe colbert to generate a vector database, which will take a long time. Please download the preprocessed wiki database from RagLab.{END}')
            user_input = input(f"Enter 'yes' to process data with Colbert Embedding, or 'no' to exit: ")
            if user_input.lower() == "yes":
                with Run().context(RunConfig(nranks = self.num_gpu, experiment = self.index_dbPath)):  # nranks specifies the number of GPUs to use.
                    config = ColBERTConfig(doc_maxlen = self.doc_maxlen, nbits = self.nbits, kmeans_niters = 4) # colbert default setting
                    indexer = Indexer(checkpoint = self.retriever_modelPath, config = config)
                    indexer.index(name = index_name, collection = collection, overwrite=True) # overwrite must in [True, False, 'reuse', 'resume', "force_silent_overwrite"] 
                with Run().context(RunConfig(experiment = self.index_dbPath)): 
                    self.searcher = Searcher(index = index_name)
            else:
                print("Exiting program...")
                sys.exit()

    def search(self, query) -> dict[int,dict]:
        ids = self.searcher.search(query, k = self.n_docs)
        passages = {}
        for passage_id, passage_rank, passage_score in zip(*ids):
            if '|' in self.searcher.collection[passage_id]:
                title, content = self.searcher.collection[passage_id].split('|',1) # max split is 1. The first place is title
                passages[passage_rank] = {'id':passage_id,'title':title.strip(),'text': content.strip(), 'score':passage_score}
            else:
                passages[passage_rank] = {'id':passage_id,'text': self.searcher.collection[passage_id], 'score':passage_score}
        return passages
