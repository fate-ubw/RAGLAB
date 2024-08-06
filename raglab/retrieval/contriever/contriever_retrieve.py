import os
from tqdm import tqdm
import time
import glob
import pickle
import numpy as np
import torch

from raglab.retrieval.contriever.src.index import Indexer
from raglab.retrieval.contriever.src.contriever import load_retriever
from raglab.retrieval.contriever.src.slurm import init_distributed_mode
from raglab.retrieval.contriever.src.data import load_passages
from raglab.retrieval.contriever.src.normalize_text import normalize
from raglab.retrieval.retrieve import Retrieve
import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "true"
class ContrieverRrtieve(Retrieve):
    def __init__(self, args):
        self.args = args
        self.index_dbPath = args.index_dbPath
        self.text_dbPath = args.text_dbPath
        self.retriever_modelPath = args.retriever_modelPath
        self.projection_size = args.projection_size
        self.indexing_batch_size = args.indexing_batch_size # TODO Can indexing_batch_size be unified with colbert's batchsize?
        self.n_subquantizers = args.n_subquantizers 
        self.n_bits = args.n_bits 
        self.n_docs = args.n_docs

    def setup_retrieve(self):
        init_distributed_mode(self.args) 
        print(f"Loading model from: {self.retriever_modelPath}")
        self.model, self.tokenizer, _ = load_retriever(self.retriever_modelPath)
        self.model.eval()
        self.model = self.model.cuda()
        self.model = self.model.half() # (32-bit) floating-point -> (16-bit) floating-point  
        self.index = Indexer(self.projection_size, self.n_subquantizers, self.n_bits) 
        input_paths = glob.glob(self.index_dbPath) # path of embedding
        input_paths = sorted(input_paths)
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        self.index_encoded_data(self.index, input_paths, self.indexing_batch_size) # load all embedding files（index_encoded_data）& （add_embeddings）
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        print("loading passages") 
        self.passages = load_passages(self.text_dbPath) # return list of passages
        self.passage_id_map = {x["id"]: x for x in tqdm(self.passages)} # define passages->{'n':conent}  
        print("passages have been loaded") 

    def search(self, query)-> dict[int,dict]:
        passages = {}
        questions_embedding = self.embed_queries(self.args, [query])
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, self.n_docs)  
        print(f"Search time: {time.time()-start_time_retrieval:.3f} s.") 
        passages = self.add_passages(self.passage_id_map, top_ids_and_scores)
        return passages

    def index_encoded_data(self, index, embedding_files, indexing_batch_size): 
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files): 
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin) # load idx and embedding of passages
            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids) #
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        print("Data indexing completed.")

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx] 
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd) 
        return embeddings, ids

    def embed_queries(self, args, queries): 
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries): 
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:  
                    q = normalize(q)
                batch_question.append(q)

                if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())
                    batch_question = []
        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")
        return embeddings.numpy() 

    def add_passages(self, passages, top_passages_and_scores):
        docs = {}
        for rank, (doc_id, score) in enumerate(zip(top_passages_and_scores[0][0], top_passages_and_scores[0][1])):
            passages_info = {'id':passages[doc_id]['id'], 'title': passages[doc_id]['title'], 'text':passages[doc_id]['text'],'score':float(score)}
            docs[rank+1] = passages_info
        return docs