
import os
import sys
sys.path.insert(0, '../')
import pdb
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
import pudb
if __name__=='__main__':
    checkpoint = '/workspace/raglab-exp/model/colbertv2.0'
    index_dbPath = '/workspace/raglab-exp/data/retrieval/colbertv2.0_embedding/wiki2023-10samples'
    dataset = os.path.basename(index_dbPath)
    collection = '/workspace/raglab-exp/data/retrieval/colbertv2.0_passages/wiki2023-10samples/enwiki-20230401-10samples.tsv'
    collection = Collection(path=collection)
    f'Loaded {len(collection):,} passages'
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300  
    index_name = dataset 
    with Run().context(RunConfig(nranks=1, experiment=index_dbPath)):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)        
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True) #assert overwrite in [True, False, 'reuse', 'resume', "force_silent_overwrite"]
    indexer.get_index() # You can get the absolute path of the index, if needed.
    with Run().context(RunConfig(experiment=index_dbPath)):
        searcher = Searcher(index=index_name)
    
    query = 'who is Aaron?'   
    print(f"#> {query}")
    results = searcher.search(query, k=3)
    for passage_id, passage_rank, passage_score in zip(*results):
        print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")