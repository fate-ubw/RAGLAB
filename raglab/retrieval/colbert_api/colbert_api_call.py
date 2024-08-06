import sys
import requests
import time
from raglab.retrieval.retrieve import Retrieve
import pdb
RED = '\033[91m'
END = '\033[0m'

class ColbertApi(Retrieve):
    def __init__(self, args):
        self.n_docs = args.n_docs
        self.url_format = "http://localhost:8893/api/search?query={query}&k={k}"

    def setup_retrieve(self):
        # setup process done when setup ColbertServer
        pass

    def search(self,query:str)-> dict[int,dict]:
        start_time = time.time()
        url = self.url_format.format_map({'query':query, 'k':self.n_docs})
        try:
            response = requests.get(url).json()
            passages = {int(k):v for k,v in response.items()}
        except:
            print(f"{RED}Warning!!! ColbertServer didn't setup properly.{END}")
        print(f"Search time: {time.time()-start_time:.3f} s.") 
        return passages
