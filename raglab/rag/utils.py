from raglab.rag.infer_alg.naive_rag import NaiveRag
from raglab.rag.infer_alg.iterative_rag import ItertiveRag
from raglab.rag.infer_alg.query_rewrite_rag import QueryRewrite_rag
from raglab.rag.infer_alg.active_rag import ActiveRag
from raglab.rag.infer_alg.self_rag_original import SelfRag_Original
from raglab.rag.infer_alg.self_rag_reproduction import SelfRag_Reproduction
from raglab.rag.infer_alg.self_ask import SelfAsk

ALGOROTHM_LIST = ['naive_rag', 'selfrag_original', 'selfrag_reproduction', 'iter_retgen', 'query_rewrite_rag', 'active_rag', 'self_ask']
def get_algorithm(args):
    if args.algorithm_name == 'naive_rag':
        Rag = NaiveRag(args)
    elif args.algorithm_name == 'selfrag_original':
        Rag = SelfRag_Original(args)
    elif args.algorithm_name == 'selfrag_reproduction':
        Rag = SelfRag_Reproduction(args)
    elif args.algorithm_name == 'iter_retgen':
        Rag = ItertiveRag(args)
    elif args.algorithm_name == 'query_rewrite_rag':
        Rag = QueryRewrite_rag(args)
    elif args.algorithm_name == 'active_rag':
        Rag = ActiveRag(args)
    elif args.algorithm_name == 'self_ask':
        Rag = SelfAsk(args)
    else:
        raise AlgorithmNotFoundError("Algorithm not recognized. Please provide a valid algorithm name.")
    return Rag

class AlgorithmNotFoundError(Exception):
    pass