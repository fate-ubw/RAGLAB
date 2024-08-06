import argparse
import pdb
import pudb
import random
import torch
import numpy as np
from raglab.dataset.utils import TASK_LIST
from utils import over_write_args_from_file
from raglab.rag.utils import get_algorithm, ALGOROTHM_LIST

def set_randomSeed(args):
    # random seed
    if args.use_seed == True:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

def get_config():
    parser = argparse.ArgumentParser()
    # common config
    parser.add_argument('--seed', type=int, default = 633, help='random  seed')
    parser.add_argument('--use_seed', action= 'store_true', help='this args will control all random seed of torch, numpy and pyhthon operation')
    parser.add_argument('--num_gpu', type = int, default = 1, help = 'the number of gpu')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None, help='GPU IDs to use (e.g., 0 1 2 3)')

    # evaluation config
    parser.add_argument('--algorithm_name', type=str, default='naive_rag', choices= ALGOROTHM_LIST, help='name of rag algorithm' )
    parser.add_argument('--task', type=str, default='', choices= TASK_LIST,  help='name of evaluation dataset, different task will select different format and instruction')
    parser.add_argument("--eval_datapath", type = str, help = 'path to test or dev dataset')
    parser.add_argument('--eval_train_datapath', type= str, help='path to train dataset')
    parser.add_argument('--output_dir', type = str, help = 'the output dir of evaluation')

    # llm config
    parser.add_argument('--llm_mode', type = str, default='HF_Model', choices=['HF_Model','Openai_api', 'Lora_Model'], help='flag of language or api')
    parser.add_argument("--llm_path", type = str, help = 'path to llm')
    parser.add_argument('--download_dir', type=str, default=".cache",help="specify vllm model download dir")
    parser.add_argument("--world_size",  type=int, default=1,help="world size to use multiple GPUs. world_size will be used in LLM() function")
    parser.add_argument("--dtype", type=str, default= "half", help="all base model inference using half(fp16)")
    parser.add_argument('--generate_maxlength', type = int, default = 50, help = 'llm generate max length')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature of decoding algorithm')
    parser.add_argument('--top_p', type=float, default=1.0, help='top-p of decoding algorithm')
    parser.add_argument('--generation_stop', type=str, default='', help='early_stop is one of the setting of generate() function, early_stop to control the outputs of llm')
    parser.add_argument('--include_stop_token', type=int, default=False, help='"include_stop_token" controls whether the generated text output should include the provided stop string.')
    parser.add_argument('--use_vllm', action = "store_true", help = 'llm generate max length')
    parser.add_argument('--use_chat_template', type = int, default=False, help = 'llama2-chat and llama3-instruction ues official chat template will get a better performance, but finetune model will mess up by this template')
    # lora config
    parser.add_argument('--basemodel_path', type = str, help = 'path of lora base model')
    parser.add_argument("--quantization", type=str, default="None",choices=["8bit", "4bit", 'None'] ,help="quantization techniques when load model")

    # api config
    parser.add_argument('--llm_name', type=str, default='gpt-3.5-turbo', help='language model name of openai api')
    parser.add_argument('--llm_api', type=str, help='API language model name')
    parser.add_argument('--api_key', type=str, help='API key for accessing the model')
    parser.add_argument('--api_base', type=str, help='Base URL for the API')
    parser.add_argument('--api_key_path', type=str, help='path of .txt which save api_key for openai api')
    parser.add_argument('--api_logprobs', type=int, default = True, help='Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.')
    parser.add_argument('--api_top_logprobs', type=int, default=1, help='An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.')
    # retrieval config
    parser.add_argument('--realtime_retrieval', type = int, default=True, help='self rag can use local passages(only)')
    parser.add_argument('--retrieval_name', type = str, default = 'colbert', choices = ['colbert','contriever', 'colbert_api','pregiven_passages'],help = 'name of retrieval model')
    parser.add_argument("--index_dbPath", type = str, help = 'path to index database. Index is index and embedding pairs')
    parser.add_argument('--text_dbPath', type = str, help='path to text database')
    parser.add_argument("--retriever_modelPath", type = str, help = 'path to colbert model')
    parser.add_argument("--n_docs", type= int, default=10, help="Number of documents to retrieve per questions")
    parser.add_argument('--passages_max_length',type=int, default=-1, help = "-1 close truncation feature. When use some databas,each passaegs has 700 words in average, and 10 passages will exceed max length of LLM. As a result, we truncate each passages")
    # colbert config
    parser.add_argument('--doc_maxlen', type = int, default = 300, help = '[colbert configs] the doc max len decided by the wikidata format, here we set 300')
    parser.add_argument('--nbits', type = int, default = 2, help = 'encode each dimension with n bits')
    # contrieval conefig
    parser.add_argument('--projection_size', type = int, default=768, help = 'size of embedding for contrieval')
    parser.add_argument('--n_subquantizers', type = int, default=0, help="Number of subquantizer used for vector quantization, if 0 flat index is used")
    parser.add_argument('--n_bits', type = int, default = 8, help="Number of bits per subquantizer")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed")   
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")# no need thie parameter in colbert
    parser.add_argument("--normalize_text", action="store_true", help="normalize text") # no need thie parameter in colbert
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    # rag common config
    parser.add_argument('--use_citation',  action="store_true", help='add citation for responses')
    # TODO Only selfrag algorithm realize use_citation in this verison of raglab. Next version will update citation feature for all algorithm
    # self rag config
    parser.add_argument('--threshold', type=float, default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true", help="use ground score")
    parser.add_argument("--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width", type=int, default=2, help="beam search width")
    parser.add_argument("--max_depth",type=int, default=2, help="tree depth width")
    parser.add_argument("--w_rel", type=float, default=1.0, help="reward weight for document relevance")
    parser.add_argument("--w_sup", type=float, default=1.0, help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use", type=float, default=1.0,help="reward weight for overall completeness / utility.")
    parser.add_argument('--retrieval_mode', type=str, help="mode to control retrieval.", default="no_retrieval", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieval']) 
    parser.add_argument('--show_specialtokens', action="store_true", help='show special tokens or remove all special tokens in outputs')
    parser.add_argument('--inference_form', type=str, default='long_form', choices=['long_form', 'short_form'], help='self rag includes short form inference and long form inference')
    parser.add_argument("--ignore_cont", action="store_true", help="filter out sentences that include [No support / Contradictory] ") 

    # Iterative rag config
    parser.add_argument('--max_iteration', type=int, default=3, help='max number of iteration in Iterative rag')

    # Active rag config
    parser.add_argument('--max_fianl_answer_length', type=int, default=300, help='max length of final answer')
    parser.add_argument('--filter_prob', type=float, default=0.8, help='filter prob is lower probability threshold in paper(https://arxiv.org/abs/2305.06983)')
    parser.add_argument('--masked_prob', type=float, default=0.4, help='masked prob is low-confidence threshold in paper(https://arxiv.org/abs/2305.06983)')

    # self ask config
    parser.add_argument('--selfask_max_iter', type=int, default=5, help='max iter of follow qeustion generation. In some situation, self ask will get stuck in a loop')
    # critic config
    parser.add_argument('--critic_path', type=str, help='path of unified critic model path')
    # evaluate parameters
    parser.add_argument('--metrics', type=str, help='Evaluation metrics')

    # config file
    parser.add_argument('--config',type = str, default = "")
    args = parser.parse_args()
    over_write_args_from_file(args, args.config)
    return args

if __name__=='__main__':
    args = get_config()
    set_randomSeed(args)
    rag = get_algorithm(args)
    evaluation_result = rag.inference(mode = 'evaluation')
    print(evaluation_result)