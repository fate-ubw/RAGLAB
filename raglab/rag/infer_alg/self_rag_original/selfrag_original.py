from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import numpy as np
from typing import Any
from tqdm import tqdm
import pdb

from raglab.dataset.base_dataset import MultiChoiceQA 
from raglab.dataset.utils import get_dataset
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
from raglab.rag.infer_alg.self_rag_original.utils import load_special_tokens, postprocess_answer_option_conditioned, preprocess_input_data
from raglab.rag.infer_alg.self_rag_original.utils import PROMPT_DICT, TASK_INST,process_data_evidences, postprocess, fix_spacing

class SelfRag_Original(NaiveRag):
    '''
    Orignal self rag algorithm. Raglab rewrite the self rag algorithm.
    '''
    def __init__(self, args):
        self.task = args.task
        self.llm_path = args.llm_path
        self.generate_maxlength = args.generate_maxlength
        self.use_vllm = args.use_vllm
        self.eval_datapath = args.eval_datapath
        self.output_dir = args.output_dir
        # llm config
        self.llm_path = args.llm_path
        self.dtype = args.dtype
        self.generate_maxlength = args.generate_maxlength
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.generation_stop = args.generation_stop
        if  self.generation_stop == '\\n':
            self.generation_stop = '\n'
        self.dtype = args.dtype
        self.world_size = args.world_size        
        # llm config
        self.llm, self.tokenizer, self.sampling_params = self.load_llm()
        #retrieval args
        self.n_docs = args.n_docs
        # self.retrieval = self.setup_retrieval()
        '''
        Diff:  we have maintained the original self_rag code in slefrag_original.py. 
        As a result, the real-time retrieval functionality has not been implemented.
        '''
        # SelfRag args
        self.download_dir = args.download_dir
        self.threshold = args.threshold
        self.use_seqscore = args.use_seqscore
        self.use_groundness = args.use_groundness
        self.use_utility = args.use_utility
        self.beam_width = args.beam_width
        self.max_depth = args.max_depth
        self.w_rel = args.w_rel
        self.w_sup = args.w_sup
        self.w_use = args.w_use
        self.retrieval_mode = args.retrieval_mode
        self.show_specialtokens = args.show_specialtokens
        self.inference_form = args.inference_form
        self.ignore_cont = args.ignore_cont
        self.use_citation = args.use_citation

    def inference(self, mode='evaluation'):
        assert mode in ['evaluation']
        # get dataset
        self.EvalData = get_dataset(self.task, self.output_dir,self.llm_path, self.eval_datapath)
        self.eval_dataset = self.EvalData.load_dataset()
        #TODO seperate instruction and preprocess from logic. Combine instruction and dataset class
        self.eval_dataset = preprocess_input_data(self.eval_dataset, task = self.task) # find task instruction 
        inference_results = []
        for instance_idx, eval_data in enumerate(tqdm(self.eval_dataset)):
            temp = {}
            source_question = eval_data['question']
            _, evidences = process_data_evidences(eval_data, self.n_docs) # use pre-given passages from the eval_data.jsonl
            
            if 'short_form' == self.inference_form:
                input = PROMPT_DICT["prompt_no_input"].format_map(eval_data) 
                response, generation_track, do_retrieve = self.short_form_generation(prompt=input, source_question=source_question, evidences=evidences,
                                                        use_seqscore = self.use_seqscore, threshold = self.threshold,
                                                        w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use)
                print(f'source question:{source_question}')
                print(f'response: {response}')
                if "SUPPORTS" in response: # the trick in self rag source code. In some situation LLM will generate SUPPORTS or REFUTES instead of true or flase
                    response = "true" 
                elif "REFUTES" in response: 
                    response = "false"
                temp['question'] = source_question
                temp['answers'] = eval_data['answers']
                temp['generation'] = response
                temp['instruction'] = input
                temp['generation_track'] = generation_track
                inference_results.append(temp)
                # calculate the error in each step
                eval_result = self.EvalData.eval_acc(inference_results)
                print(f'{self.task} Accuracy in {instance_idx} turn: {eval_result}')
            elif 'long_form' == self.inference_form:
                if self.task in TASK_INST:
                    instructions = TASK_INST[self.task]
                    prompt = instructions + "## Input:\n\n" + source_question
                elif self.task == 'Factscore':
                    prompt = eval_data['input']
                input = PROMPT_DICT["prompt_no_input"].format_map({"instruction": prompt})
                final_prediction, generation_track, do_retrieve_flag = self.long_form_generation(prompt=input, query=source_question, ctxs=evidences, 
                                                   beam_width=self.beam_width, max_depth=self.max_depth, 
                                                   w_rel=self.w_rel, w_sup=self.w_sup, w_use=self.w_use, use_seqscore=self.use_seqscore,ignore_cont=self.ignore_cont)
                
                final_prediction_with_citation, catation_docs = self.aggregate_response_with_citation(final_prediction, generation_track, add_citation=self.use_citation)                
                response_id = 0
                inference_results = self.EvalData.record_result(eval_data, final_prediction_with_citation, 
                                                        catation_docs, response_id, generation_track,
                                                        inference_results)
        # end of for loop
        self.EvalData.save_result(inference_results) 
        if 'short' == self.inference_form:
            eval_result = self.EvalData.eval_acc(inference_results)
            print(f'Final {self.task} accuracy: {eval_result}')
            return eval_result
        elif 'long' == self.inference_form:
            return 'Inference completion'
    
    def load_llm(self):
        llm = LLM(model=self.llm_path, dtype=self.dtype)
        sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_path, padding_side="left")
        return llm, tokenizer, sampling_params
    
    def aggregate_response_with_citation(self, final_predictions: dict[int,str], generation_track: dict[str, Any], add_citation = True)-> tuple[dict, dict]:
        '''
        # Aggregate response for response. If the response generate by no_retrieval mode. There is no need to add citation. 
        '''
        previous_generations = []
        output_with_citation = {}
        catation_doc = {}
        for response_idx, generated_response in final_predictions.items(): 
            final_output = ""
            docs = []
            if "splitted_sentences" not in generation_track:
                output_with_citation[response_idx] = fix_spacing(postprocess(generated_response))
                catation_doc[response_idx] = docs
            else:
                if len(postprocess(generated_response)) == 0:
                    generation_track["splitted_sentences"][response_idx], generation_track["ctxs"][response_idx] = generation_track["splitted_sentences"][response_idx], generation_track["ctxs"][response_idx] 
                for cite_idx, (sentence, doc) in enumerate(iterable=zip(generation_track["splitted_sentences"][response_idx], generation_track["ctxs"][response_idx])):
                    if len(sentence) == 0:
                        continue
                    postprocessed_result = postprocess(sentence) 
                    # remove the loopping sentence 
                    if postprocessed_result in previous_generations: 
                        continue
                    else:
                        previous_generations.append(postprocessed_result)
                    if add_citation == True:
                        final_output += postprocessed_result[:-1] + " [{}]".format(cite_idx) + ". " 
                    else:
                        final_output += postprocessed_result
                    docs.append(doc) # docs -> list[dict]

                if len(final_output) == 0:
                    final_output = fix_spacing(final_output)  
                if len(final_output) > 0 and final_output[-1] == " ":
                    final_output = final_output[:-1]
                final_output = fix_spacing(final_output)
                if add_citation == True:
                    final_output = final_output.replace(".[Continue to Use Evidence]", " [1]. ") #Diff: the source selfrag replace each [Continue to Use Evidence] to [1], but in multi-sentence answer this should 
                    final_output = final_output.replace(". [1] ", " [1]. ")
                output_with_citation[response_idx] = final_output
                catation_doc[response_idx] = docs
        # end of the for loop
        return output_with_citation, catation_doc

    def long_form_generation(self, prompt: str, query: str, ctxs=None,                              
                                     beam_width=2, max_depth=7,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore = True,ignore_cont = None) -> tuple[dict[int,str], dict, bool]: # orignal version of self rag longform
        
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(self.tokenizer, 
                                                                            use_grounding=self.use_groundness, 
                                                                            use_utility=self.use_utility)

        if  "no_retrieval" == self.retrieval_mode: 
            prompt += "[No Retrieval]" 
            preds = self.llm.generate([prompt], self.sampling_params)
            preds_text = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
            final_prediction = {0:preds_text[0]} 
            generation_track = {"original_splitted_sentences": {0:preds_text}}
            return final_prediction, generation_track, False
        
        elif "always_retrieval" == self.retrieval_mode:
            do_retrieve = True
        elif 'adaptive_retrieval' == self.retrieval_mode:
            do_retrieve = self.firstToken_retrievalRatio_longForm(prompt, ret_tokens)

        if do_retrieve is False:
            # no retrieval
            prompt += "[No Retrieval]"
            preds = self.llm.generate([prompt], self.sampling_params)
            preds_text = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
            final_prediction = {0:preds_text[0]}
            generation_track = {"original_splitted_sentences": {0:preds_text}}

            return final_prediction, generation_track, do_retrieve
        elif do_retrieve is True:
            # 开始 always or adaptive retrieval
            curr_depth = 1 
            terminated = False 
            node_id = 0 
            prediction_tree = {} 
            levels = {} 
            prediction_tree[node_id] = {"prompt": prompt, "pred": "[Retrieval]", 
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None}
            levels[0] = [0]
            while curr_depth < max_depth: 
                # build tree
                levels[curr_depth] = []
                if curr_depth-1 in levels and terminated is False:
                    for parent_node in levels[curr_depth-1]:
                        prev_pred, prompt, prev_generation, prev_score = self.get_lastTurn_generation(parent_node, prediction_tree)
                        if prev_pred == "</s>":
                            terminated = True
                            continue
                        if "[Retrieval]" in prev_pred:
                            curr_prompt = prompt + prev_generation # get new prompt
                            curr_preds, curr_scores, overall_score_dict = self.run_step_generation_batch(curr_prompt, ctxs,
                                                                                                         rel_tokens=rel_tokens,grd_tokens=grd_tokens,
                                                                                                        ret_tokens=ret_tokens, ut_tokens=ut_tokens,
                                                                                                         w_rel=w_rel, w_sup=w_sup, w_use=w_use, use_seqscore=use_seqscore)
                            prediction_tree, node_id = self.set_predictionTree(curr_depth, parent_node, node_id, curr_preds, curr_scores, curr_prompt, 
                                                                            prev_score, ctxs, prediction_tree, levels ,overall_score_dict)
                    current_rank = levels[curr_depth]
                    node2score = { node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[:beam_width] # 取 top2 结果
                    levels[curr_depth] = [node[0] for node in top_nodes] 
                    curr_depth += 1  
                else:
                    break

        best_selections = self.backtracking_prediction_tree(levels, curr_depth, prediction_tree)
        # get final_prediction 
        final_prediction = {}
        splitted_sentences = {}
        original_splitted_sentences = {}
        ctxs = {}
        for path_i, nodes in best_selections.items(): # 
            # (Pdb) nodes = [None, 0, 5] 
            final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
            splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

            ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
                ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

        generation_track = {"final_prediction": final_prediction,
                "splitted_sentences": splitted_sentences,
                "original_splitted_sentences": original_splitted_sentences,
                "best_selections": best_selections,
                "ctxs": ctxs,
                "prediction_tree": prediction_tree}
        
        return final_prediction, generation_track, do_retrieve

    def run_step_generation_batch(self, prompt, paragraphs,
                                  rel_tokens=None, grd_tokens=None, ret_tokens=None, ut_tokens=None,
                                  w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False) -> tuple[list[str], list[float], dict]:
        if paragraphs is not None:
            aug_prompts = [prompt + "[Retrieval]" + "<paragraph>{}</paragraph>".format(paragraph["title"] + "\n" + paragraph["text"]) for paragraph in paragraphs]
        else:
            aug_prompts = [prompt]

        preds = self.llm.generate(aug_prompts, self.sampling_params)
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        final_preds = []
        for p_idx, pred in enumerate(preds): 
            pred_text = pred.outputs[0].text
            print(f'output_text"{pred_text}')
            # calculate seq score
            seq_score = self.sequence_score(pred)
            # init dict in each loop
            relevance_score_dict.setdefault(p_idx, {}) 
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # relevance score 
            relevance_score, relevance_score_dict = self.relevanceToken_score(pred, rel_tokens, p_idx, relevance_score_dict)
            # Issupport score
            ground_score, grd_score_dict = self.IssupportToken_score(pred, grd_tokens, p_idx, grd_score_dict)
            # Utility score
            utility_score, ut_score_dict = self.UtilityToken_score_longform(pred, ut_tokens, grd_tokens,p_idx, ut_score_dict) #Diff:  selfrag_reproduction.py we use self.UtilityToken_score() calculate the correct utility_score
            if self.use_seqscore is True:
                final_score = seq_score + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score +  w_sup * ground_score + w_use * utility_score
            
            overall_scores[p_idx] = {"final_score": final_score, 
                                    "relevance_score": relevance_score,
                                    "ground_score": ground_score,
                                    "utility_score": utility_score,
                                    "relevance_score_dict": relevance_score_dict, 
                                    "grd_score_dict": grd_score_dict,
                                    "ut_score_dict": ut_score_dict}

            if "[No Retrieval]" in pred_text: 
                pred_text = self.modify_NoRetrieval_into_Retrieval(pred, ret_tokens)
                '''
                Diff: The method "modify_NoRetrieval_into_Retrieval" is not explicitly mentioned in the paper or its abbreviated inference. 
                Consequently, in selfrag_reproduction.py, we have adhered to the paper's standard, and this particular function has been omitted. 
                The primary objective of selfrag_reproduction.py is to precisely assess the performance of Self-RAG in long-form inference.
                '''
                final_preds.append(pred_text)
            else:
                final_preds.append(pred_text)
        # end of the "for p_idx, pred in enumerate(preds):"
        preds = final_preds
        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores]
        return preds, scores, overall_scores

    def modify_NoRetrieval_into_Retrieval(self,pred, ret_tokens)-> str:
        '''
        check the ratio of ([Retrieval] + [Continue to Use Evidence])/([Retrieval] + [Continue to Use Evidence] + [No Retrieval] )
        if the ratio > threshold modify [No Retrieval] -> [Retrieval]
        '''
        pred_text = pred.outputs[0].text
        pred_log_probs = pred.outputs[0].logprobs 
        pred_token_ids = pred.outputs[0].token_ids
        ret_token_appear_indices = []
        substrings = pred_text.split("[No Retrieval]")
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok == ret_tokens["[No Retrieval]"]:
                ret_token_appear_indices.append(tok_idx)
                substrings
                print("retrieval_tokens")

        ret_token_score_dict = {}
        retrieval_remap = {}
        for order, idx in enumerate(ret_token_appear_indices):
            ret_token_score_dict.setdefault(order, {})
            for tok, tok_id in ret_tokens.items(): 
                prob = pred_log_probs[idx][tok_id] if tok_id in pred_log_probs[idx] else -100
                ret_token_score_dict[order][tok] = np.exp(prob)
            if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0: 
                do_retrieve = (ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[Continue to Use Evidence]"]) / (
                    ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]) > self.threshold
            else:
                do_retrieve = 0.0
            if do_retrieve > self.threshold:
                retrieval_remap[order] = True
            else:
                retrieval_remap[order] = False
        processed_pred = ""
        for substr_i, substring in enumerate(iterable=substrings):
            if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                processed_pred += substring + "[Retrieval]" 
            else:
                processed_pred += substring + "[No Retrieval]"
        return processed_pred


    def backtracking_prediction_tree(self, levels: dict[int,list[int]], curr_depth: int, prediction_tree: dict[int, dict]) -> dict[int,list[int]]:
        '''
        get best tracking from prediction_tree base on levels
        '''
        parent = 0 
        best_selections = {}
        # Traverse from the bottom 
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0} # remove empty list in levels
        for path_i, node in enumerate(levels[len(levels)]):
            if node == 0:
                break
            best_selections[path_i] = [node] 
            current_node = node 
            current_level = curr_depth 
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]["parent"]
                best_selections[path_i] = [parent] + best_selections[path_i] 
                current_node = parent 
                current_level += 1
        return best_selections
    
    def set_predictionTree(self, curr_depth, parent_node, node_id,  curr_preds:list[str], curr_scores:list[float], curr_prompt:str, prev_score, ctxs, prediction_tree, levels , overall_score_dict):
        retrieval_results = {}
        for i, (curr_pred, p_score) in enumerate(zip(curr_preds, curr_scores)):
            retrieval_results[i] = {"pred": curr_pred, "score": p_score}

        for i, result in retrieval_results.items(): 
            node_id += 1 
            node_score = result["score"] * prev_score if prev_score is not None else result["score"]
            curr_pred = result["pred"] 
            prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred, 
                                        "score": node_score, "ctx": ctxs[i], "parent": parent_node,
                                        "overall_score_dict": overall_score_dict} 
            
            if "[Retrieval]" in curr_pred: 
                gen_result_index = curr_pred.index("[Retrieval]") 
                prev_generation = curr_pred[:gen_result_index] 
            else: 
                prev_generation = curr_pred
            # Diff: check wrong pattern and cutting the wrong pattern in curr_pred. 
            prediction_tree[node_id]["processed_pred"] = prev_generation 
            levels[curr_depth].append(node_id) #这个就不对了呀
        return prediction_tree, node_id

    def get_lastTurn_generation(self, parent_node, prediction_tree):
        ''' 
        get previous information from prediction_tree
        '''
        prev_pred = prediction_tree[parent_node]["pred"]
        prev_prompt = prediction_tree[parent_node]["prompt"]
        prev_generation = prediction_tree[parent_node]["processed_pred"]
        prev_generationScore = prediction_tree[parent_node]["score"]
        return prev_pred, prev_prompt, prev_generation, prev_generationScore

    def short_form_generation(self, prompt:str, source_question:str, evidences = None,
                            use_seqscore=False, threshold=0.2,w_rel=1.0, w_sup=1.0, w_use=0.5): 
        
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
                                self.tokenizer, use_grounding=self.use_groundness, use_utility=self.use_utility)
        results = {}
        if 'always_retrieval' == self.retrieval_mode:
            do_retrieve = True
        elif 'no_retrieval' == self.retrieval_mode:
            do_retrieve = False
        elif 'adaptive_retrieval' == self.retrieval_mode:
            #retrieval or not base on first token
            ratio, results = self.firstToken_retrievalRatio_shortForm(prompt, ret_tokens, results)
            do_retrieve = ratio > threshold
        # "do retrieval or not retrieval
        if do_retrieve is True:             

            evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(para["title"], para["text"]) for para in evidences] 
            
            preds = self.llm.generate(evidence_augmented_inputs, self.sampling_params)
            # calculate score of each candidate
            relevance_score_dict = {}   
            grd_score_dict = {}
            ut_score_dict = {}
            overall_scores = {}
            for p_idx, pred in enumerate(preds): 
                #sequence score 
                seq_score = self.sequence_score(pred)
                # init dict in each loop
                relevance_score_dict.setdefault(p_idx, {}) 
                grd_score_dict.setdefault(p_idx, {})
                ut_score_dict.setdefault(p_idx, {})
                # relevance score 
                relevance_score, relevance_score_dict = self.relevanceToken_score(pred, rel_tokens, p_idx, relevance_score_dict)
                # Issupport score
                ground_score, grd_score_dict = self.IssupportToken_score(pred, grd_tokens, p_idx, grd_score_dict)
                # Utility score
                utility_score, ut_score_dict = self.UtilityToken_score(pred, ut_tokens, p_idx, ut_score_dict)
                
                if use_seqscore is True:
                    final_score = seq_score + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score # 涉及不同类型数据转化的一定要涉及类型的转换和精度问题
                else:
                    final_score = w_rel * relevance_score +  w_sup * ground_score + w_use * utility_score
                overall_scores[p_idx] = {"final_score": final_score,
                                        "relevance_score": relevance_score,
                                        "ground_score": ground_score,
                                        "utility_score": utility_score,
                                        "relevance_score_dict": relevance_score_dict, 
                                        "grd_score_dict": grd_score_dict,
                                        "ut_score_dict": ut_score_dict} 
                pred_text = pred.outputs[0].text 
                results["retrieval_{}".format(p_idx)] = {"pred": pred_text, "score": float(final_score), "ctx": evidences[p_idx]}
        else: 
            # no retrieval generation
            prompt += "[No Retrieval]"
            preds = self.llm.generate([prompt], self.sampling_params)
            pred = preds[0].outputs[0].text 
            results['no_retrieval'] = {"pred": pred} # no retrieval no need score and passages
        
        # Aggregating answers
        if len(results) <= 2: 
            # post process for no retrieval
            if True == self.show_specialtokens: 
                return pred, results, do_retrieve
            else:
                # remove all sprcial tokens 
                postprocessed_pred = postprocess_answer_option_conditioned(pred) 
                return postprocessed_pred, results, do_retrieve 
        else:
            answer2score = {}
            if isinstance(self.EvalData, MultiChoiceQA): 
                '''
                Aggregating for multi-choice question
                source explaination: For SELF-RAG inference on PubHealth and ARC-C, instead of determining the output with the highest score as in other tasks, 
                                    we aggregate the scores for each option and select the answer option with the highest score.       
                paper: https://arxiv.org/abs/2310.11511
                '''
                for key, result in results.items():
                    if key == "decide_retrieval_mode":
                        continue
                    answer = postprocess_answer_option_conditioned(result["pred"])
                    score = result["score"]
                    answer2score.setdefault(answer, 0)
                    answer2score[answer] += score
                sorted_answers = sorted(
                    answer2score.items(), key=lambda x: x[1], reverse=True)
                best_option = sorted_answers[0][0]
            else:
                path2score = {key: item["score"] for key, item in results.items() if key != "decide_retrieval_mode"} 
                best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
                best_option = results[best_path]["pred"]
                if self.show_specialtokens == True:
                    pass
                else:
                    best_option = postprocess_answer_option_conditioned(best_option)

        return best_option, results, do_retrieve 

    def firstToken_retrievalRatio_shortForm(self, prompt, ret_tokens, results):
        '''
        calculate the ratio of retrieval base on first token
        '''
        preds = self.llm.generate([prompt], self.sampling_params)
        pred_log_probs = preds[0].outputs[0].logprobs 
        score_dict = {}
        for tok, id in ret_tokens.items():
            if id not in pred_log_probs[0]:
                score_dict[tok] = -100
            prob = pred_log_probs[0][id] 
            score_dict[tok] = float(prob) 
            '''
            Diff: this code should be: score_dict[tok] = np.exp(float(prob)) 
            This bug is from self rag source code [https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py#L79]
            The correct version of self rag referenced in Raglab's Selfrag-correct 
            '''
        results["decide_retrieval_mode"] = preds[0].outputs[0].text 
        ratio = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])  
        return float(ratio), results

    def firstToken_retrievalRatio_longForm(self, prompt, ret_tokens):
        # the logic is reference from origanl code
        preds = self.llm.generate([prompt], self.sampling_params)
        pred_log_probs = preds[0].outputs[0].logprobs 
        preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        '''
        Diff: the algotithm of decide retrieval or no retrieval is different to paper and short form inferene code. 
        We reproduce the code and rewrite this part of code. 
        '''
        if "[Retrieval]" not in preds[0]: 
            # In fact, preds[0] will never contain [Retrieval]. We just copy this snippet of the code from the orignal github repository[https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_long_form_static.py#L219].
            do_retrieve = False
        else:
            if self.threshold is None:
                do_retrieve = False
            else:
                ret_token_score_dict = {}
                for tok, tok_id in ret_tokens.items():
                    prob = pred_log_probs[0][tok_id] 
                    ret_token_score_dict[tok] = np.exp(prob)
                retrieve_prob = ret_token_score_dict["[Retrieval]"] / (ret_token_score_dict["[Retrieval]"] + ret_token_score_dict["[No Retrieval]"])
                do_retrieve = True if retrieve_prob > self.threshold else False
        return  do_retrieve

    def sequence_score(self,pred) ->float:
        '''
        average prob of generated sentence
        '''
        score = np.exp(pred.outputs[0].cumulative_logprob) / max(len(pred.outputs[0].token_ids), 1)
        return float(score)

    def relevanceToken_score(self, pred, rel_tokens:dict[str,int], p_idx:int, relevance_score_dict:dict) -> tuple[float, dict]:
        pred_log_probs = pred.outputs[0].logprobs
        for tok, id in rel_tokens.items(): 
            prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
            relevance_score_dict[p_idx][tok] = np.exp(float(prob))
        # calculate score
        relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (np.sum(list(relevance_score_dict[p_idx].values())))
        return float(relevance_score), relevance_score_dict

    def IssupportToken_score(self, pred, grd_tokens, p_idx, grd_score_dict) -> tuple[float, dict]:
        pred_token_ids = pred.outputs[0].token_ids
        pred_log_probs = pred.outputs[0].logprobs
        groundness_token_appear_indices = []
        # get the position of Issupport token
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(grd_tokens.values()):
                groundness_token_appear_indices.append(tok_idx)
                break
        if len(groundness_token_appear_indices) > 0:
            idx = groundness_token_appear_indices[0]
            for token, token_id in grd_tokens.items():
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100 
                grd_score_dict[p_idx][token] = np.exp(float(prob))
        # calculate score
        if len(grd_score_dict[p_idx]) == 3: 
            gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
            ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (grd_score_dict[p_idx]["[Partially supported]"] / gt_sum) # 
        else:
            ground_score = 0.0 # "If the sentence is labeled as [isRel], then [Issup] will not appear later, resulting in a ground score of 0."
        return float(ground_score), grd_score_dict
    
    def UtilityToken_score(self, pred, ut_tokens, p_idx, ut_score_dict) -> tuple[float, dict]:
        pred_token_ids = pred.outputs[0].token_ids
        pred_log_probs = pred.outputs[0].logprobs
        utility_token_appear_indices = []
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(ut_tokens.values()):
                utility_token_appear_indices.append(tok_idx)
        if len(utility_token_appear_indices) > 0:
            idx = utility_token_appear_indices[0] # position of ut_token [Utility:1-5]
            for token, token_id in ut_tokens.items():
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                ut_score_dict[p_idx][token] = np.exp(float(prob))

        if len(ut_score_dict[p_idx]) == 5: 
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            ut_scores = [-1, -0.5, 0, 0.5, 1]
            utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
        else:   
            utility_score = 0.0
        return float(utility_score), ut_score_dict
    
    def UtilityToken_score_longform(self, pred, ut_tokens, grd_tokens, p_idx:int, ut_score_dict) -> tuple[float, dict]:
        pred_token_ids = pred.outputs[0].token_ids
        pred_log_probs = pred.outputs[0].logprobs
        utility_token_appear_indices = []
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(ut_tokens.values()):
                utility_token_appear_indices.append(tok_idx)
        if len(utility_token_appear_indices) > 0:
            idx = utility_token_appear_indices[0] # position of ut_token [Utility:1-5]
            for token, token_id in grd_tokens.items(): 
                '''
                Diff: grd_tokens should be ut_tokens. In the selfrag_reproduction.py we fix this problem.
                souce code: https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_long_form_static.py#L68
                '''
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                ut_score_dict[p_idx][token] = np.exp(float(prob))
        if len(ut_score_dict[p_idx]) == 5: 
            ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
            ut_scores = [-1, -0.5, 0, 0.5, 1]
            utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
        else:   
            utility_score = 0.0
        return float(utility_score), ut_score_dict