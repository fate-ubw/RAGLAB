from typing import Optional, Any
import numpy as np
from vllm import  SamplingParams
from tqdm import tqdm

from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag, ModeNotFoundError
from raglab.rag.infer_alg.self_rag_reproduction.utils import load_special_tokens, postprocess_answer_option_conditioned
from raglab.rag.infer_alg.self_rag_reproduction.utils import postprocess, fix_spacing
from raglab.dataset.utils import get_dataset
from raglab.dataset.base_dataset import MultiChoiceQA
from raglab.language_model import BaseLM
import pdb
from pprint import pprint
class SelfRag_Reproduction(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
    
    def init(self, args):
        self.world_size = args.world_size
        self.dtype = args.dtype
        # decoding args
        self.threshold = args.threshold
        self.use_seqscore = args.use_seqscore
        self.use_groundness = args.use_groundness
        self.use_utility = args.use_utility
        self.w_rel = args.w_rel
        self.w_sup = args.w_sup
        self.w_use = args.w_use
        self.beam_width = args.beam_width
        self.max_depth = args.max_depth
        # retrieval 
        self.retrieval_mode = args.retrieval_mode
        self.show_specialtokens = args.show_specialtokens
        self.realtime_retrieval = args.realtime_retrieval
        self.inference_form = args.inference_form
        self.ignore_cont = args.ignore_cont
        self.use_citation = args.use_citation
        assert self.beam_width < self.n_docs, "The beam width should be less than the number of documents."

    def inference(self, query: Optional[str]=None, mode='interact', task=None):
        assert mode in ['interact', 'evaluation']
        if 'interact' == mode:
            input = f"### Instruction:\n{query}\n\n### Response:\n" #TODO 其实这个就是最基本的一个 instruction 
            source_question = query
            pregiven_passages = []
            if 'short_form' == self.inference_form:
                final_prediction, generation_track = self.short_form_infer(input, source_question, pregiven_passages,
                                                            use_seqscore = self.use_seqscore, threshold = self.threshold,
                                                            w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use, mode = mode)
                catation_docs = {0:None}
                return final_prediction, generation_track
            elif 'long_form' == self.inference_form:
                final_prediction, generation_track = self.long_form_infer(input, source_question, pregiven_passages, 
                                                            beam_width=self.beam_width, max_depth=self.max_depth, 
                                                            w_rel=self.w_rel, w_sup=self.w_sup, w_use=self.w_use, 
                                                            use_seqscore=self.use_seqscore,ignore_cont=self.ignore_cont)
                final_prediction_with_citation, catation_docs = self._aggregate_response_with_citation(final_prediction, generation_track, add_citation=self.use_citation)      
                return  final_prediction_with_citation, generation_track
    
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self)
            self.eval_dataset = self.EvalData.load_dataset()
            self.print_fn(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                eval_data = self.EvalData.preprocess(eval_data)
                target_instruction = self.find_algorithm_instruction('selfrag_reproduction-read', self.task)
                input = target_instruction.format_map({'query': eval_data[self.EvalData.InputStruction.question]})
                source_question = eval_data[self.EvalData.InputStruction.question]
                if self.realtime_retrieval == True:
                    pregiven_passages = []
                else:
                    # get the pregiven passages form local eval datasets.
                    pregiven_passages = eval_data[self.EvalData.InputStruction.pregiven_passages][:self.n_docs]
                if 'short_form' == self.inference_form:
                    final_prediction, generation_track = self.short_form_infer(input, source_question, pregiven_passages,
                                                                            use_seqscore = self.use_seqscore, threshold = self.threshold,
                                                                            w_rel = self.w_rel, w_sup = self.w_sup, w_use = self.w_use, mode = mode)

                    inference_results = self.EvalData.record_result(eval_data, final_prediction, inference_results)
                    # calculate metric
                    acc = self.EvalData.eval_acc(inference_results)
                    EM = self.EvalData.eval_exact_match(inference_results)
                    f1_score = self.EvalData.eval_f1_score(inference_results)
                    self.print_fn(f'{self.task} in {idx+1} turn: \n Accuracy: {acc} \n Exact match:{EM} \n F1 score: {f1_score}')
                elif 'long_form' == self.inference_form:
                    final_prediction, generation_track = self.long_form_infer(input, source_question, pregiven_passages, 
                                                                beam_width=self.beam_width, max_depth=self.max_depth, 
                                                                w_rel=self.w_rel, w_sup=self.w_sup, w_use=self.w_use, 
                                                                use_seqscore=self.use_seqscore,ignore_cont=self.ignore_cont)
                    final_prediction_with_citation, catation_docs = self._aggregate_response_with_citation(final_prediction, generation_track, add_citation=self.use_citation)  
                    response_id = 0
                    inference_results = self.EvalData.record_result(eval_data, final_prediction_with_citation, inference_results,
                                                                    catation_docs, response_id, generation_track )
                # 是否要给 ASQA 和 factscore 也计算 acc 什么的
                self.print_fn(f'{self.task} in {idx+1} turn:\n Question:{source_question} \n Rag Output:{final_prediction} \n Answers: {eval_data[self.EvalData.InputStruction.answer]}')
            # --> end of dataset loop
            self.EvalData.save_result(inference_results) 
            if 'short_form' == self.inference_form:
                eval_result = {'Accuracy':acc, 'Exact match': EM, 'F1 score':f1_score}
                self.EvalData.save_evaluation_results(eval_result)
                return eval_result
            elif 'long_form' == self.inference_form:
                return 'Inference completion'
        # --> end of evaluation
        else:
            raise ModeNotFoundError("Mode must be interact or evaluation. Please provide a valid mode.")

    def short_form_infer(self, prompt:str, source_question:str, pregiven_passages:Optional[None],
                            use_seqscore=True, threshold=0.2,w_rel=1.0, w_sup=1.0, w_use=0.5, mode = 'evaluation'): 

        retrieval_tokens, relevant_tokens, ground_tokens, utility_tokens = load_special_tokens(
                                    self.llm.tokenizer, use_grounding=self.use_groundness, use_utility=self.use_utility) 
        generation_track = {}
        if 'always_retrieval' == self.retrieval_mode:
            do_retrieve = True
        elif 'no_retrieval' == self.retrieval_mode:
            do_retrieve = False
        elif 'adaptive_retrieval' == self.retrieval_mode:
            #retrieval or not base on first token
            ratio, generation_track = self._firstToken_retrievalRatio(prompt, retrieval_tokens, generation_track)
            do_retrieve = ratio > threshold
        # do retrieval or not retrieval
        if do_retrieve is True:   
            if self.realtime_retrieval == True:
                passages = self.retrieval.search(source_question)
                passages = self._truncate_passages(passages)
                evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(passage["title"], passage["text"]) for rank, passage in passages.items()] 
            else:
                evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(para["title"], para["text"]) for para in pregiven_passages] 
            outputs_list = self.llm.generate(evidence_augmented_inputs)
            # calculate score of each candidate
            relevance_score_dict = {}
            grd_score_dict = {}
            ut_score_dict = {}
            overall_scores = {}
            for p_idx, Outputs in enumerate(outputs_list): 
                #sequence score 
                seq_score = self._sequence_score(Outputs)
                # init dict in each loop
                relevance_score_dict.setdefault(p_idx, {}) 
                grd_score_dict.setdefault(p_idx, {})
                ut_score_dict.setdefault(p_idx, {})
                # relevance score 
                relevance_score, relevance_score_dict = self._relevanceToken_score(Outputs, relevant_tokens, p_idx, relevance_score_dict)
                # Issupport score
                ground_score, grd_score_dict = self._IssupportToken_score(Outputs, ground_tokens, p_idx, grd_score_dict)
                # Utility score
                utility_score, ut_score_dict = self._UtilityToken_score(Outputs, utility_tokens, p_idx, ut_score_dict)
                if use_seqscore is True:
                    final_score = seq_score + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
                else:
                    final_score = w_rel * relevance_score +  w_sup * ground_score + w_use * utility_score
                overall_scores[p_idx] = {"final_score": final_score,
                                        "relevance_score": relevance_score,
                                        "ground_score": ground_score,
                                        "utility_score": utility_score,
                                        "relevance_score_dict": relevance_score_dict, 
                                        "grd_score_dict": grd_score_dict,
                                        "ut_score_dict": ut_score_dict} #TODO Consider the necessity of removing this code segment.
                pred_text = Outputs.text
                if self.realtime_retrieval == True:
                    generation_track["retrieval_{}".format(p_idx+1)] = {"pred": pred_text, "score": float(final_score), "ctx": passages[p_idx+1]}
                else:
                    generation_track["retrieval_{}".format(p_idx+1)] = {"pred": pred_text, "score": float(final_score), "ctx": pregiven_passages[p_idx]}
            # --> end of for loop
        # --> end of do retrieve 
        else: 
            # no retrieval generation
            prompt += "[No Retrieval]"
            outputs_list = self.llm.generate([prompt])
            pred_text = outputs_list[0].text 
            generation_track['no_retrieval'] = {"pred": pred_text} # no retrieval no need score and passages
        
        # Aggregating answers
        if len(generation_track) <= 2: 
            # post process for no retrieval
            if True == self.show_specialtokens:
                return pred_text, generation_track
            else:
                # remove all sprcial tokens 
                postprocessed_pred = postprocess_answer_option_conditioned(pred_text) 
                return postprocessed_pred, generation_track 
        else:
            # post for do retrieval 
            answer2score = {}
            if 'evaluation' == mode and isinstance(self.EvalData, MultiChoiceQA) == True:
                '''
                Aggregating for multi-choice question
                source explaination: For SELF-RAG inference on PubHealth and ARC-C, instead of determining the output with the highest score as in other tasks, 
                                    we aggregate the scores for each option and select the answer option with the highest score.       
                paper: https://arxiv.org/abs/2310.11511
                '''
                for key, result in generation_track.items():
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
                path2score = {key: item["score"] for key, item in generation_track.items() if key != "decide_retrieval_mode"} 
                best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
                best_option = generation_track[best_path]["pred"]
                if self.show_specialtokens == True:
                    pass
                else:
                    # remove all special token 
                    best_option = postprocess_answer_option_conditioned(best_option)
        # --> end of do retrieval postprocess
        return best_option, generation_track 

    def long_form_infer(self, prompt: str, source_question: str, pregiven_passages:Optional[dict],
                             beam_width=2, max_depth=7,w_rel=1.0, w_sup=1.0, w_use=0.5, 
                             use_seqscore = True,ignore_cont = None) -> tuple[dict[int,str], dict, bool]: 

        retrieval_tokens, relevant_tokens, ground_tokens, utility_tokens = load_special_tokens(self.llm.tokenizer, 
                                                                            use_grounding=self.use_groundness, 
                                                                             use_utility=self.use_utility)

        if 'no_retrieval' == self.retrieval_mode:
            prompt += "[No Retrieval]"
            outputs_list = self.llm.generate([prompt])
            preds_text = [Outputs.text.split("\n\n")[0] for Outputs in outputs_list]
            final_prediction = {0:preds_text[0]} 
            generation_track = {"original_splitted_sentences": {0:preds_text}}
            return final_prediction, generation_track
        elif 'always_retrieval' == self.retrieval_mode:
            '''
            Diff: The logic of adaptive retrieval is based on paper and GitHub issue, which is different from the self RAG source code (run_long_form_static.py). 
            Raglab has truly implemented the multi-turn retrieval proposed in the self rag paper for the first time.
            '''
            curr_depth = 1
            node_id = 0
            prediction_tree = {}
            levels = {}
            prediction_tree[node_id] = {"prompt": prompt, "pred": "", 
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None} # [First retrieve flag] means 
            levels[0] = [0]
            while curr_depth < max_depth:
                # bulid tree
                levels[curr_depth]= []
                if curr_depth - 1 in levels:
                    for parent_node in levels[curr_depth-1]:
                        prev_pred, prompt, prev_generation, prev_score = self._get_lastTurn_generation(parent_node, prediction_tree)
                        curr_prompt = prompt + prev_generation
                        # self rag input maintain special token
                        previous_sentence = postprocess(prev_pred) 
                        current_retrieval_input = source_question + previous_sentence
                        '''
                        This is implemented according to the method described in the self-rag paper. For each retrieval, 
                        the input is the source question + the previously generated sentence, and in theory, this sentence should be the one with special tokens removed. 
                        This way, the retrieval process can be more accurate during subsequent iterations.                        
                        '''
                        curr_preds, curr_scores, overall_score_dict, retrieval_docs = self._run_step_generation_batch(curr_prompt, current_retrieval_input , pregiven_passages,
                                                                                                                     retrieval_tokens=retrieval_tokens, relevant_tokens=relevant_tokens,
                                                                                                                     ground_tokens=ground_tokens, utility_tokens=utility_tokens,
                                                                                                                     w_rel=w_rel, w_sup=w_sup, w_use=w_use, use_seqscore=use_seqscore)
                        
                        prediction_tree, node_id, levels = self._set_predictionTree(curr_depth, parent_node, node_id, 
                                                                           curr_preds, curr_scores, curr_prompt,
                                                                           prev_score, retrieval_docs, prediction_tree, levels ,overall_score_dict)
                    # end of the for loop 
                    current_rank = levels[curr_depth]
                    #get the top-2 score 
                    node2score = {node_id: prediction_tree[node_id]['score'] for node_id in current_rank} 
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[:beam_width] # get top2 results
                    levels[curr_depth] = [node[0] for node in top_nodes] 
                    curr_depth += 1  
                else:
                    break
            # --> end of the while curr_depth < max_depth:
            best_selections = self._backtracking_prediction_tree(levels, curr_depth, prediction_tree)
            # get final_prediction
            final_prediction = {}
            splitted_sentences = {}
            original_splitted_sentences = {}
            ctxs = {}
            for path_i, nodes in best_selections.items():
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
        
            return final_prediction, generation_track
        # --> end of  'always_retrieval'
        elif 'adaptive_retrieval' == self.retrieval_mode:
            '''
            diff: The logic of adaptive retrieval is based on paper and GitHub issue, which is different from the self RAG source code (run_long_form_static.py). 
            Raglab has truly implemented the multi-turn retrieval proposed in the self rag paper for the first time.
            '''
            curr_depth = 1
            node_id = 0
            prediction_tree = {}
            levels = {}
            prediction_tree[node_id] = {"prompt": prompt, "pred": "", 
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None} # [First retrieve flag] means 
            levels[0] = [0]
            level_tmp = [] # level_tmp is used to store the node when [No retrieval], and then after the entire tree is maintained, all nodes in level_tmp are merged into the tree. 
            while curr_depth < max_depth:
                # bulid tree
                if curr_depth - 1 in levels and len(levels[curr_depth-1])!=0:
                    levels[curr_depth]= []
                    for parent_node in levels[curr_depth-1]:
                        prev_pred, prompt, prev_generation, prev_score = self._get_lastTurn_generation(parent_node, prediction_tree)
                        '''
                        This is implemented according to the method described in the self-rag paper. For each retrieval, 
                        the input is the source question + the previously generated sentence, and in theory, this sentence should be the one with special tokens removed. 
                        This way, the retrieval process can be more accurate during subsequent iterations.                        
                        '''
                        curr_prompt = prompt + prev_generation
                        previous_sentence = postprocess(prev_pred) 
                        current_retrieval_input = source_question + previous_sentence
                        '''
                        '''
                        ratio, _ = self._firstToken_retrievalRatio(curr_prompt, retrieval_tokens, None)
                        if ratio > self.threshold:
                            curr_preds, curr_scores, overall_score_dict, retrieval_docs = self._run_step_generation_batch(curr_prompt, current_retrieval_input , pregiven_passages,
                                                                                                        retrieval_tokens=retrieval_tokens, relevant_tokens=relevant_tokens, 
                                                                                                        ground_tokens=ground_tokens, utility_tokens=utility_tokens,
                                                                                                        w_rel=w_rel, w_sup=w_sup, w_use=w_use, use_seqscore=use_seqscore)
                            prediction_tree, node_id, levels = self._set_predictionTree(curr_depth, parent_node, node_id,curr_preds, 
                                                                                       curr_scores, curr_prompt,prev_score, retrieval_docs, 
                                                                                       prediction_tree, levels ,overall_score_dict)
                        else:
                            curr_preds, curr_scores, overall_score_dict, retrieval_docs = self._generation_without_retrieval(curr_prompt)
                            prediction_tree, node_id, level_tmp = self._set_predictionTree_NoRetrieval(curr_depth, parent_node, node_id, curr_preds, 
                                                                                                    curr_scores, curr_prompt, retrieval_docs, 
                                                                                                    prediction_tree, overall_score_dict, level_tmp)

                    # --> end of the levels loop 
                    current_rank = levels[curr_depth] 
                    #get the top-k node based on sentence final score
                    node2score = {node_id: prediction_tree[node_id]['score'] for node_id in current_rank} #
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True) # 
                    top_nodes = top_nodes[:(beam_width - len(level_tmp))] 
                    levels[curr_depth] = [node[0] for node in top_nodes] 
                    curr_depth += 1
                # --> end of Depth-First Search
                else:
                    break
            # --> end of the while curr_depth < max_depth:
            # Complete the tree(variable:levels)
                # The purpose of below snippet code is only to complete the logic of building tree(variable:levels), and it is not helpful for building the best response
            for no_retrieval_node in level_tmp:
                depth = no_retrieval_node['curr_depth']
                node_id = no_retrieval_node['node_id']
                levels[depth] = levels[depth] + [node_id]
            # {0: [0], 1: [1, 3], 2: [9, 11], 3: [15], 4: [19], 5: [23], 6: [28]}
            # backtraking the levels get the best answer
            best_selections = self._backtracking_prediction_tree(levels, curr_depth, prediction_tree)
            if len(best_selections) < self.beam_width:
                # In this situation get the last path_id in best_selections
                for path_id, best_selection in best_selections.items():
                    path_id = path_id
                best_selections = self._backtracking_prediction_tree_noRetrieval(best_selections, prediction_tree, level_tmp, path_id)
            # get final_prediction
            final_prediction = {}
            splitted_sentences = {}
            original_splitted_sentences = {}
            ctxs = {}
            for path_i, nodes in best_selections.items():
                final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
                splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
                original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                    ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

                ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
                    ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            # --> end of postprocess
            generation_track = {"final_prediction": final_prediction,
                    "splitted_sentences": splitted_sentences,
                    "original_splitted_sentences": original_splitted_sentences,
                    "best_selections": best_selections,
                    "ctxs": ctxs,
                    "prediction_tree": prediction_tree}
        # --> end of adaptive retrieval
            return final_prediction, generation_track
        else:
            raise InvalidRetrievalModeError('Invalid retrieval_mode. Self rag only havs three mode: no_retrieval, always_retrieval, adaptive_retrieval mode')

    def _generation_without_retrieval(self, prompt):
        '''
        # without retrieval and retruen one response
        '''
        prompt += "[No Retrieval]"
        outputs_list = self.llm.generate([prompt])
        curr_prediction = [Outputs.text.split("\n\n")[0] for Outputs in outputs_list]
        scores = [1] # The score of [No retrieval] output is 1. And the [No retrieval] outputs will not be sorted by score in rank process
        overall_scores = {0:None}
        retrieval_docs = {1:None}
        return curr_prediction, scores, overall_scores, retrieval_docs
    
    def _aggregate_response_with_citation(self, final_predictions: dict[int,str], generation_track: dict[str, Any], add_citation = True)-> tuple[dict, dict]:
        '''
        # Aggregate response for response. If the response generate by no_retrieval mode. There is no need to add citation. 
        '''
        output_with_citation = {}
        catation_doc = {}
        for response_idx, generated_response in final_predictions.items(): 
            final_output = ""
            docs = []
            previous_generations = [] 
            if "splitted_sentences" not in generation_track:
                output_with_citation[response_idx] = fix_spacing(postprocess(generated_response))
                catation_doc[response_idx] = docs
            else:
                if len(postprocess(generated_response)) == 0:
                    generation_track["splitted_sentences"][response_idx], generation_track["ctxs"][response_idx] = generation_track["splitted_sentences"][response_idx], generation_track["ctxs"][response_idx] 
                for cite_idx, (sentence, doc) in enumerate(iterable=zip(generation_track["splitted_sentences"][response_idx], generation_track["ctxs"][response_idx])):
                    if len(sentence) == 0:
                        continue
                    sentence = postprocess(sentence) 
                    # remove the loopping sentence
                    if sentence in previous_generations: 
                        continue
                    else:
                        previous_generations.append(sentence)

                    if add_citation == True and doc is not None: 
                        sentence = sentence.replace(".[Continue to Use Evidence]", f" [{cite_idx}]. ")
                        final_output += sentence[:-1] + " [{}]".format(cite_idx) + ". " 
                        final_output = final_output.replace(f". [{cite_idx}] ", f" [{cite_idx}]. ")
                        docs.append(doc) # docs -> list[dict]
                        '''
                        # Diff: selfrag_reproduction.py implements the citation function under multiple rounds of retrieval, which is different from the logic of selfrag_orignal.py                        
                        '''
                    else:
                        final_output += sentence
                        # [No retrieval] do not cite doc
                if len(final_output) == 0:
                    final_output = fix_spacing(final_output)  
                if len(final_output) > 0 and final_output[-1] == " ":
                    final_output = final_output[:-1]
                final_output = fix_spacing(final_output)
                output_with_citation[response_idx] = final_output
                catation_doc[response_idx] = docs
        # --> end of the for loop
        return output_with_citation, catation_doc

    def _backtracking_prediction_tree(self, levels: dict[int,list[int]], curr_depth: int, prediction_tree: dict[int, dict]) -> dict[int,list[int]]:
        '''
        get best tracking from prediction_tree base on levels
        '''
        parent = 0 
        best_selections = {}
        # Traverse from the bottom 
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0} # remove empty list in levels
        for path_i, node in enumerate(levels[len(levels)]): # beam search 
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
                current_level -= 1
        return best_selections

    def _backtracking_prediction_tree_noRetrieval(self, best_selections:dict[int,list], prediction_tree, level_tmp:list[dict], next_path_id:int):
        '''
        # back tracking the node in level_tmp 
        - level_tmp is generated by [No retrieval] inference process
        - this function is only used in adaptive retrieval in long form inference
        '''
        for no_retrieval_node in level_tmp:
            curr_depth = no_retrieval_node['curr_depth']
            node = no_retrieval_node['node_id']
            next_path_id += 1
            best_selections[next_path_id] = [node]
            current_node = node 
            current_level = curr_depth 
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]['parent']
                best_selections[next_path_id] = [parent] + best_selections[next_path_id]
                current_node = parent
                current_level -= 1
        return best_selections
    

    def _set_predictionTree(self, curr_depth, parent_node, node_id:int,  curr_preds:list[str], curr_scores:list[float], curr_prompt:str, prev_score, retrieval_docs, prediction_tree, levels , overall_score_dict):
        retrieval_results = {}
        for i, (curr_pred, p_score) in enumerate(zip(curr_preds, curr_scores)): 
            retrieval_results[i] = {"pred": curr_pred, "score": p_score}
        for i, result in retrieval_results.items(): 
            node_id += 1 
            node_score = result["score"] * prev_score if prev_score is not None else result["score"]
            curr_pred = result["pred"] 
            if self.realtime_retrieval == True:
                # the index of real time retrieved passages begin from 1, but the index of pre-given passages begin from 0.
                prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred,
                                            "score": node_score, "ctx": retrieval_docs[i+1], "parent": parent_node,
                                            "overall_score_dict": overall_score_dict} # TODO access the usage of overall_score_dict
            else:
                prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred,
                                            "score": node_score, "ctx": retrieval_docs[i], "parent": parent_node,
                                            "overall_score_dict": overall_score_dict}
            # Meet:
            if "[Retrieval]" in curr_pred:
                gen_result_index = curr_pred.index("[Retrieval]")
                prev_generation = curr_pred[:gen_result_index]
            else:
                prev_generation = curr_pred
            '''
            Diff: check wrong pattern and cutting the wrong pattern in curr_pred.
            '''
            prediction_tree[node_id]["processed_pred"] = prev_generation 
            levels[curr_depth].append(node_id)
        # --> end of set prediction_tree loop
        return prediction_tree, node_id, levels
    
    def _set_predictionTree_NoRetrieval(self, curr_depth, parent_node, node_id, curr_pred, curr_score, curr_prompt, retrieval_docs, prediction_tree, overall_score_dict, level_tmp):
        curr_pred = curr_pred[0]
        node_id += 1
        if self.realtime_retrieval == True:
            prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred, 
                                        "score": curr_score[0], "ctx": retrieval_docs[1], "parent": parent_node,
                                        "overall_score_dict": overall_score_dict} 
        else:
            prediction_tree[node_id] = {"prompt": curr_prompt, "pred": curr_pred, 
                                        "score": curr_score[0], "ctx": retrieval_docs[0], "parent": parent_node,
                                        "overall_score_dict": overall_score_dict} 
        # Meet:
        if "[Retrieval]" in curr_pred:
            gen_result_index = curr_pred.index("[Retrieval]")
            prev_generation = curr_pred[:gen_result_index]
        else:
            prev_generation = curr_pred
        prediction_tree[node_id]["processed_pred"] = prev_generation
        level_tmp.append({'curr_depth':curr_depth, 'node_id':node_id})
        return prediction_tree, node_id, level_tmp

    def _run_step_generation_batch(self, prompt, current_retrieval_input, pregiven_passages:Optional[list[dict]],
                                  retrieval_tokens=None, relevant_tokens=None, ground_tokens=None,  utility_tokens=None,
                                  w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False) -> tuple[list[str], list[float], dict]:
        if self.realtime_retrieval == True:
            passages = self.retrieval.search(current_retrieval_input)
            passages = self._truncate_passages(passages)
            evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(passage["title"], passage["text"]) for rank, passage in passages.items()] 
        else:
            evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(para["title"], para["text"]) for para in pregiven_passages] 
        outputs_list = self.llm.generate(evidence_augmented_inputs)
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        final_preds = []
        for p_idx, Outputs in enumerate(outputs_list): 
            pred_text = Outputs.text
            # calculate seq score
            seq_score = self._sequence_score(Outputs)
            # init dict in each loop
            relevance_score_dict.setdefault(p_idx, {}) 
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # relevance score
            relevance_score, relevance_score_dict = self._relevanceToken_score(Outputs, relevant_tokens, p_idx, relevance_score_dict)
            # Issupport score
            ground_score, grd_score_dict = self._IssupportToken_score(Outputs, ground_tokens, p_idx, grd_score_dict)
            # Utility score
            utility_score, ut_score_dict = self._UtilityToken_score(Outputs, utility_tokens, p_idx, ut_score_dict) 
            '''
            Diff: selfrag_reproduction.py use self.UtilityToken_score() calculate the correct utility_score, which is different from the logic of selfrag_orignal.py
            '''
            if self.use_seqscore is True:
                final_score = seq_score + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score +  w_sup * ground_score + w_use * utility_score
            overall_scores[p_idx] = {"final_score": final_score} 

            if "[No Retrieval]" in pred_text:
                pred_text = self._modify_NoRetrieval_into_Retrieval(Outputs, retrieval_tokens)
                final_preds.append(pred_text)
            else:
                final_preds.append(pred_text)
        # --> end of the clculate each generation score loop
        preds = final_preds
        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores] 
        if self.realtime_retrieval == True:
            retrieval_docs = passages
        else:
            retrieval_docs = pregiven_passages # pregiven_passages only provide in PopQA
        return preds, scores, overall_scores, retrieval_docs

    def _get_lastTurn_generation(self, parent_node, prediction_tree):
        ''' 
        get previous information from prediction_tree
        '''
        prev_pred = prediction_tree[parent_node]["pred"]
        prev_prompt = prediction_tree[parent_node]["prompt"]
        prev_generation = prediction_tree[parent_node]["processed_pred"]
        prev_generationScore = prediction_tree[parent_node]["score"]
        return prev_pred, prev_prompt, prev_generation, prev_generationScore
        
    def _firstToken_retrievalRatio(self, prompt:str, retrieval_tokens:dict[str,int], generation_track:Optional[dict[str,Any]]) -> tuple[float, dict]:
        '''
        calculate the ratio of retrieval base on first token logits
        '''
        vocab_size = self.llm.tokenizer.vocab_size
        special_token_size = len(self.llm.tokenizer.added_tokens_decoder)
        # remove redundancy special tokens
        real_special_tokens = [{idx:token} for idx,token in self.llm.tokenizer.added_tokens_decoder.items() if idx >= vocab_size]
        special_token_size = len(real_special_tokens)
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, repetition_penalty= 1, logprobs = vocab_size + special_token_size, skip_special_tokens = False)
        '''
        Diff: According to self rag's paper, when calculating the ratio, language model only to predict the next token logits.
              Source code max_tokens is often set to 50, 100 or even 300, which greatly wastes computing resources. 
              Raglab optimizes the process of self-rag inference in selfrag_reproduction.py , improves the speed of reasoning and saves a lot of computing resources
        '''
        if self.llm_mode == 'HF_Model' and self.use_vllm == True:
            outputs_list = self.llm.generate([prompt], sampling_params)
        else:
            outputs_list = self.llm.generate([prompt])
        Outputs = outputs_list[0]
        pred_log_probs = Outputs.logprobs
        score_dict = {}
        for tok, id in retrieval_tokens.items():
            if id not in pred_log_probs[0]:
                score_dict[tok] = -100
            prob = pred_log_probs[0][id] 
            score_dict[tok] = np.exp(prob)
            '''
            Diff: Raglab selfrag_reproduction.py fix the bug of "score_dict[tok] = float(prob)" and calculate the right ratio.
            Th bug is from self rag source code [https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py#L79]
            '''
        if "short_form" == self.inference_form:
            generation_track["decide_retrieval_mode"] = Outputs.text 
        ratio = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])  
        return float(ratio), generation_track

    def _sequence_score(self, pred: BaseLM.Outputs) ->float:
        '''
        average prob of generated sentence
        '''
        score = np.exp(pred.cumulative_logprob) / max(len(pred.tokens_ids), 1)
        return float(score)

    def _relevanceToken_score(self, pred: BaseLM.Outputs, relevant_tokens:dict[str,int], p_idx:int, relevance_score_dict:dict) -> tuple[float, dict]:
        pred_log_probs = pred.logprobs
        for tok, id in relevant_tokens.items(): 
            prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
            relevance_score_dict[p_idx][tok] = np.exp(float(prob))
        # calculate score
        relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (np.sum(list(relevance_score_dict[p_idx].values())))
        return float(relevance_score), relevance_score_dict

    def _IssupportToken_score(self, pred:BaseLM.Outputs, ground_tokens:dict[str,int], p_idx:int, grd_score_dict:dict) -> tuple[float, dict]:
        pred_token_ids = pred.tokens_ids
        pred_log_probs = pred.logprobs
        groundness_token_appear_indices = []
        # get the position of Issupport token
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(ground_tokens.values()):
                groundness_token_appear_indices.append(tok_idx)
                break
        # if pred contains ground_tokens, grd_score_dict will be calculated
        if len(groundness_token_appear_indices) > 0:
            idx = groundness_token_appear_indices[0]
            for token, token_id in ground_tokens.items():
                prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100 
                grd_score_dict[p_idx][token] = np.exp(float(prob))
        # calculate score
        if len(grd_score_dict[p_idx]) == 3: 
            gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
            ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (grd_score_dict[p_idx]["[Partially supported]"] / gt_sum) # 
        else:
            ground_score = 0.0 # "If the sentence is labeled as [isRel], then [Issup] will not appear later, resulting in a ground score of 0."
        return float(ground_score), grd_score_dict

    def _UtilityToken_score(self, pred: BaseLM.Outputs, utility_tokens:dict, p_idx:int, ut_score_dict:dict) -> tuple[float, dict]:
        pred_token_ids = pred.tokens_ids
        pred_log_probs = pred.logprobs
        utility_token_appear_indices = []
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok in list(utility_tokens.values()):
                utility_token_appear_indices.append(tok_idx)
        if len(utility_token_appear_indices) > 0:
            idx = utility_token_appear_indices[0] # position of ut_token [Utility:1-5]
            for token, token_id in utility_tokens.items(): 
                '''
                diff: Raglab fix the bug which in selfrag orignal code.
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

    def _modify_NoRetrieval_into_Retrieval(self, pred:BaseLM.Outputs, retrieval_tokens)-> str:
        '''
        check the ratio of ([Retrieval] + [Continue to Use Evidence])/([Retrieval] + [Continue to Use Evidence] + [No Retrieval] )
        if the ratio > threshold modify [No Retrieval] -> [Retrieval]
        '''
        pred_text = pred.text
        pred_log_probs = pred.logprobs 
        pred_token_ids = pred.tokens_ids
        ret_token_appear_indices = []
        substrings = pred_text.split("[No Retrieval]")
        for tok_idx, tok in enumerate(pred_token_ids):
            if tok == retrieval_tokens["[No Retrieval]"]:
                ret_token_appear_indices.append(tok_idx)
                substrings
        # --> end for loop
        ret_token_score_dict = {}
        retrieval_remap = {}
        for order, idx in enumerate(ret_token_appear_indices):
            ret_token_score_dict.setdefault(order, {})
            for tok, tok_id in retrieval_tokens.items(): 
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

class InvalidRetrievalModeError(Exception):
    pass