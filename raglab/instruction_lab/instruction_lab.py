ALGORITHM_INSTRUCTIONS = [
    {
        "algorithm_name": "-------------------Template-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "Rules of naming:'-' seperate for naming. For example: Algorithm_name-mode-specific_stage",
        "dataset_name": "dataset name",
        "instruction": "Fill in your instruction here"
    },
    {
        "algorithm_name": "-------------------Naive Rag-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "",
        "instruction": "### Instruction:\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "PopQA-posterior_instruction",
        "instruction": "### Instruction:\n Now, based on the passages and your internal knowledge, please answer the question more succinctly and professionally. ### Retrieved Knowledge:\n {passages}\n \n## Input:\n\n{query}\n\n ### Response:\n"
    }, 
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "Arc",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "Arc-posterior_instruction",
        "instruction": "### Instruction:\nNow, based on the passages and your internal knowledge, please answer the question more succinctly and professionally. ### Retrieved Knowledge:\n {passages}\n  Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n  ### Response:\n"
    }, 
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "Arc",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "PubHealth-posterior_instruction",
        "instruction": "### Instruction\nDetermine the statement based on the passages and your internal knowledge. ### Retrieved Knowledge:\n {passages}\n  Is the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n  n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "-------------------Query Rewrite Rag-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "PopQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "TriviaQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "StrategyQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited.### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "HotPotQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "MMLU",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "Arc",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "Arc",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "PubHealth",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "ASQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "Factscore",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "query_rewrite_rag-rewrite",
        "dataset_name": "Feverous",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "algorithm_name": "query_rewrite_rag-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "-------------------Iter-Retgen-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "Arc",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Iterative_rag-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "-------------------Active Rag-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "Arc",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "active_rag-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "-------------------Self Ask-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "",
        "instruction": "Question: When does monsoon season end in the state the area code 575 is located? Are follow up questions needed here: Yes. Follow up: Which state is the area code 575 located in? Intermediate answer: The area code 575 is located in New Mexico. Follow up: When does monsoon season end in New Mexico? Intermediate answer: Monsoon season in New Mexico typically ends in mid-September. So the final answer is: mid-September. \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "PopQA",
        "instruction": "Question: When does monsoon season end in the state the area code 575 is located? Are follow up questions needed here: Yes. Follow up: Which state is the area code 575 located in? Intermediate answer: The area code 575 is located in New Mexico. Follow up: When does monsoon season end in New Mexico? Intermediate answer: Monsoon season in New Mexico typically ends in mid-September. So the final answer is: mid-September. \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "TriviaQA",
        "instruction": "Question: When does monsoon season end in the state the area code 575 is located? Are follow up questions needed here: Yes. Follow up: Which state is the area code 575 located in? Intermediate answer: The area code 575 is located in New Mexico. Follow up: When does monsoon season end in New Mexico? Intermediate answer: Monsoon season in New Mexico typically ends in mid-September. So the final answer is: mid-September. \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "StrategyQA",
        "instruction": "Question: When does monsoon season end in the state the area code 575 is located? Are follow up questions needed here: Yes. Follow up: Which state is the area code 575 located in? Intermediate answer: The area code 575 is located in New Mexico. Follow up: When does monsoon season end in New Mexico? Intermediate answer: Monsoon season in New Mexico typically ends in mid-September. So the final answer is: mid-September. \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "HotPotQA",
        "instruction": "Question: When does monsoon season end in the state the area code 575 is located? Are follow up questions needed here: Yes. Follow up: Which state is the area code 575 located in? Intermediate answer: The area code 575 is located in New Mexico. Follow up: When does monsoon season end in New Mexico? Intermediate answer: Monsoon season in New Mexico typically ends in mid-September. So the final answer is: mid-September. \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "Question: When does monsoon season end in the state the area code 575 is located? Are follow up questions needed here: Yes. Follow up: Which state is the area code 575 located in? Intermediate answer: The area code 575 is located in New Mexico. Follow up: When does monsoon season end in New Mexico? Intermediate answer: Monsoon season in New Mexico typically ends in mid-September. So the final answer is: mid-September. \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "MMLU",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "Arc",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "Arc",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "PubHealth",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "ASQA",
        "instruction": "Question: When does monsoon season end in the state the area code 575 is located? Are follow up questions needed here: Yes. Follow up: Which state is the area code 575 located in? Intermediate answer: The area code 575 is located in New Mexico. Follow up: When does monsoon season end in New Mexico? Intermediate answer: Monsoon season in New Mexico typically ends in mid-September. So the final answer is: mid-September. \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "Factscore",
        "instruction": "Question: When does monsoon season end in the state the area code 575 is located? Are follow up questions needed here: Yes. Follow up: Which state is the area code 575 located in? Intermediate answer: The area code 575 is located in New Mexico. Follow up: When does monsoon season end in New Mexico? Intermediate answer: Monsoon season in New Mexico typically ends in mid-September. So the final answer is: mid-September. \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "self_ask-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "self_ask-followup_question",
        "dataset_name": "Feverous",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "algorithm_name": "-------------------Self Rag Reproduction-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },

    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"

    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\nYou are only allowed to answer True or False, and generating other types of responses is prohibited.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "Arc",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "selfrag_reproduction-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "-------------------Selfrag Critic model train data collector api model instructions-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "collector-selfrag-[Retrieve]",
        "dataset_name": None,
        "instruction":
        (
            "Given an instruction, please make a judgment on whether finding some external documents from the web (e.g., Wikipedia) helps to generate a better response. Please answer [Yes] or [No] and write an explanation.\n\n"
            "##\nInstruction: Give three tips for staying healthy.\n"
            "Need retrieval?: [Yes]\n"
            "Explanation: There might be some online sources listing three tips for staying healthy or some reliable sources to explain the effects of different behaviors on health. So retrieving documents is helpful to improve the response to this query.\n\n"
            "##\nInstruction: Describe a time when you had to make a difficult decision.\n"
            "Need retrieval?: [No]\n"
            "Explanation: This instruction is asking about some personal experience and thus it does not require one to find some external documents.\n\n"
            "##\nInstruction: Write a short story in third person narration about a protagonist who has to make an important career decision.\n"
            "Need retrieval?: [No]\n"
            "Explanation: This instruction asks us to write a short story, which does not require external evidence to verify.\n\n"
            "##\nInstruction: What is the capital of France?\n"
            "Need retrieval?: [Yes]\n"
            "Explanation: While the instruction simply asks us to answer the capital of France, which is a widely known fact, retrieving web documents for this question can still help.\n\n"
            "##\n Instruction: Find the area of a circle given its radius. Radius = 4\n"
            "Need retrieval?: [No]\n"
            "Explanation: This is a math question and although we may be able to find some documents describing a formula, it is unlikely to find a document exactly mentioning the answer.\n\n"
            "##\nInstruction: Arrange the words in the given sentence to form a grammatically correct sentence. quickly the brown fox jumped\n"
            "Need retrieval?: [No]\n"
            "Explanation: This task doesn't require any external evidence, as it is a simple grammatical question.\n\n"
            "##\nInstruction: Explain the process of cellular respiration in plants."
            "Need retrieval?: [Yes]\n"
            "Explanation: This instruction asks for a detailed description of a scientific concept, and is highly likely that we can find a reliable and useful document to support the response.\n\n"
            "##\nInstruction:{instruction}\n"
            "Need retrieval?: "
        )
    },
    {
        "algorithm_name": "collector-selfrag-[IsRel]",
        "dataset_name": None,
        "instruction": 
        (
            "You'll be provided with an instruction, along with evidence and possibly some preceding sentences. "
            "When there are preceding sentences, your focus should be on the sentence that comes after them. "
            "Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
            "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
            "###\nInstruction: Given four answer options, A, B, C, and D, choose the best answer.\n\n"
            "Input: Earth rotating causes\n"
            "A: the cycling of AM and PM\nB: the creation of volcanic eruptions\nC: the cycling of the tides\nD: the creation of gravity\n\n"
            "Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.\n\n"
            "Rating: [Relevant]\n"
            "Explanation: The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A.\n\n"
            "###\nInstruction: age to run for us house of representatives\n\n"
            "Evidence: The Constitution sets three qualifications for service in the U.S. Senate: age (at least thirty years of age); U.S. citizenship (at least nine years); and residency in the state a senator represents at the time of election.\n\n"
            "Rating: [Irrelevant]\n"
            "Explanation: The evidence only discusses the ages to run for the US Senate, not for the House of Representatives.\n\n"
            "###\nInstruction: {instruction}\n\n"
            "Evidence: {evidence}\n\n"
            "Rating:"
        )
    },
    {
        "algorithm_name": "collector-selfrag-[IsSup]",
        "dataset_name": None,
        "instruction":
        (
            "You will receive an instruction, evidence, and output, and optional preceding sentences.  If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement.\n"
            "Use the following entailment scale to generate a score:\n"
            "[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.\n"
            "[Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].\n"
            "[No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
            "Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n\n"
            "###\nInstruction: Explain the use of word embeddings in Natural Language Processing.\n"
            "Preceding sentences: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.\n"
            "Output: Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.\n"
            "Evidence: Word embedding\nWord embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing, sentiment analysis, next token predictions as well as analogy detection.\n"
            "Score: [Fully supported]\n"
            "Explanation: The output sentence discusses the application of word embeddings, and the evidence mentions all of the applications syntactic parsing, sentiment analysis, next token predictions as well as analogy detection as the applications. Therefore, the score should be [Fully supported].\n\n"
            "###\n"
            "Instruction: {instruction}\n"
            "Output: {target_output}\n"
            "Evidence: {evidence}\n"
            "Score: "
        )
    },
    {
        "algorithm_name": "collector-selfrag-[Utility]-source",
        "dataset_name": None,
        "instruction": 
        (
            "Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n\n"
            "The detailed criterion is as follows:\n"
            "5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
            "4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
            "3: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
            "2: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
            "1: The response is barely on-topic or completely irrelevant.\n"
            "##\n Instruction: Who is the current prime minister of the UK as of 2023?\n"
            "Output: Boris Johnson was the prime minister of the UK from 2019 - 2022.\n"
            "Perceived utility: 2\n"
            "Explanation: While the output provides a factually correct statement about the UK prime minister from 2019 to 2022, this instruction asks who the prime minister is as of 2023, so it doesn't answer the instruction. Therefore, the utility is 2.\n\n"
            "##\nInstruction: Given a description of a travel destination, recommend 10 tourist attractions with detailed explanations of each. The travel destination is Tokyo, Japan.\n"
            "Output: 'Tokyo is a vibrant city full of exciting tourist attractions. Some must-see sights include the Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen.\n"
            "Perceived utility: 3\n"
            "Explanation: This output doesn't provide descriptions of each attraction and the number of the attractions is also less than 10. While this output partially answers the instructions, it doesn't match the instructions strictly. \n\n"
            "##\nInstruction: {instruction}\n"
            "Output:{output}\n"
        )
    },
    {
        "algorithm_name": "collector-selfrag-[Utility]-imporve",
        "dataset_name": None,
        "instruction": 
        (
            "Given an instruction and an output, rate whether the response appears to be a helpful and correct answer to the query, from 1 (lowest) - 2 (highest). We call this score perceived utility.\n\n"
            "The detailed criterion is as follows:\n"
            "2: The response provides a correct, complete, helpful, and informative response to the query, fully satisfying the information needs. Even if some answers are short, as long as they are correct and complete, they still meet the requirement for a score of 2.\n"
            "1: The response is wrong, not complete or completely irrelevant.\n"
            "##\n Instruction: Cadmium Chloride is slightly soluble in this chemical, it is also called what?\n"
            "Output: Ink\n"
            "Perceived utility: 1\n"
            "Explanation: Cadmium chloride is slightly soluble in water, not ink. When dissolved in water, it dissociates into cadmium ions (Cd²⁺) and chloride ions (Cl⁻). This solubility means that only a small amount of cadmium chloride can dissolve in water, forming a clear solution."
            "##\n Instruction: Who is the current prime minister of the UK as of 2023?\n"
            "Output: Boris Johnson was the prime minister of the UK from 2019 - 2022.\n"
            "Perceived utility: 1\n"
            "Explanation: While the output provides a factually correct statement about the UK prime minister from 2019 to 2022, this instruction asks who the prime minister is as of 2023, so it doesn't answer the instruction. Therefore, the utility is 2.\n\n"
            "##\nInstruction: Given a description of a travel destination, recommend 10 tourist attractions with detailed explanations of each. The travel destination is Tokyo, Japan.\n"
            "Output: 'Tokyo is a vibrant city full of exciting tourist attractions. Some must-see sights include the Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen.\n"
            "Perceived utility: 1\n"
            "Explanation: This output doesn't provide descriptions of each attraction and the number of the attractions is also less than 10. While this output partially answers the instructions, it doesn't match the instructions strictly. \n\n"
            "##\nInstruction: What is George Rankin's occupation?"
            "output: politician"
            "Perceived utility: 2\n"
            "##\nInstruction: What is George Rankin's occupation?"
            "Explanation: The response correctly identifies George Rankin's occupation as a politician, which matches the given information. Therefore, the answer is accurate and fulfills the requirement of the instruction."
            "output: George Rankin was an Australian soldier, farmer, and politician. He served as a member of the Australian House of Representatives from 1937 to 1949 and as a senator from 1950 to 1956. "
            "Perceived utility: 2\n"
            "Explanation: The response correctly identifies George Rankin's occupation as a politician, which matches the given information. Therefore, the answer is accurate and fulfills the requirement of the instruction."
            "##\nInstruction: {instruction}\n"
            "Output:{output}\n"
        )
    },
    {   
        "algorithm_name": "collector-incorrect_sample",
        "dataset_name": None,
        "instruction": 
        (
            "Please generate an incorrect answer based on the instructions and answers provided. Directly generate the answer without producing any additional content."
            "Task instruction: {question}\n"
            "Answer: {answer}\n"
            "Answer:"
        )
    },
    {
        "algorithm_name": "collector-Most_relevantest_passages",
        "dataset_name": None,
        "instruction": 
        (
            "From now on, you are an expert in text relevance evaluation. I will provide you with an original instruction and relevant evidence." 
            "Please find all the passages in the evidence that are related to the instruction, and identify the passages with the highest relevance to the instruction." 
            "The passages are numbered [1], [2], [3], etc. Please directly return a passage number.No need to generate the content of the passages and and do not generate any other content."
            "Task instruction: {instruction}\n"
            "Evidences: {evidences}\n"
        )
    },
    {
        "algorithm_name": "collector-candidate_answers",
        "dataset_name": None,
        "instruction": 
        (
            "Question: Instruction for Generating Answers of Different Quality"
            "Context:"
            "You are an advanced language model trained to understand and generate human-like text. Your task is to generate answers to a given question with varying levels of quality. The quality of an answer can depend on several factors such as accuracy, completeness, coherence, relevance, and language clarity."
            "Task:"
            "For each question provided, generate four answers: very high quality, high quality, low quality, and very low quality. Follow the specific guidelines below for each type of answer."
            "Very High Quality Answer Guidelines:"
            "Accuracy: Ensure all information provided is correct and factually accurate."
            "Completeness: Cover all relevant aspects of the question in detail."
            "Coherence: Maintain a clear, logical flow and well-organized structure."
            "Relevance: Directly address the question with highly relevant information."
            "Language Clarity: Use precise, articulate, and grammatically perfect language."
            "High Quality Answer Guidelines:"
            "Accuracy: Ensure the information provided is correct and mostly factually accurate."
            "Completeness: Cover most relevant aspects of the question comprehensively."
            "Coherence: Maintain logical flow and structure in your response."
            "Relevance: Address the question directly and stay on topic throughout."
            "Language Clarity: Use clear, precise, and grammatically correct language."
            "Low Quality Answer Guidelines:"
            "Accuracy: Introduce some factual inaccuracies or minor errors."
            "Completeness: Provide an incomplete response that misses some key aspects of the question."
            "Coherence: Include some logical inconsistencies or disjointed statements."
            "Relevance: Partially address the question or include some irrelevant information."
            "Language Clarity: Use ambiguous, vague, or somewhat grammatically incorrect language."
            "Very Low Quality Answer Guidelines:"
            "Accuracy: Include significant factual inaccuracies or misleading information."
            "Completeness: Give a very incomplete response, missing most key aspects."
            "Coherence: Lack logical flow and structure, with significant disjointed statements."
            "Relevance: Largely irrelevant information with minimal direct address of the question."
            "Language Clarity: Use unclear, imprecise, and grammatically incorrect language."
            "Please output 4 answers directly, and prohibit the generation of irrelevant content. The format for generating answers should be: Sequence number + answer, for example [1] + answer1, [2] + answer2, [3] + answer3, [4] + answer4. "
            "Golden answer is the Very High Quality Answe. Please generate other candidate answers base on golden answer. Generate 4 candidate answers directly, no need other analysis.\n"
            "Task instruction: {instruction}\n"
            "Evidences: {evidences}\n"
            "Golden answer: {answer}\n"
        )
    },
    {
        "algorithm_name": "collector-pair_wise",
        "dataset_name": None,
        "instruction": 
        (
            "Given two responses to a task defined by an instruction and input, evaluate which response is better and provide a reference answer. Your evaluation should include:"
            "1. A comparative assessment of the two responses"
            "2. An evaluation result (1, 2, or Tie)"
            "3. A brief explanation for your evaluation"
            "4. A high-quality reference answer for the task"
            "Your goal is to help create labeled training data for improving language models. Focus on accuracy, relevance, and coherence in your evaluation and reference answer."
            "Here is an example:"
            "### Instruction: Find an example of the given kind of data. Qualitative data"
            "### Response 1: An example of qualitative data is customer feedback."
            "### Response 2: An example of qualitative data is a customer review."
            "### Evaluation: Tie.\n"
            "### Explation: Both responses are correct and provide similar examples of qualitative data.\n"
            "### Reference: An example of qualitative data is an interview transcript."
            "Task"
            "### Instruction: {instruction}"
            "### Response 1: {response_1}\n"
            "### Response 2: {response_2}\n"
            "Please strictly follow the requirements for generation, including only: ### Evaluation: [result]\n### Explanation: [reason]\n### Reference: [reference]."
        )
    },
    {
        "algorithm_name": "critic-retrieval_instruction",
        "dataset_name": None,
        "instruction": 
        (
            "When provided with instruction, please evaluate whether seeking additional information from external sources such as the web (e.g., Wikipedia) aids in producing a more comprehensive response. Respond with either [Retrieval] or [No Retrieval]. "
        )
    },
    {
        "algorithm_name": "critic-retrieval_input",
        "dataset_name": None,
        "instruction": (
            "Task instruction: {instruction} "
        )
    },
    {
        "algorithm_name": "critic-relevance_instruction",
        "dataset_name": None,
        "instruction": (
            "When given instruction and evidence, evaluate whether the evidence is relevant to the instruction and provides valuable information for generating meaningful responses.\n"
            "Use a rating of [Relevant] to indicate relevance and usefulness, and [Irrelevant] to indicate irrelevance."
        )
    },
    {
        "algorithm_name": "critic-relevance_input",
        "dataset_name": None,
        "instruction": (
            "Task instruction: {instruction}\n"
            "Evidence: {evidence}"
        )
    },
    {
        "algorithm_name": "critic-ground_instruction",
        "dataset_name": None,
        "instruction": (
            "You will receive an instruction, evidence, and output, and optional preceding sentences.  If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement\n"
            "Use the following entailment scale to generate a score:\n"
            "[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.\n"
            "[Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].\n" 
            "[No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
            "Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n\n"
        )
    },
    {
        "algorithm_name": "critic-ground_input",
        "dataset_name": None,
        "instruction": (
            "##\nTask instruction: {instruction}\n"
            "Evidence: {evidence}\n"
            "Output: {output} "
        )
    },
    {
        "algorithm_name": "critic-utility_instruction-original",
        "dataset_name": None,
        "instruction": (
            "Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n"
            "[Utility:5]: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
            "[Utility:4]: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
            "[Utility:3]: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
            "[Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
            "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
        )
    },
    {
        "algorithm_name": "critic-utility_instruction",
        "dataset_name": None,
        "instruction": (
            "Given an instruction and an output, rate whether the response appears to be a helpful and correct answer to the query, from [Utility:1] (lowest) - [Utility:2] (highest). We call this score perceived utility.\n\n"
            "The detailed criterion is as follows:\n"
            "[Utility:2]: The response provides a correct, complete, helpful, and informative response to the query, fully satisfying the information needs. Even if some answers are short, as long as they are correct and complete, they still meet the requirement for a score of 2.\n"
            "[Utility:1]: The response is wrong, not complete or completely irrelevant.\n"
        )
    },
    {
        "algorithm_name": "critic-utility_input",
        "dataset_name": None,
        "instruction": (
            "Task instruction: {instruction}\n"
            "Output: {output} "
        )
    },
    {
        "algorithm_name": "critic-Infer_improvement_answer",
        "dataset_name": None,
        "instruction": (
            "Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {evidences} \n\n### Input:\n\n{instruction}\n\n"
        )
    },
    {
        "algorithm_name": "critic-pair_wise-instruction",
        "dataset_name": None,
        "instruction": (
            "Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.\n\n"
            "### Instruction: {instruction}"
            "### Response 1: {response_1}\n"
            "### Response 2: {response_2}\n"
        )
    },
    {
        "algorithm_name": "critic-pair_wise-output",
        "dataset_name": None,
        "instruction": (
            "### Evaluation: {Eval_result}\n"
            "### Reason: {explanation}\n"
            "### Reference: {Reference}"
        )
    },
    {
        "algorithm_name": "-------------------critic model inference instruction-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "base_model_instruction",
        "dataset_name": None,
        "instruction": "Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages}\n{task_instruction} {query}"
    },
    {
        "algorithm_name": "base_model_instruction-post_passage",
        "dataset_name": None,
        "instruction": "{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages}"
    }
]