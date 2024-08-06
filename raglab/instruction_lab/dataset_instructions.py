DATA_INSTRUCTIONS = [
    {
        "tips": "-------------------Dataset Template-------------------------",
        "dataset_name": "dataset name",
        "type":"task type",
        "instruction": "dataset instruction"
    },
    {
        "dataset_name": "",
        "type":'',
        "instruction": ""
    },
    {
        "dataset_name": "nq",
        "type":'QA',
        "instruction": ""
    },
    {
        "dataset_name": "triviaqa",
        "type": 'QA',
        "instruction": ""
    },
    {
        "dataset_name": "PopQA",
        "type": 'QA',
        "instruction": ""
    },
    {
        "dataset_name": "squad",
        "type":'QA',
        "instruction": ""
    },
    {
        "dataset_name": "ms_marco",
        "type":'QA',
        "instruction": ""
    },
    {
        "dataset_name": "narrative_qa",
        "type":'QA',
        "instruction": ""
    }, 
    {
        "dataset_name": "wiki_qa",
        "type":'QA',
        "instruction": ""
    }, 
    {
        "dataset_name": "web_questions",
        "type":'QA',
        "instruction": ""
    }, 
    {
        "dataset_name": "ambig_qa",
        "type":'QA',
        "instruction": ""
    }, 
    {
        "dataset_name": "siqa",
        "type":'QA',
        "instruction": ""
    }, 
    {
        "dataset_name": "commense_qa",
        "type":'QA',
        "instruction": ""
    }, 
    {
        "dataset_name": "boolq",
        "type":'QA',
        "instruction": ""
    },
    {
        "dataset_name": "piqa",
        "type":'QA',
        "instruction": ""
    },
    {
        "dataset_name": "fermi",
        "type":'QA',
        "instruction": ""
    },
    {
        "dataset_name": "HotPotQA",
        "type": 'Multi_Hop_QA',
        "instruction": ""
    },
    {
        "dataset_name": "2WikiMultiHopQA",
        "type": 'Multi_Hop_QA',
        "instruction": ""
    },
    {
        "dataset_name": "musique",
        "type": 'Multi_Hop_QA',
        "instruction": ""
    },
    {
        "dataset_name": "bamboogle",
        "type": 'Multi_Hop_QA',
        "instruction": ""
    },
    {
        "dataset_name": "ASQA",
        "type": "Longform_QA",
        "instruction": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."
    },
    {
        "dataset_name": "eli5",
        "type": "Longform_QA",
        "instruction": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."
    },
    {
        "dataset_name": "Factscore",
        "type": 'QA',
        "instruction": ""
    },
    {
        "dataset_name": "MMLU",
        "type": "MultiChoice",
        "instruction": "Given four answer candidates, A, B, C and D, choose the best answer choice."
    },
    {
        "dataset_name": "truthful_qa",
        "type":'MultiChoice',
        "instruction": "Given four answer candidates, A, B, C and D, choose the best answer choice."
    },
    {
        "dataset_name": "hellaswag",
        "type":'MultiChoice',
        "instruction": "Given four answer candidates, A, B, C and D, choose the best answer choice."
    },
    {
        "dataset_name": "arc",
        "type": "MultiChoice",
        "instruction": "Given four answer candidates, A, B, C and D, choose the best answer choice."
    },
    {
        "dataset_name": "openbookqa",
        "type": "MultiChoice",
        "instruction": "Given four answer candidates, A, B, C and D, choose the best answer choice."
    },
    {
        "dataset_name": "PubHealth",
        "type": 'fact',
        "instruction": "Is the following statement correct or not? Say true if it's correct; otherwise say false."
    },
    {
        "dataset_name": "StrategyQA",
        "type": 'fact',
        "instruction": "You are only allowed to answer True or False, and generating other types of responses is prohibited."
    },

    {
        "dataset_name": "Feverous",
        "type": 'fact',
        "instruction": "Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters."
    },
    {
        "dataset_name": "wned",
        "type":'entity',
        "instruction": 
        (
            "In the given text passage, identify and label specific types of named entities (such as locations, names of people, etc.), and confirm the correct linking of these entities." 
            "For each marked entity, provide a clear answer to verify its correctness."
            "Named entities already marked (using [START_ENT] and [END_ENT] tags)."
         )
    },
    {
        "dataset_name": "trex",
        "type":'slot',
        "instruction": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity."
    },
]
