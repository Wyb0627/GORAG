# GORAG
The official code repository of the paper "GORAG: Graph-based Online Retrieval Augmented Generation for Dynamic Few-shot Social Media Text Classification"

Install all requirements via:

```
  pip install -r requirements.txt
```

And run LLM services on VLLM, you may change your VLLM api settings in GORAG.py.

Then run experiments via (Use the CAD dataset as an example): 

```
   python llm_4_round.py --gpu 0 --graphrag --context LLM --LLM qwen2.5 --steiner_tree --edge_weighting tfidf --desc_keywords --shot 5 --online_index all --round 4 --vllm --dataset cad --ke tfidf
```


## Parameters


--gpu: The GPU number used;

--dataset: The dataset experimented on;

--retrieval_mode: Whether to apply dense or hybrid retrieval;

--ner_threshold: The entity extraction threshold;

--max_ngram_length: The maximum n-gram considered when entity extraction;

--include_passage_nodes: Whether to include passage nodes into the graph;

--no_label_name: Set for Reuters, where the label names are not available;

--LLM: The LLM for use, available LLMs;

--edge_weighting: Whether to apply a tfidf-based edge weighting mechanism or unit weight;

--shot: The number of shots;

--ke: The keyword extraction method applied;

--round: The number of dataset split rounds set to 4 for full experiment running;

--online_index: Whether to apply the online indexing mechanism.
