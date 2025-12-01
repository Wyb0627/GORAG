import json
import math
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from utils import *
import networkx as nx
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import AzureOpenAI
import time
from prompts import *
import json
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from networkx.readwrite import json_graph
from prompts import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import defaultdict
import requests


class GORAG:
    def __init__(self, args):
        self.args = args
        if args.LLM == 'llama3':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            # model_id = 'meta-llama/Meta-Llama-3-8B'
        elif args.LLM == 'mistral':
            model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
        elif args.LLM == 'qwen2.5':
            model_id = 'Qwen/Qwen2.5-7B-Instruct'
        elif args.LLM == 'qwen2':
            model_id = 'Qwen/Qwen2-7B-Instruct'
        elif args.LLM == 'qwen3':
            model_id = 'Qwen/Qwen3-8B'
            # model_id = 'Qwen/Qwen3-30B-A3B-FP8'
            # model_id = 'Qwen/Qwen3-30B-A3B'
        elif args.LLM == 'llama3.1':
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif args.LLM == 'deepseek-8b':
            model_id = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
        elif args.LLM == 'deepseek':
            model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
        elif args.LLM == 'deepseek-14b':
            model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
        elif 'gpt' in args.LLM:
            model_id = args.LLM
        else:
            raise NotImplementedError('LLM not supported!')
        self.LLM_name = args.LLM
        self.vllm = args.vllm
        if self.vllm:
            self.vllm_client = OpenAI(
                api_key='EMPTY',
                base_url='http://localhost:8001/v1',
            )
            models = self.vllm_client.models.list()
            self.vllm_model = models.data[0].id
            print(f"Connected to remote vLLM API with model: {self.vllm_model}")
        self.Graph = nx.Graph()
        if 'gpt' not in args.LLM and not self.vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.LLM_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if args.LLM == 'qwen2.5' else 'auto',
                device_map="auto",
                attn_implementation="flash_attention_2",

            )
            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        if args.ke == 'keybert':
            from keybert import KeyBERT
            self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        if args.bert_sim:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        if args.LLM in 'qwen':
            self.tokenizer.bos_token = '<|endoftext|>'
        if args.LLM == 'deepseek':
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.token_usage = 0
        self.total_keyword = 0
        self.system_message = [{"role": "system",
                                "content": "You need to assign the given keyword to one of the most related topic provided. "
                                           "Respond only with the provided topic name, don't respond any extra words. "
                                }]
        self.node_count_list = []
        self.edge_count_list = []
        self.filtered_dict = {}
        self.st_time, self.inference_time, self.ke_time, self.oi_time, self.tfidf_time = 0, 0, 0, 0, 0
        self.index_time = 0
        self.update_count = 0
        self.df = defaultdict(int)

    def print_time_usage(self):
        print(
            f'Total time usage:\n ST: {self.st_time}\nInference: {self.inference_time}\nKE: {self.ke_time}\nOI: {self.oi_time}\nTFIDF: {self.tfidf_time}\nIndex: {self.index_time}')
        self.st_time, self.inference_time, self.ke_time, self.oi_time, self.tfidf_time = 0, 0, 0, 0, 0
        self.index_time = 0

    def get_steiner_tree_nodes(self, terminal_node: list, label_set: set):
        if self.args.graph_filter == 'unrelated':
            neighbor_set = set()
            for node in terminal_node:
                neighbor_set.update(self.Graph.neighbors(node))
            removed_set = label_set - (neighbor_set - set(terminal_node))
            input_graph = self.Graph.copy()
            input_graph.remove_nodes_from(removed_set)
            isolated_nodes = set(nx.isolates(input_graph))
            input_graph.remove_nodes_from(isolated_nodes)
            # self.node_count_list.append(input_graph.number_of_nodes())
            # self.edge_count_list.append(input_graph.number_of_edges())
            toral_removed_set = removed_set | isolated_nodes
            total_removed_node = len(toral_removed_set)
            self.node_count_list.append(total_removed_node)
        # elif self.args.graph_filter == 'llm':
        #     filter_prompt = [{"role": "system", "content": prompt_content}]
        #      filter_prompt.append({"role": "user",
        #                           "content": f"Please map the following graph nodes based on the terminal nodes: "
        #                                      f"{', '.join(terminal_node)}. "
        #                                     "Only keep the nodes that are related to the terminal nodes."})
        #     filtered_result=self.chat(filter_prompt,-1)
        #     pass
        else:
            input_graph = self.Graph
        st_begin = time.time()
        steiner_graph = nx.algorithms.approximation.steiner_tree(input_graph, terminal_node, weight='weight')
        st_end = (time.time() - st_begin) / 60
        self.st_time += st_end
        return steiner_graph

    def return_neighborhood(self, node_list: List[str], tuple_delimiter='[SEP]'):
        triples = []
        loaded_edges = []
        for node in node_list:
            if node in self.Graph.nodes():
                for neighbor in self.Graph.neighbors(node):
                    if f'{node} {neighbor}' in loaded_edges or f'{neighbor} {node}' in loaded_edges:
                        continue
                    loaded_edges.append(f'{node} {neighbor}')
                    relation = self.Graph[node][neighbor]['relation']
                    triples.append(f'({node}{tuple_delimiter}{neighbor}{tuple_delimiter}{relation})')
        return triples

    def llm_graph_filter(self, terminal_node: list):
        filter_prompt = [{"role": "system",
                          "content": system_prompt_graph_filter},
                         {"role": "user",
                          "content": f"Please coarse the following graph: "
                                     f"{', '.join(self.Graph.nodes())}. \n "
                                     f"With respect to the important terminal nodes: {', '.join(terminal_node)}. "}]
        response, _ = self.chat(filter_prompt, 4096)
        response_dict = eval(response)
        if not isinstance(response_dict, dict):
            response_dict = eval(json.loads(response))
        try:
            self.filtered_dict.update(response_dict)
        except:
            print('Error in updating filtered_dict, please check the response format.')
            print(response)

    def llm_graph_filter_step(self, terminal_node: list):
        filter_prompt = [{"role": "system",
                          "content": system_prompt_graph_filter},
                         {"role": "user",
                          "content": f"Please filter the following newly added nodes based on the existing graph nodes: "
                                     f"{', '.join(terminal_node)}. \n "
                                     f"Here are the existing graph nodes: {', '.join(self.Graph.nodes())}. "
                                     "Response me with the filtered nodes only, split them by , ."}]
        response, _ = self.chat(filter_prompt, 1024)
        keyword_set = set()
        for keyword in response.split(','):
            keyword = process_keyword(keyword.strip())
            keyword_set.add(keyword)
        return list(keyword_set)

    def fuzzy_matching(self, keywords: list, matching_model='soft'):
        entities = self.Graph.nodes()
        keyword_list = []
        hard_keyword_list = []
        # for target_keyword in keywords:
        prompt_content = FUZZY_MATCHING_PROMPT.format(entity_names=', '.join(entities),
                                                      target_entity_names=', '.join(keywords),
                                                      tuple_delimiter='[SEP]',
                                                      max_linkage='3')
        prompt = [{"role": "user", "content": prompt_content}]
        keyword_returned, keyword_score = self.chat(prompt, -1)
        keyword_returned = keyword_returned.lower()
        # if keyword.lower().replace('"', '') == 'none':
        #     continue
        # if keyword == '':
        #     continue
        for keyword in keyword_returned.split('[SEP]'.lower()):
            keyword = process_keyword(keyword)
            if keyword not in entities and keyword.replace('"', '') != 'none':
                if matching_model == 'hard':
                    while keyword not in entities:
                        loop_prompt = f"'{keyword}' is not in the keyword list provided, please give me the correct keyword, or None if you think none of the provided keywords is related to the target keywords."
                        prompt = [{"role": "user", "content": loop_prompt}]
                        keyword, _ = self.chat(prompt, -1)
                    hard_keyword_list.append(keyword)
            elif keyword.replace('"', '') == 'none' or keyword == '':
                continue
            else:
                keyword_list.append(keyword)
            self.total_keyword += 1
        return keyword_list, hard_keyword_list

    def LLM_triple_extraction(self, text: str,
                              tuple_delimiter='[SEP]',
                              record_delimiter='[SEP_REC]',
                              completion_delimiter='[FINAL]'):
        input_content = GRAPH_EXTRACTION_PROMPT.format(input_text=text,
                                                       tuple_delimiter=tuple_delimiter,
                                                       record_delimiter=record_delimiter,
                                                       completion_delimiter=completion_delimiter)
        input_prompt = [{"role": "user", "content": input_content}]
        outputs, _ = self.chat(input_prompt, -1)
        outputs = outputs.replace(completion_delimiter, '')
        outputs = outputs.split(record_delimiter)
        triple_list = []
        for extracted in outputs:
            extracted = extracted.strip().lstrip('(').rstrip(')')
            extracted = extracted.split(tuple_delimiter)
            if 'relation' in extracted[0]:
                triple_list.append(extracted[1:])  # [u,v,r,importance]
        return triple_list

    def node_link_data(self):
        print(f'Current graph:\n Nodes: {self.Graph.number_of_nodes()}, Edges: {self.Graph.number_of_edges()}')
        return json_graph.node_link_data(self.Graph)

    def print_graph(self):
        print_str = f'Current graph:\n Nodes: {self.Graph.number_of_nodes()}, Edges: {self.Graph.number_of_edges()}'
        print(print_str)
        return print_str

    def print_filtered_graph(self):
        if not self.node_count_list:
            print('No filtered graph to print.')
            return 'No filtered graph to print.'
        else:
            print_str = f'Average filtered nodes: {sum(self.node_count_list) / len(self.node_count_list)}'
            print(print_str)
            self.node_count_list = []
            return print_str

    def LLM_search(self, text: str, max_new_token: int, language='en'):
        if language == 'en':
            if max_new_token > 0:
                content = (
                    f"Describe the following academic research area shortly with less than {max_new_token} tokens: {text}. ")
            else:
                content = (
                    f"Describe the following academic research area: {text} ")
        else:
            if max_new_token > 0:
                content = (
                    f"简短描述以下用于Shell脚本分类的类别，使用不超过{max_new_token}字，该类别的不同层次以 - 分隔：{text}。")
            else:
                content = (
                    f"描述以下用于Shell脚本分类的类别，该类别的不同层次以 - 分隔：{text}。")
        prompt = [{"role": "user", "content": content}]
        response, res_score = self.chat(prompt, max_new_token)
        return response.strip()

    def keyword_entropy(self, keyword: str, candidate_label, args):
        candidate_label_str = ', '.join(candidate_label)
        entropy_dict = {}
        for response in range(args.keyword_inference_num):
            content = f"Among the following provided topics: {candidate_label_str}, the keyword '{keyword}', are most related to topic: "
            message = self.system_message.copy()
            message.append({"role": "user", "content": content})
            response_str, response_tup = self.chat(message, 16)
            response_seq, response_score = response_tup[0], response_tup[1]
            # label_token = self.tokenizer([response_str], return_tensors="pt", add_special_tokens=False).input_ids
            try:
                entropy = semantic_entropy_score_only(response_seq, response_score)
                if entropy < 0:
                    continue
            except:
                # print(entropy)
                print(response_str)
                print(response_seq)
                raise RuntimeError
            if response not in entropy_dict:
                entropy_dict[response_str] = [entropy]
            else:
                entropy_dict[response_str].append(entropy)
        return entropy_dict

    def find_path(self, keyword_set: List[str], most_diverse_label: str, path_delimiter=' -> '):
        count_dict = {}
        paths = []
        for keyword_s, keyword_t in combinations(keyword_set, 2):
            dist_d2s = 999999999999999999999
            dist_d2t = 999999999999999999999
            # if args.cluster and args.diversity:
            if nx.has_path(self.Graph, most_diverse_label, keyword_s):
                dist_d2s = nx.shortest_path_length(self.Graph, most_diverse_label, keyword_s)
            if nx.has_path(self.Graph, most_diverse_label, keyword_t):
                dist_d2t = nx.shortest_path_length(self.Graph, most_diverse_label, keyword_t)
            if nx.has_path(self.Graph, keyword_s, keyword_t):
                shortest_node_path = nx.shortest_path(self.Graph, keyword_s, keyword_t)
                shortest_path = [keyword_s]
                for i in range(1, len(shortest_node_path)):
                    n1 = shortest_node_path[i - 1]
                    n2 = shortest_node_path[i]

                    # relation = Graph[n1][n2]['relation']
                    # shortest_path.extend([relation, n2])
                    shortest_path.append(n2)
                cut_down_dist_s = int(dist_d2s / 2) if most_diverse_label.lower() != 'research area' else dist_d2s - 1
                cut_down_dist_t = int(dist_d2t / 2) if most_diverse_label.lower() != 'research area' else dist_d2t - 1
                if len(shortest_path) >= cut_down_dist_s + cut_down_dist_t:
                    continue
                graph_shortest_path = path_delimiter.join(shortest_path)
                paths.append('(' + graph_shortest_path + ')')
                for node in shortest_path[:cut_down_dist_s] + shortest_path[-cut_down_dist_t:]:
                    if node in count_dict:
                        count_dict[node] += 1
                    else:
                        count_dict[node] = 1
        return count_dict, paths

    def calculate_bert_sim(self, keyword: List[str], candidate_labels: List[str]):
        similarity_mat = []
        tokens = self.bert_tokenizer(keyword + candidate_labels, padding='longest', truncation=True,
                                     return_tensors='pt')
        with torch.no_grad():
            embeddings = self.bert_model(**tokens, output_hidden_states=False).last_hidden_state[:, 0, :]
        for idx in range(len(keyword)):
            similarity = cos_sim(embeddings[idx], embeddings[len(keyword):])
            similarity_mat.append(similarity)
        similarity_mat = torch.stack(similarity_mat, dim=0)
        max_sim_idx = torch.argmax(similarity_mat, dim=-1, keepdim=False)
        # print(similarity_mat.shape)  # [1,7]
        # print(similarity_mat)
        # print(keyword)
        # print(max_sim_idx)  # [0]
        similarity_max = torch.max(similarity_mat, dim=-1, keepdim=False)
        # print(similarity_mat_out.shape)
        most_sim_label = np.array(candidate_labels)[max_sim_idx].tolist()
        if isinstance(most_sim_label, str):
            most_sim_label = [most_sim_label]
        return most_sim_label, similarity_max.values.tolist()

    def remove_self_loop(self):
        self.Graph.remove_edges_from(nx.selfloop_edges(self.Graph))

    def index_training_text(self, training_loader: DataLoader, corpus, args):
        init_corpus_length = len(corpus)
        for data in training_loader:
            text = data['doc_token'][0]
            clean_text_list = re.findall(r'(?u)\b\w\w+\b', text)
            corpus.append(str(' '.join(clean_text_list)))
        if args.edge_weighting == 'tfidf':
            tfidf_begin = time.time()
            self.get_tfidf_vectorizer(corpus)
            tfidf_end = time.time() - tfidf_begin
            self.tfidf_time += tfidf_end / 60
        for text_idx, data in enumerate(training_loader):
            text = data['doc_token'][0]
            data_label = data['label'][0]
            if args.no_label_name:
                text_keywords, text_keywords_entropy = self.LLM_Extraction(text, -1)
            else:
                text_keywords, text_keywords_entropy = self.LLM_Extraction(text, -1, label=data_label)
            for keyword_tup in text_keywords:
                if isinstance(keyword_tup, tuple):
                    keyword_ori = keyword_tup[0]
                else:
                    keyword_ori = keyword_tup
                keyword_ori = re.sub(r'related keyword\S', '', keyword_ori, flags=re.IGNORECASE)
                keyword = process_keyword(keyword_ori).lower()
                if not keyword or re.match(r'\W', keyword):
                    continue
                cleaned_keyword = re.findall(r'(?u)\b\w\w+\b', keyword)
                cleaned_keyword = ' '.join(cleaned_keyword)
                self.total_keyword += 1
                if args.edge_weighting == 'tfidf':
                    tfidf_begin = time.time()
                    if cleaned_keyword not in self.tfidf_vectorizer.vocabulary_:
                        continue
                    else:
                        if isinstance(keyword_tup, tuple):
                            tfidf_score = keyword_tup[1]
                        else:
                            tfidf_score = self.tfidf_predict(cleaned_keyword, text_idx)
                    tfidf_end = time.time() - tfidf_begin
                    self.tfidf_time += tfidf_end / 60
                    try:
                        self.add_to_graph(keyword.lower(), data_label.lower(), 'related to', 1 - tfidf_score)
                    except:
                        print(keyword)
                        print(data_label)
                        print(cleaned_keyword)
                        print(keyword_tup)
                        raise ValueError
                    if not args.no_connect_label:
                        self.add_to_graph(data_label.lower(), 'label_connector', 'related to', 0.5)
                else:
                    self.add_to_graph(keyword.lower(), data_label.lower(), 'related to')
                    if not args.no_connect_label:
                        self.add_to_graph(data_label.lower(), 'label_connector', 'related to')

    def renew_graph(self):
        self.Graph.clear()

    def add_to_graph(self, n, v, r, w=1.0):
        index_begin = time.time()
        if n not in self.Graph.nodes():
            self.Graph.add_node(n)
        if v not in self.Graph.nodes():
            self.Graph.add_node(v)
        if (n, v) not in self.Graph.edges():
            self.Graph.add_edge(n, v, relation=r, weight=w, count=1, total_weight=w)
        else:
            self.Graph[n][v]['total_weight'] += w
            self.Graph[n][v]['count'] += 1
            self.Graph[n][v]['weight'] = self.Graph[n][v]['total_weight'] / self.Graph[n][v]['count']
            # G[n][v]['total_weight'].append(w)
            # G[n][v]['count'] += 1
            # G[n][v]['weight'] = min(G[n][v]['total_weight'])
        index_end = time.time() - index_begin
        self.index_time += index_end / 60

    def load_node_link_data(self, data):
        self.Graph = json_graph.node_link_graph(data)

    def LLM_Construct(self,
                      label: str,
                      label_desc: str,
                      args,
                      corpus=None,
                      text_idx=0,
                      candidate_labels=None):
        desc_keywords, desc_keywords_entropy = self.LLM_Extraction(label_desc, -1,
                                                                   target='desc')
        label_keywords_content = ('Give me some keywords related to the '
                                  f"academic research area: {label}, split keywords with [SEP]. "
                                  f"Please only reply me with related keywords, "
                                  f"don't reply anything else.")
        prompt = [{"role": "user", "content": label_keywords_content}]
        response, res_score = self.chat(prompt, 512)
        # print(response)
        label_keywords = response.split('[SEP]')
        keywords_relation = 'related to'
        skip = 0
        skip_old = 0
        graph_weight_list = []
        bert_sim_score = [1]
        prediction_confidence = 1
        if args.edge_weighting != 'unit' and not args.no_connect_label:
            self.add_to_graph(label.lower(), 'label_connector', keywords_relation, 0.5)
        elif not args.no_connect_label:
            self.add_to_graph(label.lower(), 'label_connector', keywords_relation)
        # keyword_set = set(desc_keywords + label_keywords + text_keywords)
        if not args.desc_keywords:
            for keyword_ori in label_keywords:
                keyword_ori = re.sub(r'related keyword\S', '', keyword_ori, flags=re.IGNORECASE)
                keyword = process_keyword(keyword_ori).lower()
                if not keyword or re.match(r'\W', keyword):
                    skip_old += 1
                    continue
                cleaned_keyword = re.findall(r'(?u)\b\w\w+\b', keyword)
                cleaned_keyword = ' '.join(cleaned_keyword)
                if args.edge_weighting == 'tfidf' and cleaned_keyword not in self.tfidf_vectorizer.vocabulary_:
                    continue
                if args.edge_weighting == 'semantic_entropy':
                    keyword_token = self.tokenizer([keyword_ori], return_tensors="pt",
                                                   add_special_tokens=False).input_ids
                    # EntropyCal.keyword_entropy(keyword, label.lower(), candidate_label)
                    keyword_entropy = semantic_entropy(keyword_token[0], res_score[0], res_score[1])
                    self.add_to_graph(keyword.lower(), label.lower(), keywords_relation, keyword_entropy)
                    graph_weight_list.append(keyword_entropy)
                elif args.edge_weighting == 'tfidf':
                    try:
                        tfidf_score = self.tfidf_predict(cleaned_keyword,
                                                         text_idx)
                    except IndexError:
                        # print(self.tfidf_matrix.shape)
                        print(text_idx)
                        raise IndexError
                    if 1 - tfidf_score <= 0:
                        print(cleaned_keyword)
                        print(tfidf_score)
                        raise ValueError
                    if args.bert_sim:
                        bert_sim_label, bert_sim_score = self.calculate_bert_sim([keyword], candidate_labels)
                        if not isinstance(bert_sim_label, list) or len(bert_sim_label[0]) == 1:
                            print(candidate_labels)
                            raise ValueError
                        self.add_to_graph(keyword.lower(), bert_sim_label[0].lower(), keywords_relation,
                                          1 - tfidf_score * bert_sim_score[0])
                    else:
                        self.add_to_graph(keyword.lower(), label.lower(), keywords_relation, 1 - tfidf_score)
                    graph_weight_list.append(1 - tfidf_score * bert_sim_score[0])
                else:
                    self.add_to_graph(keyword.lower(), label.lower(), keywords_relation)
                if args.hie_keywords:
                    self.hie_keywords_indexing(keyword,
                                               keywords_relation)
        elif args.desc_keywords:
            for keyword_ori, keyword_entropy in zip(desc_keywords,
                                                    desc_keywords_entropy):
                if isinstance(keyword_ori, tuple):
                    keyword_ori = keyword_ori[0]
                    keyword_score = keyword_ori[1]
                keyword_ori = re.sub(r'related keyword\S', '', keyword_ori, flags=re.IGNORECASE)
                keyword = process_keyword(keyword_ori).lower()
                if not keyword or re.match(r'\W', keyword):
                    skip_old += 1
                    continue
                cleaned_keyword = re.findall(r'(?u)\b\w\w+\b', keyword)
                cleaned_keyword = ' '.join(cleaned_keyword)
                if args.edge_weighting == 'tfidf' and cleaned_keyword not in self.tfidf_vectorizer.vocabulary_:
                    continue
                if args.edge_weighting == 'semantic_entropy':
                    self.add_to_graph(keyword.lower(), label.lower(), keywords_relation, keyword_entropy)
                    graph_weight_list.append(keyword_entropy)
                elif args.edge_weighting == 'tfidf':
                    tfidf_score = keyword_score
                    if args.bert_sim:
                        try:
                            bert_sim_label, bert_sim_score = self.calculate_bert_sim([keyword], candidate_labels)
                            if not isinstance(bert_sim_label, list) or len(bert_sim_label[0]) == 1:
                                print(candidate_labels)
                                raise ValueError
                            self.add_to_graph(keyword.lower(), bert_sim_label[0].lower(), keywords_relation,
                                              1 - tfidf_score * bert_sim_score[0])
                        except:
                            print(keyword)
                            print(bert_sim_label)
                            print(bert_sim_score)  # [[1.0]]
                            raise ValueError
                    else:
                        self.add_to_graph(keyword.lower(), label.lower(), keywords_relation, 1 - tfidf_score)

                    graph_weight_list.append(1 - tfidf_score * bert_sim_score[0] * prediction_confidence)
                    if 1 - tfidf_score <= 0:
                        print(cleaned_keyword)
                        print(tfidf_score)
                        raise ValueError
                else:
                    self.add_to_graph(keyword.lower(), label.lower(), keywords_relation)
                if args.hie_keywords:
                    self.hie_keywords_indexing(keyword,
                                               keywords_relation)
        return self.Graph

    def LLM_Extraction(self, text: str, max_new_token: int, target='text',
                       delimiter='[SEP]',
                       language='en',
                       label='none'):
        ex_begin = time.time()
        processed_keyword_set = set()
        keyword_entropy_list = []
        if self.args.ke == 'llm':
            if language == 'CN':
                if target == 'code':
                    content = (f'从以下脚本代码中提取一些关键指令，用{delimiter}分割，'
                               f'请只给我提取出的关键指令，不要返回其他任何内容：\n{text} ')

                else:
                    raise ValueError('Only implemented code extraction!')
            else:
                if target == 'text':
                    if label != 'none':
                        content = (f'Extract some text keywords and entities from the following '
                                   f"content, with respect to the {label} domain, split extracted keywords with {delimiter}, "
                                   f"please only give me the keywords or entities extracted, not any other words:\n{text} ")
                    else:
                        content = (f'Extract some keywords from the following '
                                   f"content, split extracted keywords with {delimiter}, "
                                   f"please only give me the keywords or entities extracted, not any other words:\n{text} ")
                elif target == 'desc':
                    content = ('Extract some keywords and entities from the following '
                               f"topic's description, split keywords with {delimiter}, "
                               f"please only give me the keywords or entities extracted, not any other words:\n{text} ")
                elif target == 'keyword':
                    content = ('Extract some keywords and entities from the following '
                               f"text content, split keywords with {delimiter}, "
                               f"please only give me the keywords or entities extracted, not any other words:\n{text} ")
                else:
                    raise ValueError('Only implemented text or description extraction!')
            prompt = [{"role": "user", "content": content}]
            response, keyword_score = self.chat(prompt, max_new_token)
            extracted_keywords = response.split(delimiter)
            skip = 0
            for keywords in extracted_keywords:
                # if args.semantic_entropy:
                if not keywords or keywords.strip() == '[]':
                    continue
                else:
                    normalized_keyword = process_keyword(keywords.strip())
                    if normalized_keyword:
                        processed_keyword_set.add(normalized_keyword)
        elif self.args.ke == 'keybert':
            extracted_keywords = self.kw_model.extract_keywords(text,
                                                                keyphrase_ngram_range=(1, 3),
                                                                use_mmr=True,
                                                                diversity=0.7)
            for tups in extracted_keywords:
                normalized_keyword = process_keyword(tups[0].strip())
                if normalized_keyword:
                    processed_keyword_set.add(normalized_keyword)
        elif self.args.ke == 'tfidf':
            sorted_scores = self.tfidf_ke([text])
            for keyword in sorted_scores:
                if keyword[1] < 0.1:
                    break
                normalized_keyword = process_keyword(keyword[0])
                if normalized_keyword:
                    processed_keyword_set.add((normalized_keyword, keyword[1]))
        else:
            raise NotImplementedError
        self.total_keyword += len(processed_keyword_set)
        ex_end = time.time() - ex_begin
        self.ke_time += ex_end / 60
        return list(processed_keyword_set), keyword_entropy_list

    def tfidf_ke(self, text: List[str]):
        df_array = np.where(self.tfidf_count_matrix > 0, 1, 0).sum(axis=0)
        doc_count = self.tfidf_count_matrix.shape[0]
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        current_counts = self.tfidf_vectorizer.transform(text)
        total_words = current_counts.sum()
        if total_words == 0:
            return {}
        sorted_scores = []
        for col_index in current_counts.indices:
            count = current_counts[0, col_index]
            tf = count / total_words
            df_val = df_array[col_index]
            # 计算TF-IDF得分
            idf = math.log((doc_count + 1) / (df_val + 1))
            normalized_score = (tf * idf) / math.log(doc_count + 1)
            word = feature_names[col_index]
            sorted_scores.append((word, normalized_score))
        # 按得分排序并返回结果
        sorted_scores.sort(key=lambda x: x[1], reverse=True)
        return sorted_scores

    def hie_keywords_indexing(self, keyword, keywords_relation):
        keyword_extract_prompt = [{"role": "user", "content": f'What is {keyword}?'}]
        keyword_desc, res_score_hie = self.chat(keyword_extract_prompt, -1)
        keyword_list, keyword_entropy = self.LLM_Extraction(keyword_desc, -1,
                                                            target='keyword')

        for keyword_of_keyword in keyword_list:
            if isinstance(keyword_of_keyword, tuple):
                keyword_of_keyword = keyword_of_keyword[0]
            keyword_of_keyword = process_keyword(keyword_of_keyword)
            if not keyword_of_keyword or re.match(r'\W', keyword):
                continue
            self.add_to_graph(keyword_of_keyword.lower(), keyword.lower(), keywords_relation)

    def chat(self, message, max_new_token, thinking=False):
        if 'gpt' not in self.LLM_name:
            if self.vllm:
                chat_completion = self.vllm_client.chat.completions.create(
                    messages=message,
                    model=self.vllm_model,
                    max_tokens=max_new_token if max_new_token > 0 else 4096,
                    temperature=0.0,
                    # stream=True,
                )
                return chat_completion.choices[0].message.content, ''
            elif 'qwen' in self.LLM_name or 'deepseek' == self.LLM_name:
                if self.LLM_name == 'qwen3':
                    text = self.tokenizer.apply_chat_template(
                        message,
                        add_generation_prompt=True,
                        tokenize=False,
                        return_tensors="pt",
                        enable_thinking=thinking
                    )
                else:
                    text = self.tokenizer.apply_chat_template(
                        message,
                        add_generation_prompt=True,
                        tokenize=False,
                        return_tensors="pt"
                    )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.LLM_model.device)
                # print(max_new_token)
                if 'deepseek' == self.LLM_name:
                    outputs = self.LLM_model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_token if max_new_token > 0 else 4096,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    outputs = self.LLM_model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_token if max_new_token > 0 else 4096,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                response = outputs.sequences[0, model_inputs.input_ids.shape[-1]:]
            else:
                input_ids = self.tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.LLM_model.device)
                # print(input_ids.shape)
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.LLM_model.device)
                if self.LLM_name == 'llama3.1':
                    max_token_llm = 4096
                else:
                    max_token_llm = None
                outputs = self.LLM_model.generate(
                    input_ids,
                    max_new_tokens=max_new_token if max_new_token > 0 else 4096,
                    eos_token_id=self.terminators,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.1,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                # print(outputs.sequences)
                response = outputs.sequences[0, input_ids.shape[-1]:]
            # print(response)
            final_response = self.tokenizer.decode(response, skip_special_tokens=True)
            if thinking:
                final_response = remove_thinking_process(final_response.lower()).strip()
            return final_response, (response, outputs.scores)
        elif 'webank' in self.LLM_name:
            return self.chat_webank(message)
        else:
            patience = 5
            api_key = '8409b8cc59224a4d83632f62c26f1606'
            # api_key = 'e95260778bf74ac9a14b27831b9dcc6c' # fteng
            self.token_usage += num_tokens_from_messages(message, self.LLM_name)
            while patience > 0:
                patience -= 1
                try:
                    client = AzureOpenAI(
                        api_key=api_key,
                        azure_endpoint="https://hkust.azure-api.net",
                        api_version='2024-10-21',
                    )
                    response = client.chat.completions.create(
                        model=self.LLM_name,
                        messages=message,
                        n=1,
                        max_tokens=4096
                    )
                    prediction = response.choices[0].message.content.strip()
                    if prediction != "" and prediction != None:
                        return prediction, ([], [])
                except Exception as e:
                    print(e)
                    time.sleep(0)
            return 'no prediction', ([], [])

    def chat_webank(self,
                    message: List[Dict[str, str]]) -> str:
        API_KEY = ''
        API_URL = ''
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        data = {
            "model": 'qwen-72b',
            "messages": message
        }
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=json.dumps(data),
                timeout=60  # 设置超时时间
            )
            # 检查响应状态码
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"

        except requests.exceptions.RequestException as e:
            return f"Network Error: {str(e)}"

    def tfidf_predict(self, query, doc_idx):
        idx = self.tfidf_vectorizer.vocabulary_[query]
        total_count_exist = self.tfidf_count_matrix[:, idx]
        df = np.where(total_count_exist > 0, 1, 0)
        doc_count = len(self.tfidf_count_matrix)
        idf = math.log(doc_count + 1 / (np.sum(df) + 1))
        current_doc_count = self.tfidf_count_matrix[doc_idx, idx]
        tf = current_doc_count / np.sum(self.tfidf_count_matrix[doc_idx, :])
        tfidf_score = tf * idf / math.log(doc_count + 1)
        return tfidf_score

    def generate_keywords(self, train_loader: DataLoader):
        label_keyword_dict = {}
        for data in train_loader:
            text = data['doc_token'][0]
            label = data['label'][0]
            if label not in label_keyword_dict:
                label_keyword_dict[label] = set()
            clean_text_list = re.findall(r'(?u)\b\w\w+\b', text)
            cleaned_text = str(' '.join(clean_text_list))
            text_keywords, _ = self.LLM_Extraction(cleaned_text, -1, target='text')
            for keyword in text_keywords:
                if isinstance(keyword, tuple):
                    label_keyword_dict[label].add(keyword[0])
                else:
                    label_keyword_dict[label].add(keyword)
        for label in label_keyword_dict:
            label_keyword_dict[label] = list(label_keyword_dict[label])
        return label_keyword_dict

    def reset_token_usage(self):
        self.token_usage = 0

    def get_tfidf_vectorizer(self, corpus: List[str]):
        self.tfidf_vectorizer = CountVectorizer(analyzer='word',
                                                lowercase=True,
                                                stop_words='english',
                                                ngram_range=(1, 3))
        self.tfidf_count_matrix = self.tfidf_vectorizer.fit_transform(corpus).toarray()

    def update_tfidf_dynamic(self, new_doc: str):
        """
        动态更新TF-IDF计数矩阵，处理新添加的文档，包括新词
        Args:
            new_doc: 新添加的文档
        """
        # 获取当前词汇表
        current_vocab = set(self.tfidf_vectorizer.get_feature_names_out())

        # 使用CountVectorizer分析新文档中的词
        temp_vectorizer = CountVectorizer(analyzer='word',
                                          lowercase=True,
                                          stop_words='english',
                                          ngram_range=(1, 3))
        temp_vectorizer.fit([new_doc])
        new_vocab = set(temp_vectorizer.get_feature_names_out())

        # 如果有新词，扩展词汇表
        if new_vocab - current_vocab:
            # 获取新词的词频统计
            new_counts = temp_vectorizer.transform([new_doc]).toarray()

            # 更新vectorizer的词汇表
            new_terms = new_vocab - current_vocab
            for idx, term in enumerate(new_terms):
                self.tfidf_vectorizer.vocabulary_[term] = len(current_vocab) + idx

            # 获取新文档在扩展词汇表下的词频统计
            new_counts_full = self.tfidf_vectorizer.transform([new_doc]).toarray()

            # 更新计数矩阵
            if not hasattr(self, 'tfidf_count_matrix'):
                self.tfidf_count_matrix = new_counts_full
            else:
                # 为现有文档添加新词的零计数
                old_matrix = np.hstack([self.tfidf_count_matrix,
                                        np.zeros((self.tfidf_count_matrix.shape[0], len(new_terms)))])
                # 添加新文档的计数
                self.tfidf_count_matrix = np.vstack([old_matrix, new_counts_full])
        else:
            # 如果没有新词，直接更新计数矩阵
            new_counts = self.tfidf_vectorizer.transform([new_doc]).toarray()
            if not hasattr(self, 'tfidf_count_matrix'):
                self.tfidf_count_matrix = new_counts
            else:
                self.tfidf_count_matrix = np.vstack([self.tfidf_count_matrix, new_counts])

    def online_indexing(self, keywords: set, responses: str, args,
                        candidate_labels: List[str], corpus: List[str], inferenced_corpus):
        oi_begin = time.time()
        keywords_list = list(keywords)
        # self.update_count += 1
        if args.edge_weighting == 'tfidf':
            tfidf_begin = time.time()
            self.update_tfidf_dynamic(corpus[-1])
            tfidf_end = time.time() - tfidf_begin
            self.tfidf_time += tfidf_end / 60
        if args.bert_sim:
            bert_sim_label, bert_sim_score = self.calculate_bert_sim(keywords_list, candidate_labels)
            if not isinstance(bert_sim_label, list) or len(bert_sim_label[0]) == 1:
                print(candidate_labels)
                raise ValueError
            raise NotImplementedError
        for idx, keyword_tup in enumerate(keywords_list):
            if isinstance(keyword_tup, tuple):
                keyword = keyword_tup[0]
            else:
                keyword = keyword_tup
            if args.edge_weighting == 'unit':
                self.add_to_graph(keyword.lower(), responses.lower().strip(), 'related to')
            else:
                if args.edge_weighting == 'semantic_entropy':
                    entropy_dict = self.keyword_entropy(keyword, candidate_labels, args)
                    for response_entropy in entropy_dict:
                        entropy_dict[response_entropy] = sum(entropy_dict[response_entropy]) / np.log(
                            args.keyword_inference_num)
                    for response_label, response_entropy in entropy_dict.items():
                        self.add_to_graph(keyword, response_label.lower(), 'related to', response_entropy)
                elif args.edge_weighting == 'tfidf':
                    cleaned_keyword = re.findall(r'(?u)\b\w\w+\b', keyword)
                    cleaned_keyword = ' '.join(cleaned_keyword)
                    if cleaned_keyword not in self.tfidf_vectorizer.vocabulary_:
                        continue
                    tfidf_begin = time.time()
                    if isinstance(keyword_tup, tuple):
                        tfidf_score = keyword_tup[1]
                    else:
                        tfidf_score = self.tfidf_predict(cleaned_keyword, -1)
                    tfidf_end = time.time() - tfidf_begin
                    self.tfidf_time += tfidf_end / 60
                    self.add_to_graph(keyword.lower().strip(), responses.lower().strip(), 'related to',
                                      1 - tfidf_score)
        oi_end = time.time() - oi_begin
        self.oi_time += oi_end / 60

    def recompute_idf(self, df_dict, total_docs, smooth=True):
        vocab_size = max(df_dict.keys()) + 1 if df_dict else 0
        idf = np.zeros(vocab_size, dtype=np.float64)
        for word_id, df in df_dict.items():
            # IDF 公式：log((1 + total_docs) / (1 + df)) + 1
            if smooth:
                idf[word_id] = np.log((1 + total_docs) / (1 + df)) + 1
            else:
                idf[word_id] = np.log(total_docs / df) + 1  # 不平滑版本
        return idf

    def index(self, documents: List[str]):
        pass
