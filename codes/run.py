import os

from argparse import ArgumentParser

import numpy as np

parser = ArgumentParser()
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--dataset", help="dataset name", type=str, default="wos")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cot", action="store_true")
parser.add_argument("--ft", action="store_true")
parser.add_argument("--vllm", action="store_true")
parser.add_argument("--ke", type=str, default='llm')
parser.add_argument("--context", type=str, default='none')
parser.add_argument("--filtered_labels", action="store_true")
parser.add_argument("--no_test", action="store_true")
parser.add_argument("--max_desc_token", type=int, default=32)
parser.add_argument("--pre_classify", action="store_true")
parser.add_argument("--diversity", action="store_true")
parser.add_argument("--sentence_filter", action="store_true")
parser.add_argument("--graphrag", action="store_true")
parser.add_argument("--LLM", type=str, default='llama3')
parser.add_argument("--fuzzy_matching", type=str, default='none')
parser.add_argument("--renew_graph", action="store_true")
parser.add_argument("--keyword_only", action="store_true")
parser.add_argument("--desc_keywords", action="store_true")
parser.add_argument("--predict_confidence", action="store_true")
parser.add_argument("--index_label_only", action="store_true")
parser.add_argument("--steiner_tree", action="store_true")
parser.add_argument("--bert_sim", action="store_true")
parser.add_argument("--shot", type=int, default=0)
parser.add_argument("--edge_weighting", type=str, default="unit")
parser.add_argument("--no_label_name", action="store_true")
parser.add_argument("--no_connect_label", action="store_true")
parser.add_argument("--steiner_direct", action="store_true")
parser.add_argument("--round", type=int, default=4)
parser.add_argument("--graph_filter", type=str, default="none")
parser.add_argument("--llm_graph_filter", action="store_true")
parser.add_argument("--gt_index", action="store_true")
parser.add_argument("--online_index", type=str, default='none',
                    help='all or filtered. Index when response in all labels or filtered labels')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
print(args)
# print(args.gpu)
# print(os.environ['CUDA_VISIBLE_DEVICES'])
from utils import *

check_device()
from dataset import *
from torch.utils.data import DataLoader, ConcatDataset
import json
import time
from GORAG import GORAG
import tqdm
import networkx as nx
from itertools import combinations, chain
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

print('Starting Time:')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
setup_seed(args.seed)
device = torch.device("cuda")


def desc_label(GORAG, candidate_label: list, short_desc=False):
    label_desc = {}
    if context == 'KG':
        for label_name in candidate_label:
            desc = wikidata_search(label_name)
            if desc:
                label_desc[label_name] = f"{label_name}: {desc}"
            else:
                no_desc.append(f"{label_name}: No description found")
    elif context == 'LLM':
        for label_name in candidate_label:
            if args.graphrag and not short_desc:
                desc = GORAG.LLM_search(label_name, -1)
            else:
                desc = GORAG.LLM_search(label_name, args.max_desc_token)
            label_desc[label_name] = f"{label_name}: {desc}"
    if label_desc:
        total_label_desc.extend(label_desc)
    return label_desc


def test(test_loader: DataLoader,
         gorag_model: GORAG,
         round_num: str,
         candidate_label=None,
         # label_desc=None,
         # max_desc_label: int = 3,
         label_desc_short=None,
         online_index_round=False
         ):
    if label_desc_short is None:
        label_desc_short = {}
    if args.online_index != 'none' and online_index_round:
        print(f'Online Indexing at round {round_num}! ')
    print('==================')
    ans_list = []
    label_list = []
    total_res = []
    truth_label = []
    bad_case = []
    good_case = []
    NO_KG_MATCHING = 0
    NO_PATH = 0
    FUZZY = 0
    FIND_CATEGORY = 0
    round_data_num = len(test_loader)
    hallucination = 0
    graph_info = []
    for data_idx, item in enumerate(tqdm.tqdm(test_loader, desc=f'{round_num} Testing')):
        prompt = item['doc_token'][0]
        label = item[f'label'][0]
        # label_num = item[f'label_#'][0]
        un_contained_keyword = set()
        st_graph_json = ''
        KG_selected_label = set()
        if args.online_index != 'none' and online_index_round:
            clean_corpus_list_oi = re.findall(r'(?u)\b\w\w+\b', prompt)
            cleaned_corpus.append(str(' '.join(clean_corpus_list_oi)))
        if args.graphrag:
            # 添加fewshot learning
            # GORAG.remove_self_loop()
            keyword_list, keyword_entropy = GORAG.LLM_Extraction(prompt, -1)
            keyword_keyword_list = []
            keyword_set = set()
            # node_list = list(Graph.nodes().keys())

            for keyword in keyword_list + keyword_keyword_list:
                if isinstance(keyword, tuple):
                    processed_keyword = process_keyword(keyword[0]).lower()
                    keyword_score = keyword[1]
                    if processed_keyword.lower() in GORAG.Graph.nodes():
                        keyword_set.add(processed_keyword.lower())
                    else:
                        un_contained_keyword.add((processed_keyword.lower(), keyword_score))
                else:
                    keyword = process_keyword(keyword)
                    if keyword.lower() in GORAG.Graph.nodes():
                        keyword_set.add(keyword.lower())
                    else:
                        un_contained_keyword.add(keyword.lower())
            if not keyword_set:
                # 整篇文档的keyword都不在KG中
                # print('No keyword found in the graph!')
                # print(f'Prompt: {prompt}')
                if args.fuzzy_matching == 'soft' or args.fuzzy_matching == 'hard':
                    fuzzy_keyword, fuzzy_keyword_hard = GORAG.fuzzy_matching(keyword_list, args.fuzzy_matching)
                    if args.fuzzy_matching == 'soft':
                        for keyword in fuzzy_keyword:
                            keyword_set.add(keyword.lower())
                    else:
                        for keyword in fuzzy_keyword + fuzzy_keyword_hard:
                            keyword_set.add(keyword.lower())
                    if keyword_set:
                        FUZZY += 1
                    for keyword in fuzzy_keyword:
                        keyword_set.add(keyword.lower())
            if keyword_set:
                if args.steiner_tree:
                    paths = []
                    nei_triple_list = []
                    try:
                        st_graph = GORAG.get_steiner_tree_nodes(list(keyword_set), candidate_label)
                        st_nodes = list(st_graph.nodes())
                    # except KeyError:
                    #     for wrong_keyword in keyword_set:
                    #         if wrong_keyword.lower() not in GORAG.Graph.nodes():
                    #            print(wrong_keyword)
                    #    raise KeyError
                    except:
                        st_nodes = []
                        st_graph = ''

                    for node in st_nodes:
                        if node in candidate_label:
                            KG_selected_label.add(node)
                    if args.steiner_direct and st_graph:
                        st_graph = nx.node_link_data(
                            st_graph, source="from", target="to"
                        )
                        st_graph.pop('directed', None)
                        st_graph.pop('graph', None)
                        st_graph.pop('nodes', None)
                        st_graph.pop('multigraph', None)
                        for d in st_graph['links']:
                            d.pop('count', None)
                            d.pop('relation', None)
                            d.pop('total_weight', None)
                            d['weight'] = format(d['weight'], '.4f')
                        st_graph_json = json.dumps(st_graph)
                else:
                    count_dict, paths = GORAG.find_path(list(keyword_set), 'research area')
                    nei_triple_list = GORAG.return_neighborhood(list(keyword_set))
                    if count_dict:
                        # Sort the count_dict w.r.t to its value
                        sorted_count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
                        for node, count in sorted_count_dict:
                            if node in candidate_label:
                                KG_selected_label.add(node)
            else:
                paths, nei_triple_list = [], []
            if not args.steiner_tree and nei_triple_list or paths:
                if not paths:
                    NO_PATH += 1

                for triples in nei_triple_list:
                    components = triples.lower().split('[SEP]'.lower())
                    if components[0] in candidate_label:
                        KG_selected_label.add(components[0])
                    if components[1] in candidate_label:
                        KG_selected_label.add(components[1])

            if KG_selected_label:
                label_desc_filtered = []
                KG_selected_label = list(KG_selected_label)
                for category in KG_selected_label:
                    if not args.no_label_name:
                        label_desc_filtered.append(label_desc_short[category])
                    else:
                        label_desc_filtered.append(', '.join(label_desc_short[category]))
                FIND_CATEGORY += 1
                desc_text = '\n'.join(label_desc_filtered)
                if args.keyword_only and keyword_list:
                    content = (
                        f"Now you need to classify texts into one of the following classes: \n{', '.join(KG_selected_label)}.\n"
                        f"Here are some descriptions for each of the above mentioned classes stating their differences:\n{desc_text}.\n "
                        # f"Here is the candidate label in KG based on the keywords extracted from the text:\n{', '.join(KG_selected_label)}.\n"
                        f"Please answer strictly by only output one of the above mentioned class based on the following keywords of the text, "
                        f"do not output any other words: \n{', '.join(keyword_list)}")
                else:
                    if args.steiner_direct and st_graph_json:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(KG_selected_label)}.\n"
                            f"Here are some descriptions for each of the above mentioned classes stating their differences:\n{desc_text}.\n"
                            f"Here is the graph structure of the keywords extracted from the text:\n{st_graph_json}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following text, do not output any other words: \n{prompt}")
                    else:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(KG_selected_label)}.\n"
                            f"Here are some descriptions for each of the above mentioned classes stating their differences:\n{desc_text}.\n "
                            # f"Here is the candidate label in KG based on the keywords extracted from the text:\n{', '.join(KG_selected_label)}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following text, do not output any other words: \n{prompt}")
            else:
                label_desc_filtered = []
                for category in candidate_label:
                    if not args.no_label_name:
                        label_desc_filtered.append(label_desc_short[category])
                    else:
                        label_desc_filtered.append(', '.join(label_desc_short[category]))
                desc_text = '\n'.join(label_desc_filtered)
                if args.keyword_only and keyword_list:
                    if not args.no_label_name:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(candidate_label)}.\n"
                            f"Here are some descriptions for each of the above mentioned classes stating their differences:\n{desc_text}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following keywords of the text,"
                            f" do not output any other words: \n{', '.join(keyword_list)}")
                    else:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(candidate_label)}.\n"
                            f"Here are some keywords for each of the above mentioned classes representing their features:\n{desc_text}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following keywords of the text,"
                            f" do not output any other words: \n{', '.join(keyword_list)}")
                else:
                    if not args.no_label_name:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(candidate_label)}.\n"
                            f"Here are some descriptions for each of the above mentioned classes stating their differences:\n{desc_text}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following text, do not output any other words: \n{prompt}")
                    else:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(candidate_label)}.\n"
                            f"Here are some keywords for each of the above mentioned classes representing their features:\n{desc_text}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following text, do not output any other words: \n{prompt}")
        else:
            if args.cot:
                content = (
                    f"Now you need to classify texts into one of the following classes: {', '.join(candidate_label)}. "
                    f"Please answer by firstly give out one of the above mentioned class and then provide your "
                    f"chain-of-thought. Here is the text to classify: '{prompt}'. "
                )
            elif args.context != 'none':
                label_desc_list = [' ,'.join(keyword_l) for keyword_l in label_desc_short.values()]
                label_desc_str = '; '.join(label_desc_list)
                content = (
                    f"Now you need to classify texts into one of the following classes: {', '.join(candidate_label)}. "
                    f"Here are some keywords for each of the above mentioned classes stating their differences: {label_desc_str}. "
                    f"Please answer shortly by only give out one of the above mentioned class based on the following text: '{prompt}'")
            else:
                content = (
                    f"Now you need to classify texts into one of the following classes: {', '.join(candidate_label)}. "
                    # f"Here are some descriptions for each of the above mentioned classes stating their differences: {'; '.join(label_desc)}. "
                    f"Please answer shortly by only give out one of the above mentioned class based on the following text: '{prompt}'")

        messages_test = [{"role": "user", "content": content}]
        counter = 0
        while True:
            inference_begin = time.time()
            res, res_score = GORAG.chat(messages_test, max_new_token, False)
            cleaned_res = re.findall(r'(?u)\b\w\w+\b', res)
            cleaned_res = ' '.join(cleaned_res)
            cleaned_res = cleaned_res.lower().strip()
            inference_end = time.time() - inference_begin
            GORAG.inference_time += inference_end / 60
            if cleaned_res and args.graphrag and un_contained_keyword and cleaned_res in candidate_label:
                # if un_contained_keyword and args.llm_graph_filter:
                # un_contained_keyword = GORAG.llm_graph_filter_step(list(un_contained_keyword))
                if args.gt_index:
                    indexed_label = label.lower().strip()
                else:
                    indexed_label = cleaned_res
                if args.online_index == 'all' and online_index_round:
                    GORAG.online_indexing(un_contained_keyword,
                                          indexed_label,
                                          args,
                                          candidate_label,
                                          cleaned_corpus,
                                          inferenced_corpus)
                elif args.online_index == 'filtered' and online_index_round:
                    GORAG.online_indexing(un_contained_keyword,
                                          indexed_label,
                                          args,
                                          KG_selected_label,
                                          cleaned_corpus,
                                          inferenced_corpus)
                break
            elif cleaned_res and cleaned_res in label2id:
                break
            counter += 1
            if counter == 1:
                # print(f'No label found for: \n {prompt}')
                # print(f'res: {cleaned_res}')
                hallucination += 1
                break

        if args.predict_confidence:
            for label_name in candidate_label:
                token_probs = []
                if label_name.lower() in cleaned_res.lower():
                    for i, score in enumerate(res_score[1]):
                        # Convert the scores to probabilities
                        probs = torch.softmax(score, -1)
                        # Take the probability for the generated tokens (at position i in sequence)
                        token_probs.append(probs[0, res_score[0][i]].item())
                    prod = 1
                    for x in token_probs:
                        prod *= x
                    inferenced_corpus.append((prompt.lower(), label_name.lower(), prod))
                    break
        # if isinstance(item['label_#'][0], torch.Tensor):
        #     truth_label_num = item['label_#'][0].tolist()
        # else:
        #     truth_label_num = item['label_#'][0]
        if label.lower().strip() == cleaned_res:
            try:
                ans_list.append(label2id[cleaned_res])
            except:
                print(cleaned_res)
                print(label2id)
                raise KeyError
            good_case_dict = {
                'text': prompt,
                'prompt': content,
                'label': item['label'][0],
                # 'label_#': truth_label_num,
                'response': cleaned_res,
            }
            good_case.append(good_case_dict)
        else:
            if cleaned_res in label2id:
                ans_list.append(label2id[cleaned_res])
            else:
                ans_list.append(len(label2id))
            bad_case_dict = {
                'text': prompt,
                'prompt': content,
                'label': item['label'][0],
                # 'label_#': truth_label_num,
                'response': cleaned_res,
            }
            bad_case.append(bad_case_dict)
        if label.lower().strip() not in label2id:
            print(label.lower().strip())
            print('\n')
            print(label2id)
        label_list.append(label2id[label.lower().strip()])
        if data_idx in [3, int(round_data_num / 2), int(round_data_num / 4), int(3 * round_data_num / 4)]:
            print(f'Graph information at {data_idx}')
            print_str = GORAG.print_graph()
            if not print_str:
                raise ValueError('Graph print str is empty!')
            graph_info.append(print_str)
    print(f'No KG matching: {NO_KG_MATCHING}')
    print(f'No path: {NO_PATH}')
    print(f'Fuzzy matched: {FUZZY}')
    print(f'Find category: {FIND_CATEGORY}')
    print(f'Hallucination rate: {hallucination / round_data_num}')
    print('\n'.join(graph_info[1:]))
    return ans_list, total_res, truth_label, bad_case, good_case, label_list


setup_seed(args.seed)
ft = args.ft
context = args.context
cot = args.cot
use_sentence_level_filter = args.sentence_filter
total_label_desc = []
no_desc = []
inferenced_corpus = []
if cot:
    max_new_token = 32
else:
    max_new_token = 32
labels = []
if args.dataset in ['wos', 'cad'] or (args.dataset == 'IFS' and not args.no_label_name):
    with open(f'../dataset/{args.dataset}/id2label.json', 'r') as file:
        id2label = json.load(file)
    with open(f'../dataset/{args.dataset}/label2id.json', 'r') as file:
        label2id = json.load(file)
    for key, value in id2label.items():
        # labels.append(f'{int(key)}: {value}')
        cleaned_label = f'{value}'.lower()
        cleaned_label = re.findall(r'(?u)\b\w\w+\b', cleaned_label)
        cleaned_label = ' '.join(cleaned_label)
        labels.append(str(cleaned_label))
else:
    label2id = {}
    if args.dataset == 'IFS' and args.no_label_name:
        for i in range(64):
            labels.append(f'label {i}')
            label2id[f'label {i}'] = i
    else:
        for i in range(31):
            labels.append(f'label {i}')
            label2id[f'label {i}'] = i
total_label_list = list(label2id.keys())
GORAG = GORAG(args)

round_data_dict = {
    'n1': {},
    'n2': {},
    'n3': {},
    'n4': {},
}
if args.round == 6:
    round_data_dict['n5'] = {}
    round_data_dict['n6'] = {}
elif args.round == 8:
    round_data_dict['n5'] = {}
    round_data_dict['n6'] = {}
    round_data_dict['n7'] = {}
    round_data_dict['n8'] = {}

if args.dataset == 'wos':
    if args.round == 6:
        round_label_dict = load_wos_taxnomy(6)
    elif args.round == 8:
        round_label_dict = load_wos_taxnomy(8)
    else:
        round_label_dict = load_wos_taxnomy(4)
elif args.dataset == 'cad':
    round_label_dict = {
        'n1': ["neutral",
               "slur",
               "threatening_language",
               "dehumanization"],
        'n2': ["glorification",
               "person_directed_abuse",
               "person_directed_counter_speech"],
        'n3': ["identity_directed_animosity",
               "identity_directed_derogation",
               "identity_directed_counter_speech"],
        'n4': ["affiliation_directed_animosity",
               "affiliation_directed_derogation",
               "affiliation_directed_counter_speech"]
    }
elif args.dataset == 'IFS':
    if args.no_label_name:
        round_label_dict = {
            'n1': [f'label {i}' for i in range(16)],
            'n2': [f'label {i}' for i in range(16, 32)],
            'n3': [f'label {i}' for i in range(32, 48)],
            'n4': [f'label {i}' for i in range(48, 64)],
        }
    else:
        round_label_dict = load_ifs_taxnomy(4)
else:
    round_label_dict = {
        'n1': [f'label {i}' for i in range(8)],
        'n2': [f'label {i}' for i in range(8, 16)],
        'n3': [f'label {i}' for i in range(16, 24)],
        'n4': [f'label {i}' for i in range(24, 31)],
    }
exp_name = return_exp_name(args)

for round_num, data_dict in round_data_dict.items():
    for split in ['train', 'test']:
        if args.dataset == 'wos':
            if args.shot == 0:
                round_dataset = WOS4RoundDataset(
                    f'../dataset/wos/entailment_data/wos_1_shot_{args.round}/split/{round_num}/{split}.txt',
                    label2id)
            else:
                round_dataset = WOS4RoundDataset(
                    f'../dataset/wos/entailment_data/wos_{args.shot}_shot_{args.round}/split/{round_num}/{split}.txt',
                    label2id)
        elif args.dataset == 'IFS':
            if args.shot == 0:
                round_dataset = IFSDataset(
                    f'../dataset/IFS/IFS_1_shot_{args.round}/split/{round_num}/{split}.txt',
                    args.no_label_name)
            else:
                round_dataset = IFSDataset(
                    f'../dataset/IFS/IFS_{args.shot}_shot_{args.round}/split/{round_num}/{split}.txt',
                    args.no_label_name)
        elif args.dataset == 'cad':
            round_dataset = CADDataset(
                f'../dataset/cad/cad_{args.shot}_shot_4/split/{round_num}/{split}.txt')
        else:
            round_dataset = ReutersDataset(
                f'../dataset/wos/entailment_data/reuters_{args.shot}_shot/split/{round_num}/{split}.txt')
        # print(f'{round_num} {split} dataset loaded!')
        # print(len(round_dataset))
        data_dict[split] = DataLoader(round_dataset, batch_size=1, shuffle=False)
# print(len(round_data_dict))
if args.graphrag:
    batch_size = 1
else:
    batch_size = 64

sample_text = ('this paper presents a novel method for the analysis of nonlinear financial and economic systems. '
               'the modeling approach integrates the classical concepts of state space representation and time series regression. '
               'the analytical and numerical scheme leads to a parameter space representation that constitutes a valid alternative to represent the dynamical behavior. '
               'the results reveal that business cycles can be clearly revealed, while the noise effects common in financial indices can '
               'elegantly be filtered out of the results.')

# print(sample_text)
current_labels = []
cleaned_corpus = []
label_desc_short = {}
current_data = []
for round_num, data_dict in round_data_dict.items():
    current_labels.append(round_label_dict[round_num])
    if context != 'none' and not args.no_label_name:
        # print(round_label_dict)
        label_desc = desc_label(GORAG, round_label_dict[round_num], short_desc=True)
        # print(label_desc)
        if not label_desc:
            # print(round_label_dict[round_num])
            raise ValueError('No label description found!')
        messages = [
            {"role": "system", "content": "You are an agent classifying texts into different classes. "
                                          f"Depending on their contents, you need to classify them strictly "
                                          f"into one of the following classes, separated by , : {labels}. "
                                          f"To help you better understand these classes, here are some descriptions "
                                          f"of each class: {'; '.join(list(label_desc.values()))}. "
             # f". For example, for this sample text: {sample_text}, the class should be: 'ece'."
             # "You will learn from some examples first and then classify the rest."
             }
        ]
    else:
        label_desc = []
        messages = [
            {"role": "system", "content": "You are an agent classifying texts into different classes. "
                                          f"Depending on their contents, you need to classify them strictly "
                                          f"into one of the following classes, separated by , : {labels}. "
             # f". For example, for this sample text: {sample_text}, the class should be: 'ece'."
             # "You will learn from some examples first and then classify the rest."
             }
        ]

    bad_case = []
    good_case = []
    count = 0
    NO_KG_MATCHING = 0
    # messages_t0.append({"role": "user", "content": "Who are you? What can you do?"})
    current_round_candidate_labels = list(chain(*current_labels))
    if args.graphrag:
        if args.no_label_name:
            label_keywords_current_round = GORAG.generate_keywords(data_dict['train'])
            label_desc_short.update(label_keywords_current_round)
        else:
            # label_desc_short_current_round = desc_label(GORAG, round_label_dict[round_num], True)
            label_desc_short.update(label_desc)
        corpus_length_before_idx = len(cleaned_corpus)
        # if not os.path.exists(
        #         f'./graph/{args.dataset}_{args.LLM}_{args.edge_weighting}_{round_num}_{args.round}_{args.graph_filter}_{args.online_index}.json') or args.online_index != 'none':
        index_start_time = time.time()
        if not args.no_label_name:
            for raw_desc in label_desc.values():
                if isinstance(raw_desc, str):
                    clean_corpus_list = re.findall(r'(?u)\b\w\w+\b', raw_desc)
                    cleaned_corpus.append(str(' '.join(clean_corpus_list)))
        if not args.index_label_only:
            training_text_loader = data_dict['train']
            GORAG.index_training_text(training_text_loader, cleaned_corpus, args)
            training_text_length = len(training_text_loader)
        else:
            training_text_length = 0
        if not args.no_label_name:
            if args.index_label_only:
                GORAG.get_tfidf_vectorizer(cleaned_corpus)
            for label_desc_idx, label_desc_key in enumerate(
                    tqdm.tqdm(label_desc.keys(), desc=f'Graph Index {round_num}')):
                label_desc_value = label_desc[label_desc_key]
                label = label_desc_value.split(':')[0]
                desc = ':'.join(label_desc_value.split(':')[1:])
                _ = GORAG.LLM_Construct(label=label,
                                        label_desc=desc,
                                        args=args,
                                        corpus=cleaned_corpus,
                                        text_idx=corpus_length_before_idx + training_text_length + label_desc_idx,
                                        candidate_labels=current_round_candidate_labels)
        GORAG.print_graph()
        index_end_time = time.time()
        print(f'Indexing time cost for {round_num} is: {(index_end_time - index_start_time) / 60} mins')
    if args.renew_graph:
        GORAG.renew_graph()
    # for message in messages:
    if not args.no_label_name:
        response, test_res_score = GORAG.chat(messages, max_new_token)
    # Done before here
    current_data.append(data_dict['test'])
    start_time = time.time()
    online_index = True
    if not args.no_test:
        label_list_current_round = []
        ans_list_current_round = []
        for round_idx, current_round_test_data in enumerate(current_data[::-1]):
            GORAG.total_keyword = 0
            current_round_name = f'n{len(current_data) - round_idx}'
            if not online_index:
                continue

            ans_list, total_res, truth_label, bad_case, good_case, label_list = test(
                test_loader=current_round_test_data,
                gorag_model=GORAG,
                round_num=round_num,
                # candidate_label=current_labels[round_idx],
                candidate_label=current_round_candidate_labels,
                # label_desc=label_desc,  # Label_desc or current round
                # max_desc_label=args.max_desc_label,
                label_desc_short=label_desc_short,  # Short label_desc or current round
                online_index_round=online_index
            )
            online_index = False

            end_time = time.time()
            if args.llm_graph_filter:
                GORAG.llm_graph_filter(current_round_candidate_labels)
            if args.graph_filter == 'unrelated':
                GORAG.print_filtered_graph()
            acc = accuracy_score(y_true=label_list, y_pred=ans_list)
            prec = precision_score(y_true=label_list, y_pred=ans_list, average='weighted', zero_division=0)
            recall = recall_score(y_true=label_list, y_pred=ans_list, average='weighted', zero_division=0)
            f1 = f1_score(y_true=label_list, y_pred=ans_list, average='weighted', zero_division=0)
            prec_macro = precision_score(y_true=label_list, y_pred=ans_list, average='macro', zero_division=0)
            recall_macro = recall_score(y_true=label_list, y_pred=ans_list, average='macro', zero_division=0)
            f1_macro = f1_score(y_true=label_list, y_pred=ans_list, average='macro', zero_division=0)
            prec_micro = precision_score(y_true=label_list, y_pred=ans_list, average='micro', zero_division=0)
            recall_micro = recall_score(y_true=label_list, y_pred=ans_list, average='micro', zero_division=0)
            f1_micro = f1_score(y_true=label_list, y_pred=ans_list, average='micro', zero_division=0)
            print(f'The accuracy for {round_num}, {current_round_name} is: ', acc)
            print(f'The weighted precision for {round_num}, {current_round_name} is: ', prec)
            print(f'The weighted recall for {round_num}, {current_round_name} is: ', recall)
            print(f'The weighted f1 for {round_num}, {current_round_name} is: ', f1)
            print(f'The macro precision for {round_num}, {current_round_name} is: ', prec_macro)
            print(f'The macro recall for {round_num}, {current_round_name} is: ', recall_macro)
            print(f'The macro f1 for {round_num}, {current_round_name} is: ', f1_macro)
            print(f'The micro precision for {round_num}, {current_round_name} is: ', prec_micro)
            print(f'The micro recall for {round_num}, {current_round_name} is: ', recall_micro)
            print(f'The micro f1 for {round_num}, {current_round_name} is: ', f1_micro)
            print(f'Average token usage: {GORAG.token_usage / len(current_round_test_data)}')
            print(f'Average keyword per query: {GORAG.total_keyword / len(current_round_test_data)}')
            GORAG.reset_token_usage()
            label_list_current_round.extend(label_list)
            ans_list_current_round.extend(ans_list)
            GORAG.print_time_usage()
            print(
                f'The inference time cost for {round_num}, {current_round_name} is: {(end_time - start_time) / 60} mins')
            if not os.path.exists(f'./llama_badcase_{round_num}_{args.round}'):
                os.makedirs(f'llama_badcase_{round_num}_{args.round}')
            try:
                with open(f'./llama_badcase_{round_num}_{args.round}/bad_case_{exp_name}.json', mode='w') as file:
                    file.write(json.dumps(bad_case, indent=4))
                with open(f'./llama_badcase_{round_num}_{args.round}/good_case_{exp_name}.json', mode='w') as file:
                    file.write(json.dumps(good_case, indent=4))
            except:
                continue
            if args.online_index == 'none':
                break
        with open(f'../result/{round_num}_{exp_name}_ans_all.json', 'w') as file:
            json.dump({'gt': label_list_current_round, 'ans': ans_list_current_round}, file, indent=2)
        # raise ValueError
if args.graph_filter == 'unrelated':
    print_str = GORAG.print_filtered_graph()
else:
    print_str = GORAG.print_graph()
print(exp_name)
