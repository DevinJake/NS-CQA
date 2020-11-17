# -*- coding: utf-8 -*-
# @Time    : 2019/5/8 15:51
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io

# -*- coding: utf-8 -*-
# @Time    : 2019/5/7 14:35
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io

# -*- coding: utf-8 -*-
# @Time    : 2019/4/1 14:12
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io

# -*- coding: utf-8 -*-
# @Time    : 2019/3/11 21:39
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io

# -*- coding:utf-8 -*-

import copy
import sys
import time

sys.path
from Preprocess.load_qadata import load_qadata, getQA_by_state
from .symbolics import Symbolics
import logging
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='/data/zjy/comp_count_appro_auto.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(message)s'
                    #日志格式
                    )
continue_num = 0
class Node(object):
    def __init__(self, value=None):
        self.value = value  # 节点值
        self.child_list = []  # 子节点列表

    def add_child(self, node):
        self.child_list.append(node)


def init():
    '''
    初始化生成式规则
    '''

    symbolics = [Node('A' + str(i)) for i in range(0, 17)]

    symbolics[0].add_child(symbolics[16])  # a17 = a2
    symbolics[16].add_child(symbolics[2])
    symbolics[2].add_child(symbolics[15])


    return symbolics[0]


def find_all_paths(node, path, paths):
    path += node.value + ">"

    if (node == None or len(node.child_list) == 0):
        temp_path = path
        paths.append(temp_path)
        return
    for child in node.child_list:
        find_all_paths(child, path, paths)


def auto_generate():
    root = init()
    paths = []
    find_all_paths(root, "", paths)
    return [path.replace(">", " ").strip().split(" ")[1:] for path in paths]


def cal_precesion(orig_answer, answer_e, orig_answer_entities, cal_answer):

    if not cal_answer: return False
    if(type(cal_answer) == int):
        return cal_answer == len(orig_answer_entities) and sorted(answer_e) == sorted(orig_answer_entities)
    if (type(cal_answer) == bool):
        return cal_answer == orig_answer
    if (type(cal_answer) == dict):
        temp = []
        for key, value in cal_answer.items():
            if (value):
                temp.extend(list(value))
        cal_answer = temp

    count = 0
    for e in cal_answer:
        if (e in orig_answer_entities):
            count += 1
    if len(orig_answer_entities) != 0:
        return count == (len(orig_answer_entities)) == len(cal_answer)
    return False


def auto_test():
    qa_set = load_qadata("/data/zjy/preprocessed_data_10k/train")

    qa_map = getQA_by_state(qa_set)

    symbolic_seqs = auto_generate()
    a = 0
    for qa in qa_map['Comparative Reasoning (Count) (All)\n']:

        context = qa['context'].replace("\n", "").strip()
        context_utterance = qa['context_utterance'].replace("\n", "")
        if ("around" not in context_utterance and "approximately" not in context_utterance):
            # print(context_utterance)
            continue
        context_entities = qa['context_entities'].replace("\n", "").split("|")
        context_relations = qa['context_relations'].replace("\n", "").split("|")
        context_types = qa['context_types'].replace("\n", "").split("|")
        context_ints = qa['context_ints'].replace("\n", "")
        context_relations.extend(['-' + r for r in context_relations])
        response_entities = qa['response_entities'].replace("\n", "").split("|")
        orig_response = qa['orig_response'].replace("\n", "")
        logging.info(str(a)+" "+ context_utterance)
        if "" in context_entities: context_entities.remove("")
        print (a,context_utterance)
        start_time = time.time()
        flag = 0
        a += 1
        if a < continue_num:
            continue
        for seq in symbolic_seqs:
            print (seq)
            seq_with_param = {i: [] for i in range(len(seq))}
            for i in range(len(seq)):
                symbolic = seq[i]
                if (int(symbolic[1:]) in [1, 8, 9, 10]):
                    for e in context_entities:
                        for r in context_relations:
                            for t in context_types:
                                seq_with_param[i].append({symbolic: (e, r, t)})

                if (int(symbolic[1:]) in [2,16]):
                    for et in context_types:
                        for r in context_relations:
                            for t in context_types:
                                seq_with_param[i].append({symbolic: (et, r, t)})
                                # print symbolic,e,r,t
                # if (int(symbolic[1:]) in [12,13,14,15] and context_ints != ""):
                #     for N in [int(n) for n in context_ints.split()]:
                #         seq_with_param[i].append({symbolic: (str(N), '', '')})

                if (int(symbolic[1:]) in [15]):
                    for e in context_entities:
                        seq_with_param[i].append({symbolic: (e, '', '')})

                if (int(symbolic[1:]) in [4,5,11]):
                    seq_with_param[i].append({symbolic: ('', '', '')})
                    seq_with_param[i].append({symbolic: ('&', '', '')})
                    seq_with_param[i].append({symbolic: ('-', '', '')})
                    seq_with_param[i].append({symbolic: ('|', '', '')})

            if (len(seq_with_param) == 3):
                for sym1 in seq_with_param[0]:
                    if flag == 4:
                        break
                    for sym2 in seq_with_param[1]:
                        if flag == 4:
                            break
                        for sym3 in seq_with_param[2]:
                            if flag == 4:
                                break
                            #print (flag)
                            sym_seq = [sym1, sym2, sym3]
                            symbolic_exe = Symbolics(sym_seq)
                            answer = symbolic_exe.executor()

                            #print(sym_seq)
                            Uset = list(set(answer).intersection(set(response_entities)))
                            #print(Uset[:10])
                            if len(Uset) > len(response_entities)/2:

                                flag += 1
                                logging.info(sym_seq)
                                print(sym_seq,time.time())


auto_test()
