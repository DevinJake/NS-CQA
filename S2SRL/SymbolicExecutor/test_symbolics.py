# -*- coding: utf-8 -*-
# @Time    : 2019/1/19 17:58
import os

import random
import time
#from urllib import urlencode
from urllib.parse import urlencode

import requests

from Model.seq2seq import Seq2Seq
from Preprocess.load_qadata import load_qadata
from Preprocess.question_parser import QuestionParser
from . import symbolics
from params import get_params




def test_sparql(self,e='Q148',r = 'P17',t = 'Q4022'):
    answer_dict = {}
    anser_values = []
    sparql = {"query": "SELECT ?river WHERE { \
                                ?river wdt:" + r + " wd:"+ e +". \
                                ?river wdt:P31  wd:"+ t + ". \
                           }",
              "format" : "json",
              }
    # print sparql
    sparql =  urlencode(sparql)
    print(sparql)
    url = 'https://query.wikidata.org/sparql?'+sparql
    r = requests.get(url)
    #print r.json()["results"]
    for e in r.json()["results"]["bindings"]:
        entity =  e["river"]["value"].split("/")[-1]
        anser_values.append(entity)
    answer_dict[e] = anser_values

def test_select(self):
    params = get_params("/data/zjy/csqa_data", "/home/zhangjingyao/preprocessed_data_10k")

    # ls = LuceneSearch(params["lucene_dir"])
    # 读取知识库
    # try:
    #     print("loading...")
    #     wikidata = pickle.load(open('/home/zhangjingyao/data/wikidata.pkl','rb'))
    #     print("loading...")
    #     item_data = pickle.load(open('/home/zhangjingyao/data/entity_items','rb'))
    #     print("loading...")
    #     prop_data = None
    #     print("loading...")
    #     child_par_dict = pickle.load(open('/home/zhangjingyao/data/type_kb.pkl','rb'))
    # except:
    # wikidata, item_data, prop_data, child_par_dict = load_wikidata(params["wikidata_dir"])# data for entity ,property, type

    # 读取qa文件集
    qa_set = load_qadata("/home/zhangjingyao/preprocessed_data_10k/demo")
    question_parser = QuestionParser(params, True)

    f = open("log.txt", 'w+')
    for qafile in qa_set.itervalues():
        for qid in range(len(qafile["context"])):
            # 得到一个qa数据
            q = {k:v[qid] for k,v in qafile.items()}

            # 解析问句
            qstring = q["context_utterance"]
            entities = question_parser.getNER(q)
            relations = question_parser.getRelations(q)
            types = question_parser.getTypes(q)

            # 得到操作序列
            states = random.randint(1,18) # 随机生成操作序列
            seq2seq = Seq2Seq()
            symbolic_seq = seq2seq.simple(qstring,entities,relations,types, states)

            # 符号执行
            time_start = time.time()
            symbolic_exe = symbolics.Symbolics(symbolic_seq)
            answer = symbolic_exe.executor()

            print("answer is :", answer)
            if (type(answer) == dict):
                for key in answer:
                    print([v for v in answer[key]])

            time_end = time.time()
            print('time cost:', time_end - time_start)
            print("--------------------------------------------------------------------------------")


    print(0)
# #wikidata, reverse_dict, prop_data, child_par_dict, wikidata_fanout_dict = load_wikidata(param["wikidata_dir"])

def test_folder(fpath):
    # 读取qa文件集
    for root, dirnames, filenames in os.walk(fpath):
        print(root)
        filenames = [ff for ff in filenames if not ff.endswith("result.txt")]
        for f in  filenames:
            time_start = time.time()
            test_file(root, f)
            time_end = time.time()
            print (f, 'time cost:', time_end - time_start)

def test_file(root, f):
    params = get_params("/data/zjy/csqa_data", "/home/zhangjingyao/preprocessed_data_10k")
    qa_path = root + f
    qa_file = open(qa_path)
    qa_result = open(qa_path[:-4] + "_result.txt", "w+")
    qa_result.truncate()
    question_parser = QuestionParser(params, True)
    sym_seq = []
    flag = 0
    qa_id = 0
    for line in qa_file:
        if line.startswith("symbolic_seq.append"):
            flag = 1
            key = line[line.find("{") + 1:line.find('}')].split(':')[0].replace('\"', '').strip()
            val = line[line.find("{") + 1:line.find('}')].split(':')[1].strip()
            val = val.replace('[', '').replace(']', '').replace("\'", "").split(',')

            sym_seq.append({key: val})
        if line.startswith("response_entities"):
            count = 0

            answer_entities = line.replace("response_entities:", '').strip().split("|")
        if line.startswith("orig_response"):
            orig_response = line.replace("orig_response:", '').strip()

        if (line.startswith("-----------") and flag == 1):
            time_start = time.time()
            symbolic_exe = symbolics.Symbolics(sym_seq)
            answer = symbolic_exe.executor()

            if (type(answer) == dict):
                temp = []
                for key,value in answer.items():
                    if(value):
                        temp.extend(list(value))
                answer = temp

            elif type(answer) == type([]) or type(answer) == type(set([])):
                answer = sorted((list(answer)))
            elif type(answer) == int:
                answer = [answer]
            else:
                answer = [answer]
            time_end = time.time()

            if (orig_response == "None") and answer == []:
                answer = ['None']
                answer_entities = ['None']

            if len(answer) > 500:
                print(("answer is :", list(answer)[:500]), end="", file=qa_result)
            else:
                print(("answer is :", list(answer)), end="", file=qa_result)
            print(('time cost:', time_end - time_start), end="", file=qa_result)
            for e in answer_entities:
                if (e in answer):
                    count += 1

            print(("orig:", len(answer_entities), "answer:", len(answer), "right:", count), end="", file=qa_result)
            print('===============================', end="", file=qa_result)
            flag = 0
            sym_seq = []

        if ("response") in line or line.startswith("context_utterance") or line.replace("\n", "").isdigit() or "state" in line:
            print((qa_result, line,), end="", file=qa_result)

if __name__ == "__main__":
    #test_folder("/home/zhangjingyao/demoqa/")
    fname = "16logical_diff_single.txt"
    test_file("/data/zjy/demoqa2/", fname)