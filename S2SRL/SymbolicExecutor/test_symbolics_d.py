# -*- coding: utf-8 -*-
# @Time     : 2019/05/03 20:21
# @Author   : Devin Hua
import sys
sys.path.append('..')
import os
import time
from symbolics import Symbolics

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
    qa_path = root + f
    qa_file = open(qa_path)
    qa_result = open(qa_path[:-4] + "_result.txt", "w")
    print("qa_result path is: %s" %(str(qa_result)))
    qa_result.truncate()
    sym_seq = []
    flag = 0
    qa_id = 0
    for line in qa_file:
        if line.startswith("symbolic_seq.append"):
            flag = 1
            key = line[line.find("{")+1: line.find('}')].split(':')[0].replace('\"', '').strip()
            val = line[line.find("{")+1: line.find('}')].split(':')[1].strip()
            val = val.replace('[', '').replace(']', '').replace("\'", "").split(',')

            sym_seq.append({key: val})
        if line.startswith("response_entities"):
            count = 0

            answer_entities = line.replace("response_entities:", '').strip().split("|")
        if line.startswith("orig_response"):
            orig_response = line.replace("orig_response:", '').strip()

        if (line.startswith("-----------") and flag == 1):
            time_start = time.time()
            # Actions execution.
            symbolic_exe = Symbolics(sym_seq)
            answer = symbolic_exe.executor()

            # To judge the returned answers are in dict format or boolean format.
            if (type(answer) == dict):
                temp = []
                if '|BOOL_RESULT|' in answer:
                    temp.extend(answer['|BOOL_RESULT|'])
                else:
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
                print(("answer is :", list(answer)[:500]), end='\n', file=qa_result)
            else:
                print(("answer is :", list(answer)), end='\n', file=qa_result)
            print(('time cost:', time_end - time_start), end='\n', file=qa_result)
            for e in answer_entities:
                if (e in answer):
                    count += 1

            print(("orig:", len(answer_entities), "answer:", len(answer), "right:", count), end='\n', file=qa_result)
            print('===============================', end='\n', file=qa_result)
            flag = 0
            sym_seq = []

        if ("response") in line or line.startswith("context_utterance") or line.replace("\n", "").isdigit() or "state" in line:
            print((qa_result, line,), end='\n', file=qa_result)

if __name__ == "__main__":
    #test_folder("/home/zhangjingyao/demoqa/")
    # fname = "Comparative_else.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "appro_quant_countOver_multi_et.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "appro_quant_atleast_single_et.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "appro_quant_atleast_multi_et.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "appro_quant_countOver_single_et.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "quant_countOver_single_et.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "quant_countOver_multi_et.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "Comparative_Count_else.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "train_bool_all.txt"
    # fname = "bool_test.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "quant_atleast_multi_et.txt"
    # test_file("../../data/demoqa2/", fname)
    # fname = "train_bool_all.txt"
    # test_file("../../data/demoqa2/", fname)
    fname = "train_quanti_all.txt"
    test_file("../../data/demoqa2/", fname)
    # fname = "train_count_all.txt"
    # test_file("../../data/demoqa2/", fname)
