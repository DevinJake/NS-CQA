# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 21:02
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io

# coding:utf-8
'''Get all questions, annotated actions, entities, relations, types together in JSON format.
'''

import json
from .symbolics_webqsp import Symbolics_WebQSP
from transform_util import transformBooleanToString, list2dict
import logging
log1 = logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='../../data/auto_QA_data/test_result/1.0%_sample_testdataset_result_s2s.log',
                    filemode='w',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(message)s'
                    #日志格式
                    )

def transMask2Action():
    with open("../../data/auto_QA_data/CSQA_ANNOTATIONS_test.json", 'r') as load_f, open("../../data/saves/crossent_even_1%/sample_final_predict.actions", 'r') as predict_actions \
            , open("../../data/auto_QA_data/mask_test/SAMPLE_FINAL_test.question", 'r') as RL_test:
        linelist = list()
        load_dict = json.load(load_f)
        num = 0
        total_precision = 0
        total_recall = 0
        total_jaccard = 0
        total_f1 = 0
        total_right_count = 0
        total_answer_count = 0
        total_response_count = 0
        bool_right_count = 0
        count_right_count = 0
        for x, y in zip(predict_actions, RL_test):
            action = x.strip().split(":")[1]
            id = y.strip().split()[0]

            if True:
                num += 1
                entity_mask = load_dict[id]["entity_mask"] if load_dict[id]["entity_mask"] != None else {}
                relation_mask = load_dict[id]["relation_mask"] if load_dict[id]["relation_mask"] != None else {}
                type_mask = load_dict[id]["type_mask"] if load_dict[id]["type_mask"] != None else {}
                response_entities = load_dict[id]["response_entities"].strip() if load_dict[id][
                                                                                      "response_entities"] != None else ""
                response_entities = response_entities.strip().split("|")
                orig_response = load_dict[id]["orig_response"].strip() if load_dict[id]["orig_response"] != None else ""
                # Update(add) elements in dict.
                entity_mask.update(relation_mask)
                entity_mask.update(type_mask)
                new_action = list()
                # Default separator of split() method is any whitespace.
                for act in action.split():
                    for k, v in entity_mask.items():
                        if act == v:
                            act = k
                            break
                    new_action.append(act)
                print("{0}".format(num))
                '''print("{0}: {1}->{2}".format(num, id, action))'''
                logging.info("%d: %s -> %s", num, id, action)
                #print(" ".join(new_action))
                symbolic_seq = list2dict(new_action)
                symbolic_exe = Symbolics_WebQSP(symbolic_seq)
                answer = symbolic_exe.executor()

                right_count = 0
                for e in response_entities:
                    if (e in answer):
                        right_count += 1
                total_right_count += right_count
                total_answer_count += len(answer)
                total_response_count += len(response_entities)

                # precision
                precision = right_count / float(len(answer)) if len(answer) != 0 else 0
                total_precision += precision

                # recall
                recall = (right_count / float(len(response_entities))) if len(response_entities) != 0 else 0
                total_recall += recall

                # jaccard
                intersec = set(response_entities).intersection(set(answer))
                union = set([])
                union.update(response_entities)
                union.update(answer)
                jaccard = float(len(intersec)) / float(len(answer)) if len(answer) != 0 else 0
                total_jaccard += jaccard

                # f1
                f1 = float(len(intersec))/float(len(response_entities)) if len(response_entities) != 0 else 0
                total_f1 += f1

                '''print("orig:", len(response_entities), "answer:", len(answer), "right:", right_count)
                print("Precision:", precision),
                print("Recall:", recall)
                print("Recall:", jaccard)
                print('===============================')'''
                logging.info("orig:%d, answer:%d, right:%d", len(response_entities), len(answer), right_count)
                logging.info("Precision:%f", precision)
                logging.info("Recall:%f", recall)
                logging.info("Jaccard:%f", jaccard)
                logging.info("F1:%f", f1)
                logging.info("============================")

        # print answer
        mean_pre = total_precision / num
        mean_recall = total_recall / num
        mean_jaccard = total_jaccard / num
        mean_f1 = total_f1 / num
        mean_pre2 = float(total_right_count) / total_answer_count
        mean_recall2 = float(total_right_count) / total_response_count
        string_mean_pre = "state::mean_pre::mean_recall -> %s::%f::%f" %(state, mean_pre, mean_recall)
        string_mean_pre2 = "state::mean_pre2::mean_recall2 -> %s::%f::%f" %(state, mean_pre2, mean_recall2)
        print(string_mean_pre)
        print(string_mean_pre2)
        print("++++++++++++++")
        logging.info("state::mean_pre::mean_recall -> %s::%f::%f", state, mean_pre, mean_recall)
        logging.info("state::mean_pre2::mean_recall2 -> %s::%f::%f", state, mean_pre2, mean_recall2)
        logging.info("++++++++++++++")
        linelist.append(string_mean_pre + '\r\n')
        linelist.append(string_mean_pre2 + '\r\n')
        linelist.append('++++++++++++++\n\n')
        return linelist

if __name__ == "__main__":
    # QuantitativeReasoning(Count)(All)
    # QuantitativeReasoning(All)
    # ComparativeReasoning(Count)(All)
    # ComparativeReasoning(All)
    # Verification(Boolean)(All)
    # SimpleQuestion(Direct)
    # LogicalReasoning(All)
    linelist = list()
    fw = open('../../data/auto_QA_data/test_result/1%_sample_testdataset_result_s2s.txt', 'w', encoding="UTF-8")
    state_list = ["SimpleQuestion(Direct)","Verification(Boolean)(All)","QuantitativeReasoning(Count)(All)","QuantitativeReasoning(All)","ComparativeReasoning(Count)(All)","ComparativeReasoning(All)","LogicalReasoning(All)"]
    # state_list = ["Verification(Boolean)(All)"]
    linelist = transMask2Action()
    fw.writelines(linelist)
    fw.close()
# print (calc_pression('D:/study/nmt/nmt/nmt/csqa/data/dev.action', 'D:/study/nmt/nmt/nmt/csqa/data/nmt_model/shuffle52k/output_dev', 'D:/study/nmt/nmt/nmt/csqa/data/nmt_model/shuffle52k'))