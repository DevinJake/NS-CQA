# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 21:02
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io

# coding:utf-8
'''Get all questions, annotated actions, entities, relations, types together in JSON format.
'''

import json
from symbolics import Symbolics

def transMask2Action(state):
    with open("CSQA_ANNOTATIONS_test.json", 'r') as load_f, open("final_predict.actions", 'r') as predict_actions \
            , open("FINAL_test.question", 'r') as RL_test:
        count = 1
        load_dict = json.load(load_f)
        num = 0
        total_precision = 0
        total_recall = 0
        total_right_count = 0
        total_answer_count = 0
        total_response_count = 0
        bool_right_count = 0
        count_right_count = 0
        for x, y in zip(predict_actions, RL_test):
            action = x.strip().split(":")[1]
            id = y.strip().split()[0]

            if (id.startswith(state)):
                num += 1
                entity_mask = load_dict[id]["entity_mask"] if load_dict[id]["entity_mask"] != None else {}
                relation_mask = load_dict[id]["relation_mask"] if load_dict[id]["relation_mask"] != None else {}
                type_mask = load_dict[id]["type_mask"] if load_dict[id]["type_mask"] != None else {}
                response_entities = load_dict[id]["response_entities"].strip() if load_dict[id][
                                                                                      "response_entities"] != None else ""
                response_entities = response_entities.strip().split("|")
                orig_response = load_dict[id]["orig_response"].strip() if load_dict[id]["type_mask"] != None else ""
                entity_mask.update(relation_mask)
                entity_mask.update(type_mask)
                new_action = list()
                for act in action.split():
                    for k, v in entity_mask.items():
                        if act == v:
                            act = k
                            break
                    new_action.append(act)
                print("{0}:{1}".format(id, action))
                #print(" ".join(new_action))
                symbolic_seq = list2dict(new_action)
                # symbolic_seq.append({"A11":["","",""]})### A11
                # if state.startswith("Verification(Boolean)(All)"):
                #     symbolic_seq[-1] = {"A3":["","",""]} if not symbolic_seq[-1].has_key("A3") else symbolic_seq[-1]### A3
                # if state.startswith("QuantitativeReasoning(Count)(All)") or state.startswith("ComparativeReasoning(Count)(All)"):
                #     symbolic_seq[-1] = {"A11": ["", "", ""]} if not symbolic_seq[-1].has_key("A11") else symbolic_seq[-1]
                symbolic_exe = Symbolics(symbolic_seq)
                answer = symbolic_exe.executor()

                if state.startswith("QuantitativeReasoning(Count)(All)") or state.startswith("ComparativeReasoning(Count)(All)"):
                    print(symbolic_seq)
                    print("answer::orig_response", answer, orig_response)

                    if orig_response.isdigit() and answer == int(orig_response):
                        count_right_count += 1
                        print("count_right_count+1")
                    else:
                        import re
                        orig_response = re.findall(r"\d+\.?\d*", orig_response)
                        orig_response = sum([int(i) for i in orig_response])
                        if answer == orig_response:
                            count_right_count += 1
                            print("count_right_count+1")
                if state.startswith("Verification(Boolean)(All)"):
                    if answer == True:
                        answer = "YES"
                    if answer == False:
                        answer = "NO"
                    if answer == orig_response:
                        bool_right_count+=1
                if (type(answer) == dict):
                    temp = []
                    for key, value in answer.items():
                        if (value):
                            temp.extend(list(value))
                    answer = temp

                elif type(answer) == type([]) or type(answer) == type(set([])):
                    answer = sorted((list(answer)))
                elif type(answer) == int:
                    answer = [answer]
                else:
                    answer = [answer]

                right_count = 0
                for e in response_entities:
                    if (e in answer):
                        right_count += 1
                total_right_count += right_count
                total_answer_count += len(answer)
                total_response_count += len(response_entities)
                precision = right_count / float(len(answer)) if len(answer) != 0 else 0
                total_precision += precision
                recall = (right_count / float(len(response_entities))) if len(response_entities) != 0 else 0
                total_recall += recall
                print("orig:", len(response_entities), "answer:", len(answer), "right:", right_count)
                print("Precision:", precision),
                print("Recall:", recall)
                print('===============================')
            # print answer
        print("bool_right_count", bool_right_count)
        print("count_right_count", count_right_count)
        print("total_num::total_right::total_answer::total_response", num, total_right_count, total_answer_count, total_response_count)
        mean_pre = total_precision / num
        mean_recall = total_recall / num
        mean_pre2 = float(total_right_count) / total_answer_count
        mean_recall2 = float(total_right_count) / total_response_count
        print(state + "::mean_pre::mean_recall", mean_pre, mean_recall)
        print(state + "::mean_pre2::mean_recall2", mean_pre2, mean_recall2)
        print("++++++++++++++")


def list2dict(list):
    final_list = []
    temp_list = []
    new_list = []
    for a in list:
        if (a == "("):
            new_list = []
            continue
        if (a == ")"):
            if ("-" in new_list):
                new_list[new_list.index("-") + 1] = "-" + new_list[new_list.index("-") + 1]
                new_list.remove("-")
            if (new_list == []):
                new_list = ["", "", ""]
            if (len(new_list) == 1):
                new_list = [new_list[0], "", ""]
            if ("&" in new_list):
                new_list = ["&", "", ""]
            if ("-" in new_list):
                new_list = ["-", "", ""]
            if ("|" in new_list):
                new_list = ["|", "", ""]
            temp_list.append(new_list)
            continue
        if not a.startswith("A"):
            if a.startswith("E"):  a = "Q17"
            if a.startswith("T"):  a = "Q17"
            new_list.append(a)

    i = 0
    for a in list:
        if (a.startswith("A")):
            final_list.append({a: temp_list[i]})
            # temp_dict[a] = temp_list[i]
            i += 1

    return final_list

def list2dict_webqsp(list):
    final_list = []
    temp_list = []
    new_list = []
    for a in list:
        if (a == "("):
            new_list = []
            continue
        if (a == ")"):
            if ("-" in new_list):
                new_list[new_list.index("-") + 1] = "-" + new_list[new_list.index("-") + 1]
                new_list.remove("-")
            if (new_list == []):
                new_list = ["", "", ""]
            if (len(new_list) == 1):
                new_list = [new_list[0], "", ""]
            if ("&" in new_list):
                new_list = ["&", "", ""]
            if ("-" in new_list):
                new_list = ["-", "", ""]
            if ("|" in new_list):
                new_list = ["|", "", ""]
            temp_list.append(new_list)
            continue
        if not a.startswith("A"):
            # if a.startswith("E"):  a = "Q17"
            # if a.startswith("T"):  a = "Q17"
            new_list.append(a)

    i = 0
    for a in list:
        if (a.startswith("A")):
            final_list.append({a: temp_list[i]})
            # temp_dict[a] = temp_list[i]
            i += 1

    return final_list


if __name__ == "__main__":
    # QuantitativeReasoning(Count)(All)
    # QuantitativeReasoning(All)
    # ComparativeReasoning(Count)(All)
    # ComparativeReasoning(All)
    # Verification(Boolean)(All)
    # SimpleQuestion(Direct)
    # LogicalReasoning(All)
    state_list = ["SimpleQuestion(Direct)","Verification(Boolean)(All)","QuantitativeReasoning(Count)(All)","QuantitativeReasoning(All)",
                  "ComparativeReasoning(Count)(All)",
                  "ComparativeReasoning(All)","LogicalReasoning(All)"]
    # for state in state_list:
    #     transMask2Action(state)
    transMask2Action("ComparativeReasoning(Count)(All)")
# print (calc_pression('D:/study/nmt/nmt/nmt/csqa/data/dev.action', 'D:/study/nmt/nmt/nmt/csqa/data/nmt_model/shuffle52k/output_dev', 'D:/study/nmt/nmt/nmt/csqa/data/nmt_model/shuffle52k'))




