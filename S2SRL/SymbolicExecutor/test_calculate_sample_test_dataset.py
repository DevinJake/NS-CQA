# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 21:02
# @Author  : Devin Hua

# coding:utf-8
# Get all questions, annotated actions, entities, relations, types together in JSON format.
import os
import json
from symbolics import Symbolics
from transform_util import transformBooleanToString, list2dict
import logging
import sys
dir = '../../data/auto_QA_data/test_result/'
os.makedirs(dir, exist_ok=True)
log = logging.basicConfig(level = logging.INFO,
                           filename ='../../data/auto_QA_data/test_result/crossent.log',
                           filemode ='w', format = '%(message)s')


def calc_single_sample(withint=False):
    r"""To get the result of the predicted action sequence.
            Args:
                qid: user should input the qid of the testing question.
                action: user should input the generated action for the question.

            Yields:
                testing result.

            Example::
                # >>> Input id:
                # >>> SimpleQuestion(Direct)4879
                # >>> Input action:
                # >>> 244: A1 ( ENTITY1 RELATION1 TYPE1 ) A9 ( ENTITY2 RELATION1 TYPE1 )
                # >>> answer:
                # >>> {'|': ['Q674182', 'Q1065', 'Q81299', 'Q7184', 'Q842490',
                'Q19771', 'Q37143', 'Q188822', 'Q1043527']}

            """
    if withint:
        json_path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_test_INT.json'
    else:
        json_path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_test.json'
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
        print('Dict loading is done!')
        while True:
            print("------------------------------")
            print("$$ represents exit.")
            print("Input id:")
            qid = sys.stdin.readline().strip()
            if qid == '$$':
                break
            print("Input action:")
            action = sys.stdin.readline().strip()
            if action == '$$':
                break
            action, id = action.strip().split(':')[1].strip(), qid.strip()
            print('question:')
            print(str(load_dict[id]["question"]))
            entity_mask = load_dict[id]["entity_mask"] \
                if load_dict[id]["entity_mask"] is not None else {}
            relation_mask = load_dict[id]["relation_mask"] \
                if load_dict[id]["relation_mask"] is not None else {}
            type_mask = load_dict[id]["type_mask"] \
                if load_dict[id]["type_mask"] is not None else {}
            int_mask = load_dict[id]["int_mask"] \
                if 'int_mask' in load_dict[id] else {}
            response_entities = load_dict[id]["response_entities"].strip() \
                if load_dict[id]["response_entities"] is not None else ""
            response_entities = response_entities.strip().split("|")
            orig_response = load_dict[id]["orig_response"].strip() \
                if load_dict[id]["orig_response"] is not None else ""
            # Update(add) elements in dict.
            entity_mask.update(relation_mask)
            entity_mask.update(type_mask)
            entity_mask.update(int_mask)
            new_action = list()
            # Default separator of split() method is any whitespace.
            for act in action.split():
                for k, v in entity_mask.items():
                    if act == v:
                        act = k
                        break
                new_action.append(act)
            symbolic_seq = list2dict(new_action)
            print('action sequence:')
            print(symbolic_seq)
            symbolic_exe = Symbolics(symbolic_seq)
            answer = symbolic_exe.executor()
            print('answer:')
            print(answer)
            print('number:')
            print(len(answer) if isinstance(answer, list) else 0)
            print('orig_response:')
            print(orig_response)
            print('response_entities:')
            print(response_entities)
            print('response_number:')
            print(len(response_entities) if isinstance(response_entities, list) else 0)

def transMask2Action(state, withint):
    if withint:
        json_path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_test_INT.json'
        question_path = '../../data/auto_QA_data/mask_test/FINAL_INT_test.question'
    else:
        json_path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_test.json'
        question_path = '../../data/auto_QA_data/mask_test/FINAL_test.question'
    with open(json_path, 'r') as load_f, \
            open("../../data/saves/crossent_1%_withINT_att=0_w2v=300/sample_final_int_predict.actions", 'r') as predict_actions, \
            open(question_path, 'r') as RL_test:
        linelist = list()
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

            if id.startswith(state):
                num += 1
                entity_mask = load_dict[id]["entity_mask"] \
                    if load_dict[id]["entity_mask"] is not None else {}
                relation_mask = load_dict[id]["relation_mask"] \
                    if load_dict[id]["relation_mask"] is not None else {}
                type_mask = load_dict[id]["type_mask"] \
                    if load_dict[id]["type_mask"] is not None else {}
                int_mask = load_dict[id]["int_mask"] \
                    if 'int_mask' in load_dict[id] else {}
                response_entities = load_dict[id]["response_entities"].strip() \
                    if load_dict[id]["response_entities"] is not None else ""
                response_entities = response_entities.strip().split("|")
                orig_response = load_dict[id]["orig_response"].strip() \
                    if load_dict[id]["orig_response"] is not None else ""
                # Update(add) elements in dict.
                entity_mask.update(relation_mask)
                entity_mask.update(type_mask)
                entity_mask.update(int_mask)
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
                # symbolic_seq.append({"A11":["","",""]})### A11
                # Modify with magic.
                # if state.startswith("Verification(Boolean)(All)"):
                #     symbolic_seq[-1] = {"A3":["","",""]} if not symbolic_seq[-1].has_key("A3") else symbolic_seq[-1]### A3
                # if state.startswith("QuantitativeReasoning(Count)(All)") or state.startswith("ComparativeReasoning(Count)(All)"):
                #     symbolic_seq[-1] = {"A11": ["", "", ""]} if not symbolic_seq[-1].has_key("A11") else symbolic_seq[-1]
                symbolic_exe = Symbolics(symbolic_seq)
                answer = symbolic_exe.executor()

                if state.startswith("QuantitativeReasoning(Count)(All)") or state.startswith("ComparativeReasoning(Count)(All)"):
                    '''print (symbolic_seq)
                    print ("%s::%s" %(answer, orig_response))'''
                    logging.info(symbolic_seq)
                    logging.info("answer:%s, orig_response:%s", answer, orig_response)

                    if orig_response.isdigit() and answer == int(orig_response):
                        count_right_count += 1
                        '''print ("count_right_count+1")'''
                        logging.info("count_right_count+1")
                    else:
                        import re
                        orig_response = re.findall(r"\d+\.?\d*", orig_response)
                        orig_response = sum([int(i) for i in orig_response])
                        if answer == orig_response:
                            count_right_count += 1
                            '''print ("count_right_count+1")'''
                            logging.info("count_right_count+1")
                # For boolean, the returned answer is a list.
                if state.startswith("Verification(Boolean)(All)"):
                    # To judge the returned answers are in dict format or boolean format.
                    if type(answer) == dict:
                        temp = []
                        if '|BOOL_RESULT|' in answer:
                            temp.extend(answer['|BOOL_RESULT|'])
                            answer = temp
                            answer_string = transformBooleanToString(answer)
                            if answer_string!='' and answer_string == orig_response:
                                bool_right_count += 1
                                '''print("bool_right_count+1")'''
                                logging.info("bool_right_count+1")
                    else:
                        if answer:
                            answer = "YES"
                        if not answer:
                            answer = "NO"
                        if answer == orig_response:
                            bool_right_count += 1
                            '''print("bool_right_count+1")'''
                            logging.info("bool_right_count+1")

                # To judge the returned answers are in dict format or boolean format.
                if type(answer) == dict:
                    temp = []
                    if '|BOOL_RESULT|' in answer:
                        temp.extend(answer['|BOOL_RESULT|'])
                    else:
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
                    if e in answer:
                        right_count += 1
                total_right_count += right_count
                total_answer_count += len(answer)
                total_response_count += len(response_entities)
                precision = right_count / float(len(answer)) if len(answer) != 0 else 0
                total_precision += precision
                recall = (right_count / float(len(response_entities))) if len(response_entities) != 0 else 0
                total_recall += recall
                '''print("orig:", len(response_entities), "answer:", len(answer), "right:", right_count)
                print("Precision:", precision),
                print("Recall:", recall)
                print('===============================')'''
                logging.info("orig:%d, answer:%d, right:%d", len(response_entities), len(answer), right_count)
                logging.info("Precision:%f", precision)
                logging.info("Recall:%f", recall)
                logging.info("============================")
            # print answer
        string_bool_right = "bool_right_count: %d" %bool_right_count
        string_count_right_count = "count_right_count: %d" %count_right_count
        string_total_num = "total_num::total_right::total_answer::total_response -> %d::%d::%d::%d" \
                           % (num, total_right_count, total_answer_count, total_response_count)
        print (string_bool_right)
        print (string_count_right_count)
        print (string_total_num)
        logging.info("bool_right_count:%d", bool_right_count)
        logging.info("count_right_count:%d", count_right_count)
        logging.info("total_num::total_right::total_answer::total_response -> %d::%d::%d::%d",
                     num, total_right_count, total_answer_count, total_response_count)
        linelist.append(string_bool_right + '\r\n')
        linelist.append(string_count_right_count + '\r\n')
        linelist.append(string_total_num + '\r\n')

        mean_pre = total_precision / num if num != 0 else 0.0
        mean_recall = total_recall / num if num != 0 else 0.0
        mean_pre2 = float(total_right_count) / total_answer_count if total_answer_count!=0 else 0.0
        mean_recall2 = float(total_right_count) / total_response_count if total_response_count!=0 else 0.0
        string_mean_pre = "state::mean_pre::mean_recall -> %s::%f::%f" % (state, mean_pre, mean_recall)
        string_mean_pre2 = "state::mean_pre2::mean_recall2 -> %s::%f::%f" % (state, mean_pre2, mean_recall2)
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

def calculate_RL_or_DL_result(file_path, withint):
    dir = '../../data/auto_QA_data/test_result/'
    path = dir+file_path+'.txt'
    os.makedirs(dir, exist_ok=True)
    linelist = list()
    fw = open(path, 'w', encoding="UTF-8")
    state_list = ["SimpleQuestion(Direct)", "Verification(Boolean)(All)", "QuantitativeReasoning(Count)(All)",
                  "QuantitativeReasoning(All)", "ComparativeReasoning(Count)(All)", "ComparativeReasoning(All)",
                  "LogicalReasoning(All)"]
    # state_list = ["Verification(Boolean)(All)"]
    for state in state_list:
        linelist += transMask2Action(state, withint)
    fw.writelines(linelist)
    fw.close()


if __name__ == "__main__":
    calculate_RL_or_DL_result('crossent', withint=True)
