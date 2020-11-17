# coding:utf-8
'''
Get all questions, annotated actions, entities, relations, types together in JSON format.
Get the training processDataset and test processDataset for seq2seq (one question to one action ).
Get the training processDataset and test processDataset for REINFORCE (one question to many actions).
Get the training processDataset and test processDataset for REINFORCE with True Reward (one question with one answer).
Make the processDataset with more even distribution as:
simple: 2K, logical: 2K, quantitative: 2K, count: 1K, bool: 20, comp:40, compcount: 40.
'''
from itertools import islice
import sys
import json
#Python codes to read the binary files.
import numpy as np
import random
SEED = 1988
LINE_SIZE = 100000

CATEGORY_SIZE = 2800
COMP_SIZE = 932
COMP_APPRO_SIZE = 1769
COMP_COUNT_SIZE = 554
COMP_COUNT_APPRO_SIZE = 1536
QUANTATIVE_SIZE = 1880
COUNT_SIZE = 733
BOOL_SIZE = 427
from random import shuffle

special_counting_characters = {'-','|','&'}
special_characters = {'(',')','-','|','&'}

# Get the training processDataset and test processDataset for seq2seq (one question to one action ).
def getTrainingDatasetForPytorch(percentage):

    path = '../../data/webquestionssp/RL_mask_even_' + percentage + '/PT_train.question'
    fwTrainQ = open(path, 'w', encoding="UTF-8")
    path = '../../data/webquestionssp/RL_mask_even_' + percentage + '/PT_train.action'
    fwTrainA = open(path, 'w', encoding="UTF-8")
    path = '../../data/webquestionssp/RL_mask_even_' + percentage + '/PT_test.question'
    fwTestQ = open(path, 'w', encoding="UTF-8")
    path = '../../data/webquestionssp/RL_mask_even_' + percentage + '/PT_test.action'
    fwTestA = open(path, 'w', encoding="UTF-8")
    # with open("../../data/webquestionssp/WebQSP_ANNOTATIONS_test.json", 'r', encoding="UTF-8") as load_f:
    with open("../../data/webquestionssp/WebQSP_ANNOTATIONS_subtest.json", 'r', encoding="UTF-8") as load_f:
        train_action_string_list, test_action_string_list, train_question_string_list, test_question_string_list = list(), list(), list(), list()
        dict_list = json.load(load_f)
        # random.seed(SEED)
        # random.shuffle(dict_list)
        for item in dict_list:
            print("item", item)
            key = list(dict(item).keys())[0]
            value = list(dict(item).values())[0]
            try:
                # print(item)
                # actions = eval(str(value['mask_action_sequence_list']))
                mask_action_sequence_list = value['mask_action_sequence_list']
                actions = eval(str(mask_action_sequence_list))
            except Exception:
                continue
            if len(actions) > 0:
                action_string = ''
                action = actions[0]
                for temp_dict in actions:
                    for temp_key, temp_value in temp_dict.items():
                        action_string += temp_key + ' ( '
                        for token in temp_value:
                            if '-' in token:
                                token = '- ' + token.replace('-','')
                            action_string += str(token) + ' '
                        action_string += ') '
                question_string = '<E> '
                entities = value['entity_mask']
                if len(entities) > 0:
                    for entity_key, entity_value in entities.items():
                        if str(entity_value) != '':
                            question_string += str(entity_value) + ' '
                question_string += '</E> <R> '
                relations = value['relation_mask']
                if len(relations) > 0:
                    for relation_key, relation_value in relations.items():
                        if str(relation_value) !='':
                            question_string += str(relation_value) + ' '
                question_string += '</R> <T> '
                types = value['type_mask']
                if len(types) > 0:
                    for type_key, type_value in types.items():
                        if str(type_value) !='':
                            question_string += str(type_value) + ' '
                question_string += '</T> '
                question_token = str(value['question']).lower().replace('?', '')
                question_token = question_token.replace(',', ' ')
                question_token = question_token.replace(':', ' ')
                question_token = question_token.replace('(', ' ')
                question_token = question_token.replace(')', ' ')
                question_token = question_token.replace('"', ' ')
                question_token = question_token.strip()
                question_string += question_token
                question_string = question_string.strip() + '\n'
                action_string = action_string.strip() + '\n'

                dict_temp = {}
                dict_temp.setdefault('q', str(key) + ' ' + question_string)
                dict_temp.setdefault('a', str(key) + ' ' + action_string)
                dict_list.append(dict_temp)

    random.seed(SEED+1)
    random.shuffle(dict_list)
    # train_size = int(len(dict_list) * 0.95)
    train_size = int(len(dict_list))
    print("train_size", train_size)
    for i, item in enumerate(dict_list):
        try:
            if i < train_size:
                train_action_string_list.append(item['a'])
                train_question_string_list.append(item['q'])
            elif train_size <= i:
                test_action_string_list.append(item['a'])
                test_question_string_list.append(item['q'])
        except Exception:
            continue
    fwTrainQ.writelines(train_question_string_list)
    fwTrainA.writelines(train_action_string_list)
    fwTestQ.writelines(test_question_string_list)
    fwTestA.writelines(test_action_string_list)
    fwTrainQ.close()
    fwTrainA.close()
    fwTestQ.close()
    fwTestA.close()
    print ("Getting SEQUENCE2SEQUENCE processDataset is done!")

# Get the training processDataset and test processDataset for REINFORCE (one question to many actions).
def getTrainingDatasetForRl(percentage):
    path = '../../data/auto_QA_data/mask_even_' + percentage + '/RL_train.question'
    fwTrainQ = open(path, 'w', encoding="UTF-8")
    path = '../../data/auto_QA_data/mask_even_' + percentage + '/RL_train.action'
    fwTrainA = open(path, 'w', encoding="UTF-8")
    path = '../../data/auto_QA_data/mask_even_' + percentage + '/RL_test.question'
    fwTestQ = open(path, 'w', encoding="UTF-8")
    path = '../../data/auto_QA_data/mask_even_' + percentage + '/RL_test.action'
    fwTestA = open(path, 'w', encoding="UTF-8")
    # fwNoaction = open('../../data/auto_QA_data/mask_even/no_action_question.txt', 'w', encoding="UTF-8")
    no_action_question_list = list()
    questionSet = set()
    actionSet = set()
    with open("../../data/auto_QA_data/WebQSP_ANNOTATIONS_test.json", 'r', encoding="UTF-8") as load_f:
        train_action_string_list, test_action_string_list, train_question_string_list, test_question_string_list = list(), list(), list(), list()
        dict_list = list()
        load_dict = json.load(load_f)
        list_of_load_dict = list(load_dict.items())
        random.seed(SEED+2)
        random.shuffle(list_of_load_dict)
        load_dict = dict(list_of_load_dict)
        count_dict = {'simple_': 0, 'logical_': 0, 'quantative_': 0, 'count_': 0, 'bool_': 0, 'comp_': 0, 'compcount_': 0, 'compcountappro_': 0, 'compappro_': 0}
        for key, value in load_dict.items():
            try:
                actions = eval(str(value['mask_action_sequence_list']))
            except SyntaxError:
                pass
            if len(actions) > 0:
                if 'simple_' in key and count_dict['simple_'] < CATEGORY_SIZE:
                    count_dict['simple_'] = count_dict['simple_'] + 1
                elif 'logical_' in key and count_dict['logical_'] < CATEGORY_SIZE:
                    count_dict['logical_'] = count_dict['logical_'] + 1
                elif 'quantative_' in key and count_dict['quantative_'] < QUANTATIVE_SIZE:
                    count_dict['quantative_'] = count_dict['quantative_'] + 1
                elif 'count_' in key and 'compcount_' not in key and count_dict['count_'] < COUNT_SIZE:
                    count_dict['count_'] = count_dict['count_'] + 1
                elif 'bool_' in key and count_dict['bool_'] < BOOL_SIZE:
                    count_dict['bool_'] = count_dict['bool_'] + 1
                elif 'comp_' in key and count_dict['comp_'] < COMP_SIZE:
                    count_dict['comp_'] = count_dict['comp_'] + 1
                elif 'compcount_' in key and count_dict['compcount_'] < COMP_COUNT_SIZE:
                    count_dict['compcount_'] = count_dict['compcount_'] + 1
                elif 'compcountappro_' in key and count_dict['compcountappro_'] < COMP_COUNT_APPRO_SIZE:
                    count_dict['compcountappro_'] = count_dict['compcountappro_'] + 1
                elif 'compappro_' in key and count_dict['compappro_'] < COMP_APPRO_SIZE:
                    count_dict['compappro_'] = count_dict['compappro_'] + 1
                else:
                    continue
                action_string_list = list()
                for action in actions:
                    action_string = ''
                    for temp_dict in action:
                        for temp_key, temp_value in temp_dict.items():
                            action_string += temp_key + ' ( '
                            for token in temp_value:
                                if '-' in token:
                                    token = '- ' + token.replace('-','')
                                action_string += str(token) + ' '
                            action_string += ') '
                    action_string = action_string.strip() + '\n'
                    action_string_list.append(action_string)
                question_string = '<E> '
                entities = value['entity_mask']
                if len(entities) > 0:
                    for entity_key, entity_value in entities.items():
                        if str(entity_value) != '':
                            question_string += str(entity_value) + ' '
                question_string += '</E> <R> '
                relations = value['relation_mask']
                if len(relations) > 0:
                    for relation_key, relation_value in relations.items():
                        if str(relation_value) !='':
                            question_string += str(relation_value) + ' '
                question_string += '</R> <T> '
                types = value['type_mask']
                if len(types) > 0:
                    for type_key, type_value in types.items():
                        if str(type_value) !='':
                            question_string += str(type_value) + ' '
                question_string += '</T> '
                question_token = str(value['question']).lower().replace('?', '')
                question_token = question_token.replace(',', ' ')
                question_token = question_token.replace(':', ' ')
                question_token = question_token.replace('(', ' ')
                question_token = question_token.replace(')', ' ')
                question_token = question_token.replace('"', ' ')
                question_token = question_token.strip()
                question_string += question_token
                question_string = question_string.strip() + '\n'

                question_tokens = question_string.strip().split(' ')
                question_tokens_set = set(question_tokens)
                questionSet = questionSet.union(question_tokens_set)

                action_withkey_list = list()
                if len(action_string_list) > 0:
                    for action_string in action_string_list:
                        action_tokens = action_string.strip().split(' ')
                        action_tokens_set = set(action_tokens)
                        actionSet = actionSet.union(action_tokens_set)
                        action_withkey_list.append(str(key) + ' ' + action_string)

                dict_temp = {}
                dict_temp.setdefault('q', str(key) + ' ' + question_string)
                dict_temp.setdefault('a', action_withkey_list)
                dict_list.append(dict_temp)
            elif len(actions) == 0:
                no_action_question_list.append(str(key) + ' ' + str(value['question']).lower().replace('?', '').strip() + '\n')

    random.seed(SEED+3)
    random.shuffle(dict_list)
    train_size = int(len(dict_list))
    # train_size = int(len(dict_list) * 0.95)
    for i, item in enumerate(dict_list):
        if i < train_size:
            for action_string in item.get('a'):
                train_action_string_list.append(action_string)
            train_question_string_list.append(item.get('q'))
        elif train_size <= i:
            for action_string in item.get('a'):
                test_action_string_list.append(action_string)
            test_question_string_list.append(item.get('q'))
    fwTrainQ.writelines(train_question_string_list)
    fwTrainA.writelines(train_action_string_list)
    fwTestQ.writelines(test_question_string_list)
    fwTestA.writelines(test_action_string_list)
    questionList = list()
    for item in questionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            questionList.append(temp)
    actionList = list()
    for item in actionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            actionList.append(temp)
    # fwNoaction.writelines(no_action_question_list)
    fwTrainQ.close()
    fwTrainA.close()
    fwTestQ.close()
    fwTestA.close()
    # fwNoaction.close()
    print ("Getting RL processDataset is done!")

# Get the training processDataset and test processDataset for REINFORCE with True Reward (one question with one answer).
def getTrainingDatasetForRlWithTrueReward(percentage, SIZE):
    path = '../../data/webqsp_data/RL_mask_even_' + percentage + '/RL_train_TR_sub_webqsp.question'
    fwTrainQ = open(path, 'w', encoding="UTF-8")
    path = '../../data/webqsp_data/RL_mask_even_' + percentage + '/RL_test_TR_sub_webqsp.question'
    fwTestQ = open(path, 'w', encoding="UTF-8")
    with open("../../data/webqsp_data/WebQSP_ANNOTATIONS_subtrain.json", 'r', encoding="UTF-8") as load_f:
        dict_list = json.load(load_f)
        dict_dict = dict()
        random.seed(SEED)
        random.shuffle(dict_list)
        for item in dict_list:
            key = list(dict(item).keys())[0]
            value = list(dict(item).values())[0]
            orig_response, question = "", ""
            try:
                orig_response = value['action_sequence_list']
                question = value['question']
            except SyntaxError:
                pass
            if len(orig_response) > 0 and len(question) >0:
                dict_dict[key] = value
    dict_train_list, dict_test_list = {}, {}
    train_size = int(len(dict_list))
    # train_size = int(len(dict_list) * 0.95)
    temp_count = 0
    for key,value in dict_dict.items():
        if temp_count < train_size:
            dict_train_list[key] = value
            temp_count+=1
        elif train_size <= temp_count:
            dict_test_list[key] = value
            temp_count+=1
    fwTrainQ.writelines(json.dumps(dict_train_list, indent=1, ensure_ascii=False))
    fwTrainQ.close()
    fwTestQ.writelines(json.dumps(dict_test_list, indent=1, ensure_ascii=False))
    fwTestQ.close()
    print ("Getting RL_TR processDataset is done!")

# Run getTrainingDatasetForPytorch() to get evenly-distributed training and test processDataset for seq2seq model training.
# Run getTrainingDatasetForRl() to get evenly-distributed training and test processDataset for REINFORCE-seq2seq model training.
# Vocabulary and FINAL_test files are same as the share.question and FINAL-related files used in mask processDataset.
if __name__ == "__main__":
    # percentage represents how much samples (0.2% ~ 1.2%) are drawn from the whole training dataset.
    percentage = '100.0%'
    # size = 1479
    size = 64
    getTrainingDatasetForPytorch(percentage)
    # getTrainingDatasetForRl(percentage)
    getTrainingDatasetForRlWithTrueReward(percentage, size)




