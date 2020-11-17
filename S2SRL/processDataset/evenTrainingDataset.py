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
import os
#Python codes to read the binary files.
import numpy as np
import random
OLD_SEED = 1988
SEED = 2019
LINE_SIZE = 100000

CATEGORY_SIZE = 2800
COMP_SIZE = 932
COMP_APPRO_SIZE = 2023
COMP_COUNT_SIZE = 554
COMP_COUNT_APPRO_SIZE = 1536
QUANTATIVE_SIZE = 1880
COUNT_SIZE = 733
BOOL_SIZE = 427
from random import shuffle

special_counting_characters = {'-','|','&'}
special_characters = {'(',')','-','|','&'}

# Get the training data for NSM to training by REINFORCE with True Reward.
def getTrainingDatasetForNSM(percentage, withint):
    # Create target directory & all intermediate directories if don't exists
    dirName = '../../data/auto_QA_data/nsm_mask_even_' + percentage
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    if withint:
        path1 = dirName + '/RL_train_TR_INT.question'
        path2 = dirName + '/RL_test_TR_INT.question'
        path3 = '../../data/auto_QA_data/CSQA_DENOTATIONS_full_INT.json'
        path4 = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full_INT.json'
        path5 = '../../data/auto_QA_data/CSQA_ANNOTATIONS_test_INT.json'
    else:
        path1 = dirName + '/RL_train_TR.question'
        path2 = dirName + '/RL_test_TR.question'
        path3 = '../../data/auto_QA_data/CSQA_DENOTATIONS_full.json'
        path4 = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full.json'
        path5 = '../../data/auto_QA_data/CSQA_ANNOTATIONS_test.json'

    dict_list = {}
    with open(path4, 'r', encoding="UTF-8") as load_f_annotations, open(path3, 'r', encoding="UTF-8") as load_f_denotations, open(path5, 'r', encoding="UTF-8") as test_load_f:
        # Load annotations.
        anotation_load_dict = json.load(load_f_annotations)
        list_of_anotation_load_dict = list(anotation_load_dict.items())
        random.seed(SEED)
        random.shuffle(list_of_anotation_load_dict)
        anotation_load_dict = dict(list_of_anotation_load_dict)

        # Load denotations.
        denotation_load_dict = json.load(load_f_denotations)
        fwtrain = open(path1, 'w', encoding="UTF-8")
        fwtest = open(path2, 'w', encoding="UTF-8")

        # Load test data for boolean category.
        test_load_dict = json.load(test_load_f)

        count_dict = {'simple_': 0, 'logical_': 0, 'quantative_': 0, 'count_': 0, 'bool_': 0, 'comp_': 0, 'compcount_': 0, 'compcountappro_': 0, 'compappro_': 0}
        count = 0
        for key, value in anotation_load_dict.items():
            try:
                actions = eval(str(value['mask_action_sequence_list']))
            except SyntaxError:
                pass
            if len(actions) > 0:
                if 'simple_' in key and count_dict['simple_'] < CATEGORY_SIZE:
                    count_dict['simple_'] = count_dict['simple_'] + 1
                elif 'logical_' in key and count_dict['logical_'] < CATEGORY_SIZE:
                    count_dict['logical_'] = count_dict['logical_'] + 1
                elif 'quantative_' in key and count_dict['quantative_'] < CATEGORY_SIZE:
                    count_dict['quantative_'] = count_dict['quantative_'] + 1
                elif 'count_' in key and 'compcount_' not in key and count_dict['count_'] < CATEGORY_SIZE:
                    count_dict['count_'] = count_dict['count_'] + 1
                elif 'bool_' in key and count_dict['bool_'] < CATEGORY_SIZE:
                    count_dict['bool_'] = count_dict['bool_'] + 1
                elif 'boolean_' in key and count_dict['bool_'] < CATEGORY_SIZE:
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
                action_string = ''
                action = actions[0]
                for temp_dict in action:
                    for temp_key, temp_value in temp_dict.items():
                        action_string += temp_key + ' ( '
                        for token in temp_value:
                            token = str(token)
                            if '-' in token:
                                token = '- ' + token.replace('-', '')
                            action_string += token + ' '
                        action_string += ') '
                question_token = str(value['question'])
                temp_flag = False
                for key, value in denotation_load_dict.items():
                    orig_response, question = "", ""
                    try:
                        orig_response = value['orig_response']
                        question = value['question']
                    except SyntaxError:
                        pass
                    if len(orig_response) > 0 and len(question) > 0 and question == question_token:
                        value['pseudo_gold_program'] = action_string
                        if 'response_bools' not in value:
                            value['response_bools'] = []
                        dict_list[key] = value
                        count += 1
                        temp_flag = True
                        break
                if not temp_flag:
                    for key, value in test_load_dict.items():
                        orig_response, question = "", ""
                        try:
                            orig_response = value['orig_response']
                            question = value['question']
                        except SyntaxError:
                            pass
                        if len(orig_response) > 0 and len(question) > 0 and question == question_token:
                            value['pseudo_gold_program'] = action_string
                            if 'response_bools' not in value:
                                value['response_bools'] = []

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
                                    if str(relation_value) != '':
                                        question_string += str(relation_value) + ' '
                            question_string += '</R> <T> '
                            types = value['type_mask']
                            if len(types) > 0:
                                for type_key, type_value in types.items():
                                    if str(type_value) != '':
                                        question_string += str(type_value) + ' '
                            question_string += '</T> '

                            if withint and 'int_mask' in value:
                                question_string += '<I> '
                                types = value['int_mask']
                                if len(types) > 0:
                                    for type_key, type_value in types.items():
                                        if str(type_value) != '':
                                            question_string += str(type_value) + ' '
                                question_string += '</I> '

                            question_token = str(value['question']).lower().replace('?', '')
                            question_token = question_token.replace(',', ' ')
                            question_token = question_token.replace(':', ' ')
                            question_token = question_token.replace('(', ' ')
                            question_token = question_token.replace(')', ' ')
                            question_token = question_token.replace('"', ' ')
                            question_token = question_token.strip()
                            question_string += question_token
                            question_string = question_string.strip()
                            value['input'] = question_string

                            dict_list[key] = value
                            count += 1
                            break

            if count % 100 == 0:
                print(count)

    dict_train_list, dict_test_list = {}, {}
    train_size = int(len(dict_list))
    # train_size = int(len(dict_list) * 0.95)
    temp_count = 0
    for key, value in dict_list.items():
        if temp_count < train_size:
            dict_train_list[key] = value
            temp_count += 1
        elif train_size <= temp_count:
            dict_test_list[key] = value
            temp_count += 1
    fwtrain.writelines(json.dumps(dict_train_list, indent=1, ensure_ascii=False))
    fwtrain.close()
    fwtest.writelines(json.dumps(dict_test_list, indent=1, ensure_ascii=False))
    fwtest.close()
    print("Getting NSM_RL_TR dataset is done!")

# Get the training processDataset and test processDataset for seq2seq (one question to one action ).
def getTrainingDatasetForPytorch(percentage, withint):
    # Create target directory & all intermediate directories if don't exists
    dirName = '../../data/auto_QA_data/mask_even_' + percentage
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    if withint:
        path1 = dirName + '/PT_train_INT.question'
        path2 = dirName + '/PT_train_INT.action'
        path3 = dirName + '/PT_test_INT.question'
        path4 = dirName + '/PT_test_INT.action'
        path5 = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full_INT.json'
    else:
        path1 = dirName + '/PT_train.question'
        path2 = dirName + '/PT_train.action'
        path3 = dirName + '/PT_test.question'
        path4 = dirName + '/PT_test.action'
        path5 = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full.json'

    fwTrainQ = open(path1, 'w', encoding="UTF-8")
    fwTrainA = open(path2, 'w', encoding="UTF-8")
    fwTestQ = open(path3, 'w', encoding="UTF-8")
    fwTestA = open(path4, 'w', encoding="UTF-8")
    with open(path5, 'r', encoding="UTF-8") as load_f:
        train_action_string_list, test_action_string_list, train_question_string_list, test_question_string_list = list(), list(), list(), list()
        dict_list = list()
        load_dict = json.load(load_f)
        list_of_load_dict = list(load_dict.items())
        random.seed(SEED)
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
                elif 'quantative_' in key and count_dict['quantative_'] < CATEGORY_SIZE:
                    count_dict['quantative_'] = count_dict['quantative_'] + 1
                elif 'count_' in key and 'compcount_' not in key and count_dict['count_'] < CATEGORY_SIZE:
                    count_dict['count_'] = count_dict['count_'] + 1
                elif 'bool_' in key and count_dict['bool_'] < CATEGORY_SIZE:
                    count_dict['bool_'] = count_dict['bool_'] + 1
                elif 'boolean_' in key and count_dict['bool_'] < CATEGORY_SIZE:
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
                action_string = ''
                action = actions[0]
                for temp_dict in action:
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
                        if str(type_value) != '':
                            question_string += str(type_value) + ' '
                question_string += '</T> '
                if withint and 'int_mask' in value:
                    question_string += '<I> '
                    types = value['int_mask']
                    if len(types) > 0:
                        for type_key, type_value in types.items():
                            if str(type_value) != '':
                                question_string += str(type_value) + ' '
                    question_string += '</I> '

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
    for i, item in enumerate(dict_list):
        if i < train_size:
            train_action_string_list.append(item.get('a'))
            train_question_string_list.append(item.get('q'))
        elif train_size <= i:
            test_action_string_list.append(item.get('a'))
            test_question_string_list.append(item.get('q'))
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
def getTrainingDatasetForRl(percentage, withint):
    # Create target directory & all intermediate directories if don't exists
    dirName = '../../data/auto_QA_data/mask_even_' + percentage
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

    if withint:
        path1 = dirName + '/RL_train_INT.question'
        path2 = dirName + '/RL_train_INT.action'
        path3 = dirName + '/RL_test_INT.question'
        path4 = dirName + '/RL_test_INT.action'
        path5 = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full_INT.json'
    else:
        path1 = dirName + '/RL_train.question'
        path2 = dirName + '/RL_train.action'
        path3 = dirName + '/RL_test.question'
        path4 = dirName + '/RL_test.action'
        path5 = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full.json'

    fwTrainQ = open(path1, 'w', encoding="UTF-8")
    fwTrainA = open(path2, 'w', encoding="UTF-8")
    fwTestQ = open(path3, 'w', encoding="UTF-8")
    fwTestA = open(path4, 'w', encoding="UTF-8")
    # fwNoaction = open('../../data/auto_QA_data/mask_even/no_action_question.txt', 'w', encoding="UTF-8")
    no_action_question_list = list()
    questionSet = set()
    actionSet = set()
    with open(path5, 'r', encoding="UTF-8") as load_f:
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
                elif 'quantative_' in key and count_dict['quantative_'] < CATEGORY_SIZE:
                    count_dict['quantative_'] = count_dict['quantative_'] + 1
                elif 'count_' in key and 'compcount_' not in key and count_dict['count_'] < CATEGORY_SIZE:
                    count_dict['count_'] = count_dict['count_'] + 1
                elif 'bool_' in key and count_dict['bool_'] < CATEGORY_SIZE:
                    count_dict['bool_'] = count_dict['bool_'] + 1
                elif 'boolean_' in key and count_dict['bool_'] < CATEGORY_SIZE:
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
                if withint and 'int_mask' in value:
                    question_string += '<I> '
                    types = value['int_mask']
                    if len(types) > 0:
                        for type_key, type_value in types.items():
                            if str(type_value) != '':
                                question_string += str(type_value) + ' '
                    question_string += '</I> '
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
    print("Getting RL processDataset is done!")

# Get the training processDataset and test processDataset for REINFORCE with True Reward (one question with one answer).
def getTrainingDatasetForRlWithTrueReward(percentage, SIZE, withint):
    # Create target directory & all intermediate directories if don't exists
    dirName = '../../data/auto_QA_data/mask_even_' + percentage
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    if withint:
        path1 = dirName + '/RL_train_TR_new_INT.question'
        path2 = dirName + '/RL_test_TR_new_INT.question'
        path3 = '../../data/auto_QA_data/CSQA_DENOTATIONS_full_944K_INT.json'
    else:
        path1 = dirName + '/RL_train_TR_new_2k.question'
        path2 = dirName + '/RL_test_TR_new_2k.question'
        path3 = '../../data/auto_QA_data/CSQA_DENOTATIONS_full_944K.json'

    fwTrainQ = open(path1, 'w', encoding="UTF-8")
    fwTestQ = open(path2, 'w', encoding="UTF-8")
    with open(path3, 'r', encoding="UTF-8") as load_f:
        dict_list = {}
        load_dict = json.load(load_f)
        list_of_load_dict = list(load_dict.items())
        random.seed(SEED+8)
        random.shuffle(list_of_load_dict)
        load_dict = dict(list_of_load_dict)
        count_dict = {'simple_': 0, 'logical_': 0, 'quantative_': 0, 'count_': 0, 'bool_': 0, 'comp_': 0, 'compcount_': 0}
        for key, value in load_dict.items():
            orig_response, question = "", ""
            try:
                orig_response = value['orig_response']
                question = value['question']
            except SyntaxError:
                pass
            if len(orig_response) > 0 and len(question) >0:
                if 'Simple Question (Direct)_' in key and count_dict['simple_'] < SIZE:
                    count_dict['simple_'] = count_dict['simple_'] + 1
                elif 'Logical Reasoning (All)_' in key and count_dict['logical_'] < SIZE:
                    count_dict['logical_'] = count_dict['logical_'] + 1
                elif 'Quantitative Reasoning (All)_' in key and count_dict['quantative_'] < SIZE:
                    count_dict['quantative_'] = count_dict['quantative_'] + 1
                elif 'Quantitative Reasoning (Count) (All)_' in key and count_dict['count_'] < SIZE:
                    count_dict['count_'] = count_dict['count_'] + 1
                elif 'Verification (Boolean) (All)_' in key and count_dict['bool_'] < SIZE:
                    count_dict['bool_'] = count_dict['bool_'] + 1
                elif 'Comparative Reasoning (All)_' in key and count_dict['comp_'] < SIZE:
                    count_dict['comp_'] = count_dict['comp_'] + 1
                elif 'Comparative Reasoning (Count) (All)_' in key and count_dict['compcount_'] < SIZE:
                    count_dict['compcount_'] = count_dict['compcount_'] + 1
                else:
                    continue
                dict_list[key] = value
    dict_train_list, dict_test_list = {}, {}
    train_size = int(len(dict_list))
    # train_size = int(len(dict_list) * 0.95)
    temp_count = 0
    for key,value in dict_list.items():
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
    print("Getting RL_TR processDataset is done!")

# Run getTrainingDatasetForPytorch() to get evenly-distributed training and test processDataset for seq2seq model training.
# Run getTrainingDatasetForRl() to get evenly-distributed training and test processDataset for REINFORCE-seq2seq model training.
# Vocabulary and FINAL_test files are same as the share.question and FINAL-related files used in mask processDataset.
if __name__ == "__main__":
    # percentage represents how much samples (0.2% ~ 1.2%) are drawn from the whole training dataset.
    percentage = '1.0%'
    size = 1479

    # getTrainingDatasetForPytorch(percentage, withint=True)
    # getTrainingDatasetForRl(percentage, withint=True)
    # getTrainingDatasetForRlWithTrueReward(percentage, size, withint=True)
    getTrainingDatasetForNSM(percentage, withint=False)
