# coding:utf-8
'''Get all questions, annotated actions, entities, relations, types together in JSON format.
'''
from itertools import islice
from Preprocess.load_qadata import load_qadata, getQA_by_state_py3
import sys
import json
#Python codes to read the binary files.
import numpy as np
import random
SEED = 1987
LINE_SIZE = 100000
from random import shuffle

special_counting_characters = {'-','|','&'}
special_characters = {'(',')','-','|','&'}

# Get items_wikidata_n.json.
# This json contains a dict with keys as the ids of wikidata entities and values as their string labels.
def getEntitiyID2Label():
    with open("../../data/official_downloaded_data/items_wikidata_n.json", 'r', encoding="UTF-8") as load_f:
        load_dict = json.load(load_f)
    print ('Reading items_wikidata_n.json is done!')
    return load_dict

# To get the pure predicted actions in the file sample_final_maml_predict.actions.
def getPurePredictedActions(r_path, w_path):
    lines, processed_lines = list(), list()
    with open(r_path, 'r', encoding="UTF-8") as infile:
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                lines.append(line.strip())
    fw = open(w_path, 'w', encoding="UTF-8")
    for count, line in enumerate(lines):
        processed_lines.append(str(count+1) + ':' + str(line).split(':')[1] + '\n')
    fw.writelines(processed_lines)
    fw.close()
    print('Processing is done!')

def getAllQustionsAndAnswers(json_path, training_folder_path, dict_path, withint):
    fw = open(json_path, 'w', encoding="UTF-8")
    qa_set = load_qadata(training_folder_path)
    qa_map = getQA_by_state_py3(qa_set)
    # load_dict: a dict with keys as the ids of wikidata entities and values as their string labels.
    load_dict = getEntitiyID2Label()
    question_dict = {}
    totality = 0
    # The tokens appear in the training set.
    fwQuestionDic = open(dict_path, 'w', encoding="UTF-8")
    questionSet = set()
    for qafile_key, qafile_value in qa_map.items():
        id_prefix = qafile_key
        count = 0
        for qa in qafile_value:
            # context = qa['context'].replace("\n", "").strip()
            context_utterance = qa['context_utterance'].replace("\n", "").strip()
            if len(context_utterance) == 0:
                continue
            count += 1
            id = id_prefix.replace("\n", "").strip() + '_' + str(count)
            context_entities = [] if(len(qa['context_entities'].replace("\n", "").strip())==0) else qa['context_entities'].replace("\n", "").strip().split("|")
            context_relations = [] if(len(qa['context_relations'].replace("\n", "").strip())==0) else qa['context_relations'].replace("\n", "").strip().split("|")
            context_types = [] if (len(qa['context_types'].replace("\n", "").strip())==0) else qa['context_types'].replace("\n", "").strip().split("|")
            context_ints = qa['context_ints'].replace("\n", "").strip()
            if context_ints.isdigit():
                context_ints = str(int(context_ints))
            else:
                context_ints = ''
            # Get reverse relation: has_child and -has_child.
            # context_relations.extend(['-' + r for r in context_relations])
            response_entities = [] if (len(qa['response_entities'].replace("\n", "").strip())==0) else qa['response_entities'].replace("\n", "").strip().split("|")
            orig_response = qa['orig_response'].replace("\n", "").strip()
            response_bools = [] if (len(qa['response_bools'].replace("\n", "").strip())==0) else qa['response_bools'].replace("\n", "").strip().split("|")
            # response_ints = qa['response_ints'].replace("\n", "").split("|")
            question_info = {}
            question_info.update({'question': context_utterance,
                                  'entity': context_entities, 'relation': context_relations,
                                  'type': context_types, 'context_ints': context_ints,
                                  'response_entities': response_entities, 'orig_response': orig_response,
                                  'response_bools': response_bools})
            # Get masked entity.
            entity_maskID = {}
            if len(context_entities) != 0:
                context_utterance_low = context_utterance.lower()
                entity_index_dict = {}
                for entity in context_entities:
                    if entity in load_dict:
                        entity_name = load_dict.get(entity).lower()
                        entity_index_dict[entity] = entity_name
                    else:
                        entity_index_dict[entity] = entity
                for key, value in entity_index_dict.items():
                    if value in context_utterance_low:
                        entity_index_dict[key] = context_utterance_low.index(value)
                    else:
                        entity_index_dict[key] = LINE_SIZE
                temp_count = 0
                # To sort a dict by value.
                for key, value in sorted(entity_index_dict.items(), key=lambda item: item[1]):
                    entity_maskID[key] = 'ENTITY' + str(temp_count + 1)
                    temp_count += 1
            question_info['entity_mask'] = entity_maskID
            # question_info.update({'entity_mask': entity_maskID})

            # Get masked relation.
            relation_maskID = {}
            if len(context_relations) != 0:
                relation_index = 0
                for relation in context_relations:
                    relation = relation.replace('-', '')
                    if relation not in relation_maskID:
                        relation_index += 1
                        relation_maskID[relation] = 'RELATION' + str(relation_index)
            question_info['relation_mask'] = relation_maskID
            # question_info.update({'relation_mask': relation_maskID})

            # Get masked type.
            type_maskID = {}
            if len(context_types) != 0:
                for i, context_type in enumerate(context_types):
                    type_maskID[context_type] = 'TYPE' + str(i + 1)
            question_info['type_mask'] = type_maskID
            # question_info.update({'type_mask': type_maskID})

            if withint:
                int_maskID = {}
                if context_ints != '':
                    int_maskID[int(context_ints)] = 'INT'
                question_info['int_mask'] = int_maskID

            question_string = '<E> '
            if len(entity_maskID) > 0:
                for entity_key, entity_value in entity_maskID.items():
                    if str(entity_value) != '':
                        question_string += str(entity_value) + ' '
            question_string += '</E> <R> '
            if len(relation_maskID) > 0:
                for relation_key, relation_value in relation_maskID.items():
                    if str(relation_value) != '':
                        question_string += str(relation_value) + ' '
            question_string += '</R> <T> '
            if len(type_maskID) > 0:
                for type_key, type_value in type_maskID.items():
                    if str(type_value) != '':
                        question_string += str(type_value) + ' '
            question_string += '</T> '
            if withint:
                question_string += '<I> '
                if len(int_maskID) > 0:
                    for type_key, type_value in int_maskID.items():
                        if str(type_value) != '':
                            question_string += str(type_value) + ' '
                question_string += '</I> '

            question_token = str(context_utterance).lower().replace('?', '')
            question_token = question_token.replace(',', ' ')
            question_token = question_token.replace(':', ' ')
            question_token = question_token.replace('(', ' ')
            question_token = question_token.replace(')', ' ')
            question_token = question_token.replace('"', ' ')
            question_token = question_token.strip()
            question_string += question_token
            question_string = question_string.strip()
            question_tokens = question_string.strip().split(' ')
            question_tokens_set = set(question_tokens)
            questionSet = questionSet.union(question_tokens_set)
            question_info['input'] = question_string
            question_dict[id] = question_info
            totality += 1
            if totality % 10000 == 0:
                print(totality)
    fw.writelines(json.dumps(question_dict, indent=1, ensure_ascii=False))
    fw.close()
    questionList = list()
    for item in questionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            questionList.append(temp)
    fwQuestionDic.writelines(questionList)
    fwQuestionDic.close()
    print("Writing %s is done!" % json_path)

# Get all questions and answers in CQA-full data-set for REINFORCEMENT learning by using denotations.
# Also treat the JSON as the file of training set used for REINFORCEMENT learning with denotations.
def getAllQustionsAndAnswersForWebqsp():
    fw = open('../../data/webqsp_data/CSQA_DENOTATIONS_full_944K.json', 'w', encoding="UTF-8")
    qa_set = load_qadata("../../data/official_downloaded_data/944k/train")
    qa_map = getQA_by_state_py3(qa_set)
    load_dict = getEntitiyID2Label()
    question_dict = {}
    totality = 0
    fwQuestionDic = open('../../data/webqsp_data/mask/dic_rl_tr.question', 'w', encoding="UTF-8")
    questionSet = set()
    for qafile_key, qafile_value in qa_map.items():
        id_prefix = qafile_key
        count = 0
        for qa in qafile_value:
            # context = qa['context'].replace("\n", "").strip()
            context_utterance = qa['context_utterance'].replace("\n", "").strip()
            if(len(context_utterance)==0):
                continue
            count += 1
            id = id_prefix.replace("\n", "").strip() + '_' + str(count)
            context_entities = [] if(len(qa['context_entities'].replace("\n", "").strip())==0) else qa['context_entities'].replace("\n", "").strip().split("|")
            context_relations = [] if(len(qa['context_relations'].replace("\n", "").strip())==0) else qa['context_relations'].replace("\n", "").strip().split("|")
            context_types = [] if (len(qa['context_types'].replace("\n", "").strip())==0) else qa['context_types'].replace("\n", "").strip().split("|")
            context_ints = qa['context_ints'].replace("\n", "").strip()
            # Get reverse relation: has_child and -has_child.
            # context_relations.extend(['-' + r for r in context_relations])
            response_entities = []  if (len(qa['response_entities'].replace("\n", "").strip())==0) else qa['response_entities'].replace("\n", "").strip().split("|")
            orig_response = qa['orig_response'].replace("\n", "").strip()
            response_bools = [] if (len(qa['response_bools'].replace("\n", "").strip())==0) else qa['response_bools'].replace("\n", "").strip().split("|")
            # response_ints = qa['response_ints'].replace("\n", "").split("|")
            question_info = {}
            question_info.update({'question': context_utterance,
                                  'entity': context_entities, 'relation': context_relations,
                                  'type': context_types,'context_ints': context_ints,
                                  'response_entities': response_entities,'orig_response': orig_response,
                                  'response_bools': response_bools})
            # Get masked entity.
            entity_maskID = {}
            if len(context_entities)!=0:
                context_utterance_low = context_utterance.lower()
                entity_index_dict = {}
                for entity in context_entities:
                    if(entity in load_dict):
                        entity_name = load_dict.get(entity).lower()
                        entity_index_dict[entity] = entity_name
                    else:
                        entity_index_dict[entity] = entity
                for key, value in entity_index_dict.items():
                    if(value in context_utterance_low):
                        entity_index_dict[key] = context_utterance_low.index(value)
                    else:
                        entity_index_dict[key] = LINE_SIZE
                temp_count = 0
                # To sort a dict by value.
                for key, value in sorted(entity_index_dict.items(), key=lambda item: item[1]):
                    entity_maskID[key] = 'ENTITY' + str(temp_count + 1)
                    temp_count+=1
            question_info['entity_mask'] = entity_maskID

            # Get masked relation.
            relation_maskID = {}
            if len(context_relations) != 0:
                relation_index = 0
                for relation in context_relations:
                    relation = relation.replace('-', '')
                    if relation not in relation_maskID:
                        relation_index += 1
                        relation_maskID[relation] = 'RELATION' + str(relation_index)
            question_info['relation_mask'] = relation_maskID

            # Get masked type.
            type_maskID = {}
            if len(context_types) != 0:
                for i, type in enumerate(context_types):
                    type_maskID[type] = 'TYPE' + str(i + 1)
            question_info['type_mask'] = type_maskID

            question_string = '<E> '
            if len(entity_maskID) > 0:
                for entity_key, entity_value in entity_maskID.items():
                    if str(entity_value) != '':
                        question_string += str(entity_value) + ' '
            question_string += '</E> <R> '
            if len(relation_maskID) > 0:
                for relation_key, relation_value in relation_maskID.items():
                    if str(relation_value) != '':
                        question_string += str(relation_value) + ' '
            question_string += '</R> <T> '
            if len(type_maskID) > 0:
                for type_key, type_value in type_maskID.items():
                    if str(type_value) != '':
                        question_string += str(type_value) + ' '
            question_string += '</T> '

            question_token = str(context_utterance).lower().replace('?', '')
            question_token = question_token.replace(',', ' ')
            question_token = question_token.replace(':', ' ')
            question_token = question_token.replace('(', ' ')
            question_token = question_token.replace(')', ' ')
            question_token = question_token.replace('"', ' ')
            question_token = question_token.strip()
            question_string += question_token
            question_string = question_string.strip()
            question_tokens = question_string.strip().split(' ')
            question_tokens_set = set(question_tokens)
            questionSet = questionSet.union(question_tokens_set)
            question_info['input'] = question_string
            question_dict[id] = question_info
            totality += 1
            if(totality%1000==0):
                print(totality)
    fw.writelines(json.dumps(question_dict, indent=1, ensure_ascii=False))
    fw.close()
    questionList = list()
    for item in questionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            questionList.append(temp)
    fwQuestionDic.writelines(questionList)
    fwQuestionDic.close()
    print("Writing WEBQSP_DENOTATIONS_full.JSON is done!")

# Get annotations of logical questions from CSQA_ANNOTATIONS_full.json.
def getNewAnnotationsForLogical():
    count = 0
    lines = list()
    with open("../../data/auto_QA_data/CSQA_ANNOTATIONS_full.json", 'r', encoding="UTF-8") as load_f:
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            if 'logical_' in key:
                try:
                    question = str(value['question'])
                    lines.append(str(str(key).split('_')[1]) + ' ' + question + '\n')
                    actions = eval(str(value['action_sequence_list']))
                    for action in actions:
                        lines.append(str(action)+'\n')
                    count += 1
                except SyntaxError:
                    pass
    fw = open('../../data/annotation_logs/logical_auto.log', 'w', encoding="UTF-8")
    fw.writelines(lines)
    print("Writing to %s is done!" %('logical_auto.log'))
    fw.close()

def getAllQuestionsAndActions(withint):
    if withint:
        path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full_INT.json'
    else:
        path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full.json'
    fw = open(path, 'w', encoding="UTF-8")
    '''dictMerged2 = dict( dict1, **dict2 ) is :
        dictMerged2 = dict1.copy()
        dictMerged2.update( dict2 )'''
    question_dicts = dict(
        getQuestionsAndActions('../../data/annotation_logs/simple_auto.log', '../../data/annotation_logs/simple_orig.log', withint),
    **(getQuestionsAndActions('../../data/annotation_logs/quantitative_combine_auto.log','../../data/annotation_logs/quantative_orig.log', withint)))
    question_dicts = dict(question_dicts,
        **(getQuestionsAndActions('../../data/annotation_logs/logical_auto.log','../../data/annotation_logs/logical_orig.log', withint)))
    question_dicts = dict(question_dicts,
                          **(getQuestionsAndActions('../../data/annotation_logs/count_combine_auto.log',
                                                    '../../data/annotation_logs/count_orig.log', withint)))
    # Annotated test boolean questions.
    question_dicts = dict(question_dicts,
                          **(getQuestionsAndActions('../../data/annotation_logs/bool_auto.log',
                                                    '../../data/annotation_logs/bool_orig.log', withint)))
    # Annotated training boolean questions.
    question_dicts = dict(question_dicts,
                          **(getQuestionsAndActions('../../data/annotation_logs/boolean_auto.log',
                                                    '../../data/annotation_logs/boolean_orig.log', withint)))
    question_dicts = dict(question_dicts,
                          **(getQuestionsAndActions('../../data/annotation_logs/comp_auto.log',
                                                    '../../data/annotation_logs/comp_orig.log', withint)))
    question_dicts = dict(question_dicts,
                          **(getQuestionsAndActions('../../data/annotation_logs/compcount_auto.log',
                                                    '../../data/annotation_logs/compcount_orig.log', withint)))
    question_dicts = dict(question_dicts,
                          **(getQuestionsAndActions('../../data/annotation_logs/compappro_auto.log',
                                                    '../../data/annotation_logs/compappro_orig.log', withint)))
    question_dicts = dict(question_dicts,
                          **(getQuestionsAndActions('../../data/annotation_logs/compcountappro_auto.log',
                                                    '../../data/annotation_logs/compcountappro_orig.log', withint)))
    fw.writelines(json.dumps(question_dicts, indent=1, ensure_ascii=False))
    fw.close()
    print("Writing JSON is done!")

def getQuestionAndContextints():
    contextIntPath = '../../data/demoqa2/context_ints_10k.txt'
    lines = list()
    with open(contextIntPath, 'r', encoding="UTF-8") as infile:
        count = 0
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                lines.append(line.strip())
            count = count + 1
            print(count)
    count = 0
    q_ints = dict()
    while count < len(lines):
        line = str(lines[count])
        if line.startswith('context_utterance'):
            q_utterance = ' '.join(line.split(' ')[1:]).strip()
            q_ints[q_utterance] = ''
            count += 1
            if count < len(lines) and str(lines[count]).startswith('context_ints'):
                int_tokens = ' '.join(str(lines[count]).split(' ')[1:]).strip()
                if int_tokens != '' and int_tokens.strip().isdigit():
                    q_ints[q_utterance] = str(int(int_tokens))
            count += 1
    return q_ints

def getQuestionsAndActions(annotationPath, origPath, withint):
    if withint:
        q_ints = getQuestionAndContextints()
    annotation_list = list()
    orig_list = list()
    question_dict = {}
    with open(annotationPath, 'r', encoding="UTF-8") as infile:
        count = 0
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                annotation_list.append(line.strip())
            count = count + 1
            print(count)
    with open(origPath, 'r', encoding="UTF-8") as infile:
        count = 0
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                orig_list.append(line.strip())
            count = count + 1
            print(count)
    count = 0
    # Each question is corresponding to one action sequence.
    while count < len(annotation_list):
        line_string = str(annotation_list[count]).strip()
        if str(line_string).startswith('['):
            count += 1
        else:
            string_list = str(line_string).split(' ')
            ID_string = ''
            if 'simple_' in str(annotationPath):
                ID_string = 'simple_' + string_list[0]
            elif 'logical_' in str(annotationPath):
                ID_string = 'logical_' + string_list[0]
            elif 'quantative_' in str(annotationPath) or 'quantitative_' in str(annotationPath):
                ID_string = 'quantative_' + string_list[0]
            elif 'count_' in str(annotationPath) and 'compcount_' not in str(annotationPath):
                ID_string = 'count_' + string_list[0]
            elif 'bool_' in str(annotationPath):
                ID_string = 'bool_' + string_list[0]
            elif 'boolean_' in str(annotationPath):
                ID_string = 'boolean_' + string_list[0]
            elif 'comp_' in str(annotationPath):
                ID_string = 'comp_' + string_list[0]
            elif 'compcount_' in str(annotationPath):
                ID_string = 'compcount_' + string_list[0]
            elif 'compcountappro_' in str(annotationPath):
                ID_string = 'compcountappro_' + string_list[0]
            elif 'compappro_' in str(annotationPath):
                ID_string = 'compappro_' + string_list[0]
            question_string = ' '.join(string_list[1:])
            actionSequenceList = list()
            count += 1
            while count < len(annotation_list) and str(annotation_list[count]).strip().startswith('['):
                actionSequence = eval(str(annotation_list[count]).strip())
                actionSequenceList.append(actionSequence)
                count += 1
            question_info = {}
            # Original annotation information.
            # question_info.setdefault('question', question_string.strip())
            # question_info.setdefault('action_sequence_list', actionSequenceList)
            # question_dict.setdefault(ID_string, question_info)
            question_info['question'] = question_string.strip()
            question_info['action_sequence_list'] = actionSequenceList
            question_dict[ID_string] = question_info

    ID_string = ''
    count = 0
    while count < len(orig_list):
        if 'simple_' in str(origPath):
            ID_string = 'simple_' + str(orig_list[count]).strip()
        elif 'logical_' in str(origPath):
            ID_string = 'logical_' + str(orig_list[count]).strip()
        elif 'quantative_' in str(origPath):
            ID_string = 'quantative_' + str(orig_list[count]).strip()
        elif 'count_' in str(origPath) and 'compcount_' not in str(origPath):
            ID_string = 'count_' + str(orig_list[count]).strip()
        elif 'bool_' in str(origPath):
            ID_string = 'bool_' + str(orig_list[count]).strip()
        elif 'boolean_' in str(origPath):
            ID_string = 'boolean_' + str(orig_list[count]).strip()
        elif 'comp_' in str(origPath):
            ID_string = 'comp_' + str(orig_list[count]).strip()
        elif 'compcount_' in str(origPath):
            ID_string = 'compcount_' + str(orig_list[count]).strip()
        elif 'compcountappro_' in str(origPath):
            ID_string = 'compcountappro_' + str(orig_list[count]).strip()
        elif 'compappro_' in str(annotationPath):
            ID_string = 'compappro_' + str(orig_list[count]).strip()
        if ID_string in question_dict:
            question_info_new = question_dict.get(ID_string)
            entity_list = list()
            relation_list = list()
            type_list = list()
            if count+2 < len(orig_list) and str(orig_list[count+2]).startswith('context_entities'):
                entity_string = str(orig_list[count+2]).strip()
                entity_string = entity_string.replace('context_entities:','').strip()
                if len(entity_string) > 0:
                    entity_list = entity_string.split(',')
            if count+3 < len(orig_list) and str(orig_list[count+3]).startswith('context_relations'):
                relation_string = str(orig_list[count+3]).strip()
                relation_string = relation_string.replace('context_relations:','').strip()
                if len(relation_string) > 0:
                    relation_list = relation_string.split(',')
            if count+4 < len(orig_list) and str(orig_list[count+4]).startswith('context_types'):
                type_string = str(orig_list[count+4]).strip()
                type_string = type_string.replace('context_types:','').strip()
                if len(type_string) > 0:
                    type_list = type_string.split(',')
            # question_info_new.setdefault('entity', entity_list)
            # question_info_new.setdefault('relation', relation_list)
            # question_info_new.setdefault('type', type_list)
            question_info_new['entity'] = entity_list
            question_info_new['relation'] = relation_list
            question_info_new['type'] = type_list

            entity_maskID = {}
            if len(entity_list) != 0:
                for i, entity in enumerate(entity_list):
                    # entity_maskID.setdefault(entity, 'ENTITY'+str(i+1))
                    entity_maskID[entity] = 'ENTITY'+str(i+1)
            # question_info_new.setdefault('entity_mask', entity_maskID)
            question_info_new['entity_mask'] = entity_maskID

            relation_maskID = {}
            if len(relation_list) != 0:
                relation_index = 0
                for relation in relation_list:
                    relation = relation.replace('-','')
                    if relation not in relation_maskID:
                        relation_index += 1
                        relation_maskID[relation] = 'RELATION' + str(relation_index)
                        # relation_maskID.setdefault(relation, 'RELATION' + str(relation_index))
            question_info_new['relation_mask'] = relation_maskID
            # question_info_new.setdefault('relation_mask', relation_maskID)
            type_maskID = {}
            if len(type_list) != 0:
                for i, type in enumerate(type_list):
                    type_maskID[type] = 'TYPE' + str(i + 1)
                    # type_maskID.setdefault(type, 'TYPE' + str(i + 1))
            question_info_new['type_mask'] = type_maskID
            # question_info_new.setdefault('type_mask', type_maskID)

            if withint:
                q_string = question_info_new['question']
                if q_string in q_ints:
                    question_info_new['context_ints'] = q_ints[q_string]
                else:
                    question_info_new['context_ints'] = ''
                    print('Question not in q_ints is: %s' %(ID_string + ': ' + q_string))
                int_maskID = {}
                if question_info_new['context_ints'] != '':
                    int_maskID[question_info_new['context_ints']] = 'INT'
                question_info_new['int_mask'] = int_maskID

            actions = question_info_new.get('action_sequence_list')
            MASK_actions = list()
            if len(actions) > 0:
                for action in actions:
                    MASK_action = list()
                    for dict in action:
                        MASK_dict = {}
                        for temp_key, temp_value in dict.items():
                            MASK_key = temp_key
                            MASK_value = list()
                            for token in temp_value:
                                if isinstance(token, int):
                                    if withint:
                                        if str(token) in int_maskID:
                                            MASK_value.append(int_maskID[str(token)])
                                    else:
                                        MASK_value.append(token)
                                elif '-' in token and token != '-':
                                    token_new = token.replace('-','')
                                    if token_new in relation_maskID:
                                        MASK_value.append('-' + str(relation_maskID.get(token_new)))
                                else:
                                    if token in entity_maskID:
                                        MASK_value.append(entity_maskID.get(token))
                                    elif token in relation_maskID:
                                        MASK_value.append(relation_maskID.get(token))
                                    elif token in type_maskID:
                                        MASK_value.append(type_maskID.get(token))
                                    elif token in special_counting_characters:
                                        MASK_value.append(token)
                            MASK_dict[MASK_key] = MASK_value
                            # MASK_dict.setdefault(MASK_key, MASK_value)
                        MASK_action.append(MASK_dict)
                    MASK_actions.append(MASK_action)
            question_info_new['mask_action_sequence_list'] = MASK_actions
            # question_info_new.setdefault('mask_action_sequence_list', MASK_actions)
            question_dict[ID_string] = question_info_new
            # question_dict.setdefault(ID_string, question_info_new)
        count += 5
    return question_dict

# Get training data for sequence2sequence.
def getTrainingDatasetForPytorch(withint):
    if withint:
        fwTrainQ = open('../../data/auto_QA_data/mask/PT_train_INT.question', 'w', encoding="UTF-8")
        fwTrainA = open('../../data/auto_QA_data/mask/PT_train_INT.action', 'w', encoding="UTF-8")
        fwTestQ = open('../../data/auto_QA_data/mask/PT_test_INT.question', 'w', encoding="UTF-8")
        fwTestA = open('../../data/auto_QA_data/mask/PT_test_INT.action', 'w', encoding="UTF-8")
        fwQuestionDic = open('../../data/auto_QA_data/mask/dic_py_INT.question', 'w', encoding="UTF-8")
        fwActionDic = open('../../data/auto_QA_data/mask/dic_py_INT.action', 'w', encoding="UTF-8")
        path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full_INT.json'
    else:
        fwTrainQ = open('../../data/auto_QA_data/mask/PT_train.question', 'w', encoding="UTF-8")
        fwTrainA = open('../../data/auto_QA_data/mask/PT_train.action', 'w', encoding="UTF-8")
        fwTestQ = open('../../data/auto_QA_data/mask/PT_test.question', 'w', encoding="UTF-8")
        fwTestA = open('../../data/auto_QA_data/mask/PT_test.action', 'w', encoding="UTF-8")
        fwQuestionDic = open('../../data/auto_QA_data/mask/dic_py.question', 'w', encoding="UTF-8")
        fwActionDic = open('../../data/auto_QA_data/mask/dic_py.action', 'w', encoding="UTF-8")
        path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full.json'
    questionSet = set()
    actionSet = set()
    with open(path, 'r', encoding="UTF-8") as load_f:
        count = 1
        train_action_string_list, test_action_string_list, train_question_string_list, test_question_string_list = list(), list(), list(), list()
        dict_list = list()
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            try:
                actions = eval(str(value['mask_action_sequence_list']))
            except SyntaxError:
                pass
            if len(actions) > 0:
                count += 1
                action_string = ''
                action = actions[0]
                for dict in action:
                    for temp_key, temp_value in dict.items():
                        action_string += temp_key + ' ( '
                        for token in temp_value:
                            token = str(token)
                            if '-' in token:
                                token = '- ' + token.replace('-','')
                            action_string += token + ' '
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
                action_tokens = action_string.strip().split(' ')
                action_tokens_set = set(action_tokens)
                actionSet = actionSet.union(action_tokens_set)

                question_tokens = question_string.strip().split(' ')
                question_tokens_set = set(question_tokens)
                questionSet = questionSet.union(question_tokens_set)

                dict_temp = {}
                dict_temp['q'] = str(key) + ' ' + question_string
                dict_temp['a'] = str(key) + ' ' + action_string
                # dict_temp.setdefault('q', str(key) + ' ' + question_string)
                # dict_temp.setdefault('a', str(key) + ' ' + action_string)
                dict_list.append(dict_temp)

    random.seed(SEED)
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
    fwQuestionDic.writelines(questionList)
    fwActionDic.writelines(actionList)
    print ("Getting SEQUENCE2SEQUENCE processDataset is done!")

# Get training data for REINFORCE (using annotation instead of denotation as reward).
def getTrainingDatasetForRl(withint):
    if withint:
        fwTrainQ = open('../../data/auto_QA_data/mask/RL_train_INT.question', 'w', encoding="UTF-8")
        fwTrainA = open('../../data/auto_QA_data/mask/RL_train_INT.action', 'w', encoding="UTF-8")
        fwTestQ = open('../../data/auto_QA_data/mask/RL_test_INT.question', 'w', encoding="UTF-8")
        fwTestA = open('../../data/auto_QA_data/mask/RL_test_INT.action', 'w', encoding="UTF-8")
        fwQuestionDic = open('../../data/auto_QA_data/mask/dic_rl_INT.question', 'w', encoding="UTF-8")
        fwActionDic = open('../../data/auto_QA_data/mask/dic_rl_INT.action', 'w', encoding="UTF-8")
        fwNoaction = open('../../data/auto_QA_data/mask/no_action_question_INT.txt', 'w', encoding="UTF-8")
        path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full_INT.json'
    else:
        fwTrainQ = open('../../data/auto_QA_data/mask/RL_train.question', 'w', encoding="UTF-8")
        fwTrainA = open('../../data/auto_QA_data/mask/RL_train.action', 'w', encoding="UTF-8")
        fwTestQ = open('../../data/auto_QA_data/mask/RL_test.question', 'w', encoding="UTF-8")
        fwTestA = open('../../data/auto_QA_data/mask/RL_test.action', 'w', encoding="UTF-8")
        fwQuestionDic = open('../../data/auto_QA_data/mask/dic_rl.question', 'w', encoding="UTF-8")
        fwActionDic = open('../../data/auto_QA_data/mask/dic_rl.action', 'w', encoding="UTF-8")
        fwNoaction = open('../../data/auto_QA_data/mask/no_action_question.txt', 'w', encoding="UTF-8")
        path = '../../data/auto_QA_data/CSQA_ANNOTATIONS_full.json'
    no_action_question_list = list()
    questionSet = set()
    actionSet = set()
    with open(path, 'r', encoding="UTF-8") as load_f:
        count = 1
        train_action_string_list, test_action_string_list, train_question_string_list, test_question_string_list = list(), list(), list(), list()
        dict_list = list()
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            try:
                actions = eval(str(value['mask_action_sequence_list']))
            except SyntaxError:
                pass
            if len(actions) > 0:
                action_string_list = list()
                count += 1
                for action in actions:
                    action_string = ''
                    for dict in action:
                        for temp_key, temp_value in dict.items():
                            action_string += temp_key + ' ( '
                            for token in temp_value:
                                token = str(token)
                                if '-' in token:
                                    token = '- ' + token.replace('-','')
                                action_string += token + ' '
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
                dict_temp['q'] = str(key) + ' ' + question_string
                dict_temp['a'] = action_withkey_list
                # dict_temp.setdefault('q', str(key) + ' ' + question_string)
                # dict_temp.setdefault('a', action_withkey_list)
                dict_list.append(dict_temp)
            elif len(actions) == 0:
                no_action_question_list.append(str(key) + ' ' + str(value['question']).lower().replace('?', '').strip() + '\n')

    random.seed(SEED)
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
    fwQuestionDic.writelines(questionList)
    fwActionDic.writelines(actionList)
    fwNoaction.writelines(no_action_question_list)
    fwTrainQ.close()
    fwTrainA.close()
    fwTestQ.close()
    fwTestA.close()
    fwQuestionDic.close()
    fwActionDic.close()
    fwNoaction.close()
    print("Getting RL processDataset is done!")

def getShareVocabulary(infile1_path, infile2_path, infile3_path, infile4_path, infile5_path, fw_path):
    questionVocab = set()
    actionVocab = set()
    actionVocab_list = list()
    with open(infile1_path, 'r', encoding="UTF-8") as infile1, open(infile2_path, 'r', encoding="UTF-8") as infile2, open(infile3_path, 'r', encoding="UTF-8") as infile3:
        count = 0
        while True:
            lines_gen = list(islice(infile1, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                questionVocab.add(token)
            count = count + 1
            print(count)
        count = 0
        while True:
            lines_gen = list(islice(infile2, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                questionVocab.add(token)
            count = count + 1
            print(count)
        count = 0
        while True:
            lines_gen = list(islice(infile3, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                questionVocab.add(token)
            count = count + 1
            print(count)
    with open(infile4_path, 'r', encoding="UTF-8") as infile1, open(infile5_path, 'r', encoding="UTF-8") as infile2:
        count = 0
        while True:
            lines_gen = list(islice(infile1, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                actionVocab.add(token)
            count = count + 1
            print(count)
        count = 0
        while True:
            lines_gen = list(islice(infile2, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                actionVocab.add(token)
            count = count + 1
            print(count)
    action_size = 0
    for word in actionVocab:
        if word not in questionVocab and word not in special_characters:
            actionVocab_list.append(word)
            action_size += 1
        elif word in special_characters:
            actionVocab_list.append(word)
            action_size += 1
            if word in questionVocab:
                questionVocab.remove(word)
    questionVocab_list = list(questionVocab)
    share_vocab_list = actionVocab_list + questionVocab_list
    for i in range(len(share_vocab_list)):
        share_vocab_list[i] = share_vocab_list[i] + '\n'
    fw = open(fw_path, 'w', encoding="UTF-8")
    fw.writelines(share_vocab_list)
    fw.close()
    print("Writing %s is done!" %fw_path)
    return action_size


# Get all questions and answers in CQA-10K data-set for REINFORCEMENT learning by using denotations.
# Also treat the JSON as the file of training set used for REINFORCEMENT learning with denotations.
def rl_truereward_training_sets_for_10k(withint):
    training_folder_path = '../../data/official_downloaded_data/10k/train_10k'
    if withint:
        json_path = '../../data/auto_QA_data/CSQA_DENOTATIONS_full_INT.json'
        dict_path = '../../data/auto_QA_data/mask/dic_rl_tr_INT.question'
    else:
        json_path = '../../data/auto_QA_data/CSQA_DENOTATIONS_full.json'
        dict_path = '../../data/auto_QA_data/mask/dic_rl_tr.question'
    getAllQustionsAndAnswers(json_path, training_folder_path, dict_path, withint)


# Get all questions and answers in CQA-full data-set for REINFORCEMENT learning by using denotations.
# Also treat the JSON as the file of training set used for REINFORCEMENT learning with denotations.
def rl_truereward_training_sets_for_944k(withint):
    training_folder_path = '../../data/official_downloaded_data/944k/train_944k'
    if withint:
        json_path = '../../data/auto_QA_data/CSQA_DENOTATIONS_full_944K_INT.json'
        dict_path = '../../data/auto_QA_data/mask/dic_rl_tr_944K_INT.question'
    else:
        json_path = '../../data/auto_QA_data/CSQA_DENOTATIONS_full_944K.json'
        dict_path = '../../data/auto_QA_data/mask/dic_rl_tr_944K.question'
    getAllQustionsAndAnswers(json_path, training_folder_path, dict_path, withint)


def get_vocabulary_for_10k(withint):
    if withint:
        infile1_path = '../../data/auto_QA_data/mask/dic_py_INT.question'
        infile2_path = '../../data/auto_QA_data/mask/dic_rl_INT.question'
        infile3_path = '../../data/auto_QA_data/mask/dic_rl_tr_INT.question'
        infile4_path = '../../data/auto_QA_data/mask/dic_py_INT.action'
        infile5_path = '../../data/auto_QA_data/mask/dic_rl_INT.action'
        fw_path = '../../data/auto_QA_data/share_INT.question'
        print(getShareVocabulary(infile1_path, infile2_path, infile3_path, infile4_path, infile5_path, fw_path))

    else:
        infile1_path = '../../data/auto_QA_data/mask/dic_py.question'
        infile2_path = '../../data/auto_QA_data/mask/dic_rl.question'
        infile3_path = '../../data/auto_QA_data/mask/dic_rl_tr.question'
        infile4_path = '../../data/auto_QA_data/mask/dic_py.action'
        infile5_path = '../../data/auto_QA_data/mask/dic_rl.action'
        fw_path = '../../data/auto_QA_data/share.question'
        print(getShareVocabulary(infile1_path, infile2_path, infile3_path, infile4_path, infile5_path, fw_path))

def get_vocabulary_for_944k(withint):
    if withint:
        infile1_path = '../../data/auto_QA_data/mask/dic_py_INT.question'
        infile2_path = '../../data/auto_QA_data/mask/dic_rl_INT.question'
        infile3_path = '../../data/auto_QA_data/mask/dic_rl_tr_944k_INT.question'
        infile4_path = '../../data/auto_QA_data/mask/dic_py_INT.action'
        infile5_path = '../../data/auto_QA_data/mask/dic_rl_INT.action'
        fw_path = '../../data/auto_QA_data/share_944K_INT.question'
        print(getShareVocabulary(infile1_path, infile2_path, infile3_path, infile4_path, infile5_path, fw_path))

    else:
        infile1_path = '../../data/auto_QA_data/mask/dic_py.question'
        infile2_path = '../../data/auto_QA_data/mask/dic_rl.question'
        infile3_path = '../../data/auto_QA_data/mask/dic_rl_tr_944k.question'
        infile4_path = '../../data/auto_QA_data/mask/dic_py.action'
        infile5_path = '../../data/auto_QA_data/mask/dic_rl.action'
        fw_path = '../../data/auto_QA_data/share_944k.question'
        print(getShareVocabulary(infile1_path, infile2_path, infile3_path, infile4_path, infile5_path, fw_path))


# Run getAllQuestionsAndActions to get the JSON file which contains all information related to questions.
# Run getAllQustionsAndAnswers() to get the JSON file which contains all information and answers related to questions.
# Run getTrainingDatasetForPytorch() to get training and test processDataset for seq2seq model training.
# Run getTrainingDatasetForRl() to get training and test processDataset for REINFORCE-seq2seq model training.
# Run getShareVocabulary() to get vocabulary in all training processDataset.
if __name__ == "__main__":
    # The annotations is mangled, so a new file is created from the initial annotation.json file.
    # getNewAnnotationsForLogical()
    # If 'withint' is 'True', the int info is combined in the input sequence, otherwise not.
    # getAllQuestionsAndActions(withint=True)
    # getTrainingDatasetForPytorch(withint=True)
    # getTrainingDatasetForRl(withint=True)
    # rl_truereward_training_sets_for_10k(withint=True)
    # rl_truereward_training_sets_for_944k(withint=True)
    # get_vocabulary_for_10k(withint=True)
    # get_vocabulary_for_944k(withint=True)
    # getShareVocabularyForWebQSP()
    r_path = '../../data/saves/maml_reptile/sample_final_maml_predict.actions'
    w_path = '../../data/saves/maml_reptile/processed_sample_final_maml_predict.actions'
    getPurePredictedActions(r_path, w_path)




