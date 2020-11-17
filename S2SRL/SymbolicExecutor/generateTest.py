# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 23:44
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io
import json
import re
import fnmatch
import os

def load_qadata(qa_dir):
    """

    :param qa_dir: 预处理后的qa数据的文件夹，eg：/home/zhangjingyao/preprocessed_data_10k/test
    :return: 这个文件夹下面，问答数据的字典。最外层是序号: QA_1_
    """
    print("begin_load_qadata")
    qa_set = {}
    for root, dirnames, filenames in os.walk(qa_dir):
        if(dirnames == []):
            qa_id = root[root.rfind("_")+1:]
            qa_dict ={}
            for filenames in fnmatch.filter(filenames, '*.txt'):
                pattern = re.compile('QA_\d+_')
                keystr = re.sub(pattern,"", filenames).replace(".txt","")
                qa_dict[keystr] = open(root+"/"+filenames).readlines()
            qa_set[qa_id] = qa_dict
    print("load_qadata_success")
    return qa_set

def testGenerate():
    qa_set = load_qadata("/data/zjy/test_full")
    with open("CSQA_ANNOTATIONS_test.json", 'w', encoding="UTF-8") as test_json, open("test_entities.txt", 'r', encoding="UTF-8") as test_entities\
            , open("test_questions.txt", 'r', encoding="UTF-8") as test_questions, open("test_types.txt", 'r', encoding="UTF-8") as test_types \
            , open("test_relations.txt", 'r', encoding="UTF-8") as test_relations:
        count = 1


        question_dict = {}
        question_info_new = {}
        for entity,relation,type,question, in zip(test_entities, test_relations, test_types, test_questions):
            ID_string = "test" + str(count)
            entities = [e.strip() for e in  entity.strip().split("<t>") if e != '']
            relations = [e.strip() for e in  relation.strip().split("<t>") if e != '']
            types = [e.strip() for e in  type.strip().split("<t>") if e != '']
            question_str =  question
            cou = 0
            # response_entity = []
            orig_response = ""
            for k,v in qa_set.items():
                if "context_utterance" and "state" and "response_entities" and "orig_response" in v:
                    for question_context,question_state,r_entities,o_response in zip(v["context_utterance"],v["state"],v["response_entities"],v["orig_response"]):
                        if question_str == question_context:
                            #print(question_str,question_context)
                            response_entity = r_entities
                            orig_response = o_response
                            state = question_state
                            break
            ID_string = state.strip() + str(count)
            if(count%1000 == 0): print(count)

            ID_string = "".join(ID_string.split())
            question_info_new = {}
            question_info_new.setdefault('question', question.strip())
            question_info_new.setdefault('entity', entities)
            question_info_new.setdefault('relation', relations)
            question_info_new.setdefault('type', types)
            question_info_new.setdefault('response_entities', response_entity)
            question_info_new.setdefault('orig_response', orig_response)
            entity_maskID = {}
            if len(entities) != 0:
                for i, entity in enumerate(entities):
                    entity_maskID.setdefault(entity, 'ENTITY' + str(i + 1))
            question_info_new.setdefault('entity_mask', entity_maskID)
            relation_maskID = {}
            if len(relations) != 0:
                relation_index = 0
                for relation in relations:
                    relation = relation.replace('-', '')
                    if relation not in relation_maskID:
                        relation_index += 1
                        relation_maskID.setdefault(relation, 'RELATION' + str(relation_index))
            question_info_new.setdefault('relation_mask', relation_maskID)
            type_maskID = {}
            if len(types) != 0:
                for i, type in enumerate(types):
                    type_maskID.setdefault(type, 'TYPE' + str(i + 1))
            question_info_new.setdefault('type_mask', type_maskID)
            question_dict.setdefault(ID_string, question_info_new)
            count+=1
        test_json.writelines(json.dumps(question_dict, indent=1, ensure_ascii=False,sort_keys=False))





if __name__ == "__main__":
    testGenerate()