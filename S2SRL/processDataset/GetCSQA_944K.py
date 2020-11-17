# -*- coding: utf-8 -*-
import json

typelist = ['Simple Question (Direct)_',
            'Verification (Boolean) (All)_',
            'Quantitative Reasoning (Count) (All)_',
            'Logical Reasoning (All)_',
            'Comparative Reasoning (Count) (All)_',
            'Quantitative Reasoning (All)_',
            'Comparative Reasoning (All)_'
            ]
result_dict = {}

with open("CSQA_DENOTATIONS_full_155K.json", "r", encoding='UTF-8') as CSQA_List:
    load_dict = json.load(CSQA_List)
    for key, value in load_dict.items():
        entity_count = len(value['entity'])
        relation_count = len(value['relation'])
        type_count = len(value['type'])
        relation_list = value['relation']
        question = value['question']

        relation_str = '_'.join(relation_list)

        type_name = typelist[0]
        for type in typelist:
            if type in key:
                type_name = type
        key_name = '{0}{1}_{2}_{3}'.format(type_name, entity_count, relation_count, type_count)
        if key_name in result_dict:
            key_question = {key: question}
            result_dict[key_name].append(key_question)
        else:
            new_item = {key_name: [key]}
            key_question = {key: question}
            new_item = {key_name: [key_question]}
            result_dict.update(new_item)

    allcount = 0
    for key, value in result_dict.items():
        allcount += len(value)
        print(key, value[0], len(value))
    print(allcount)
    with open('CSQA_result_question_type_count944k.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
