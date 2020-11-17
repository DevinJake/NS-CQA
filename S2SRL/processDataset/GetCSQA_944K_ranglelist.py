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

typelist_dict = {}
for type in typelist:
    typelist_dict[type] = 0

result_dict = []
range_dict = {}

with open("CSQA_result_question_type_count944k.json", "r", encoding='UTF-8') as CSQA_List:
    load_dict = json.load(CSQA_List)
    index = 0
    end = 0
    for key, value in load_dict.items():
        print(key)
        start = end
        end = start + len(value)
        print(start, end)
        range_dict.update({key: {"start": start, "end": end}})
        # print("all: ", len(value))
        for item_key in value:
            for questonkey, quesion in item_key.items():
                type_name = typelist[0]
                for type in typelist:
                    if type in key:
                        type_name = type
                        typelist_dict[type] += 1
                        # count_order_name = "{0}{1}".format(type, typelist_dict[type])
                        type_order_item = {questonkey : quesion}
                        result_dict.append(type_order_item)

    print(typelist_dict)

    with open('CSQA_result_question_type_count944k_orderlist.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

    with open('955k_rangeDict.json', 'w') as f:
        json.dump(range_dict, f, indent=2)

