# -*- coding: utf-8 -*-
import json
final_data_RL_train = {}
mask_path_RL = '../../data/webqsp_data/test/'

with open(mask_path_RL + 'final_webqsp_train_RL.json', "r", encoding='UTF-8') as correct_list:
    final_data_RL_train = json.load(correct_list)


kg = {}

kg = json.load(open('../../data/webqsp_data/webQSP_freebase_subgraph.json'))

print(len(kg))
def get_triple_kg(e, r):
    if e in kg and r in kg[e]:
        return kg[e][r]
    return []

data_list = []

fileObject = open('data_list.json', 'w')

i = 0
sub_kg = {}
for key, value in final_data_RL_train.items():
    # if i >= 100:
    #     break
    i += 1
    print(key)
    cur_kg = {}

    entity_list = value["entity"]
    relation_list = value["relation"]
    type_list = value["type"]

    for e in entity_list:
        e_dict = {}
        # sub_kg[e] = {}
        for r in relation_list:
            response_types = get_triple_kg(e, r)
            if len(response_types) > 0:
                e_dict.update({r: response_types})
        if len(e_dict) > 0:
            sub_kg.update({e: e_dict})

    cur_kg["kg"] = sub_kg
    cur_kg["name"] = key
    cur_kg["num_props"] = []
    cur_kg["datetime_props"] = []
    cur_kg["row_ents"] = list(set(entity_list + type_list))
    cur_kg["props"] = relation_list

    cur_kg["kg"] = sub_kg
    # data_list.append(cur_kg)

    jsondata = json.dumps(cur_kg)
    fileObject.write(jsondata)
    fileObject.write('\n')
fileObject.close()




