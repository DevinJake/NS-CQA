import os
import requests
from urllib import request
import pickle
import json
from tqdm import tqdm
def get_id(idx):
    return int(idx[1:])

def create_kb():
    entity_items=json.load(open('/data/zjy/csqa_data/wikidata_dir/items_wikidata_n.json'))

    max_id=0
    for idx in tqdm(entity_items,total=len(entity_items)):
        max_id=max(max_id,get_id(idx))

    graph =[{} for i in range(max_id+1)]  
    cont=0
    for idx in tqdm(entity_items,total=len(entity_items)):
        graph[get_id(idx)]['name']=entity_items[idx]
        
        
    sub_predict_obj=json.load(open('/data/zjy/csqa_data/wikidata_dir/wikidata_short_1.json'))
    for idx in tqdm(sub_predict_obj,total=len(sub_predict_obj)):
            for x in sub_predict_obj[idx]:
                sub_predict_obj[idx][x]=set(sub_predict_obj[idx][x])
            graph[get_id(idx)]['sub']=sub_predict_obj[idx]

    sub_predict_obj=json.load(open('/data/zjy/csqa_data/wikidata_dir/wikidata_short_2.json'))
    for idx in tqdm(sub_predict_obj,total=len(sub_predict_obj)):
            for x in sub_predict_obj[idx]:
                sub_predict_obj[idx][x]=set(sub_predict_obj[idx][x])        
            graph[get_id(idx)]['sub']=sub_predict_obj[idx]   

    obj_predict_sub=json.load(open('/data/zjy/csqa_data/wikidata_dir/comp_wikidata_rev.json'))
    for idx in tqdm(obj_predict_sub,total=len(obj_predict_sub)):
            for x in obj_predict_sub[idx]:
                obj_predict_sub[idx][x]=set(obj_predict_sub[idx][x])
            graph[get_id(idx)]['obj']=obj_predict_sub[idx] 
    pickle.dump(graph,open('/data/zjy/wikidata.pkl','wb'))
    
def create_entity_type():
    print("Creating entity_type dictionary...")
    '''
    Build a dictionary
    key: ids of entity
    values: ids of type
    '''
    dic=json.load(open('/data/zjy/csqa_data/wikidata_dir/par_child_dict.json'))
    max_id=0
    for d in tqdm(dic,total=len(dic)):
        for idx in dic[d]:
                max_id=max(max_id,get_id(idx))

    type_dict =[[] for i in range(max_id+1)]
    # par_dict  =['' for i in range(max_id+1)]
    # for d in dic:
    #     par_dict[get_id(d)] = dic[d]
    # pickle.dump(par_dict, open('/data/zjy/par_child.pkl', 'wb'))
    # print("pickle down")
    for d in dic:
        for idx in dic[d]:# BUG?
            type_dict[get_id(idx)].append(d)
    #pickle.dump(type_dict,open('/data/zjy/child_par.pkl','wb'))
    print("pickle down")
    return type_dict


def create_entity_type():
    print("Creating entity_type dictionary...")
    '''
    Build a dictionary
    key: ids of type
    values: ids of entities
    '''
    dic = json.load(open('data/kb/par_child_dict.json'))
    max_id = 0
    for d in tqdm(dic, total=len(dic)):
        for idx in dic[d]:
            max_id = max(max_id, get_id(idx))

    type_dict = ['' for i in range(max_id + 1)]
    for d in dic:
        for idx in dic[d]:
            type_dict[get_id(idx)] = d
    pickle.dump(type_dict, open('data/BFS/type_kb.pkl', 'wb'))

    return type_dict

if __name__ == "__main__":
    print("Building knowledge base....")
    #create_kb()
    create_entity_type()
