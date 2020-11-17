# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 21:50
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io
import json
from urllib.parse import urlencode

import requests
from tqdm import tqdm

# def get_id(idx):
#     return int(idx[1:])
#
# entity_items = json.load(open('/data/zjy/csqa_data/wikidata_dir'+'/items_wikidata_n.json'))
# # pickle.dump(entity_items, open('/home/zhangjingyao/data/entity_items.pkl', 'wb'))
# max_id = 0
# for idx in tqdm(entity_items, total=len(entity_items)):
#     max_id = max(max_id, get_id(idx))
#
# graph = [{} for i in range(max_id + 1)]
# cont = 0
# for idx in tqdm(entity_items, total=len(entity_items)):
#     graph[get_id(idx)]['name'] = entity_items[idx]
#
# child_par=pickle.load(open('/data/zjy/child_par.pkl','rb'))
# print("child_par Load done!",len(child_par))
#
# names = {}
# for id in range(0,len(child_par)):
#     if not child_par[id] == None and not len(child_par[id]) == 0:
#
#         names.setdefault(graph[id]['name'],[]).append(id)
#
# a = 'Is Tepic present in Heretsried and Mexico ? '
# for k,v in names.items():
#     if k in a:
#         print(k, v)

def test_sparql(e='Q148',r = 'P17',t = 'Q4022'):
    answer_dict = {}
    anser_values = []
    sparql = {"query": "SELECT ?river WHERE { \
                                ?river wdt:" + r + " wd:"+ e +". \
                                ?river wdt:P31  wd:"+ t + ". \
                           }",
              "format" : "json",
              }

    sparql =  urlencode(sparql)

    url = 'https://query.wikidata.org/sparql?'+sparql
    r = requests.get(url)
    #print r.json()["results"]
    for e in r.json()["results"]["bindings"]:
        entity =  e["river"]["value"].split("/")[-1]
        anser_values.append(entity)
    answer_dict[e] = anser_values

test_sparql()
