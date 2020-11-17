import os
import json
import pickle
from flask import Flask, request, jsonify
app = Flask(__name__)

def get_id(idx):
    return int(idx[1:])

def select(e,r,t):
    if r.startswith("-") and 'obj' in graph[get_id(e)] and r[1:] in graph[get_id(e)]['obj']:
        return [ee for ee in graph[get_id(e)]['obj'][r[1:]] if t in is_A(ee)]
    elif 'sub' in graph[get_id(e)] and r in graph[get_id(e)]['sub']:
        return [ee for ee in graph[get_id(e)]['sub'][r] if t in  is_A(ee)]
    elif 'obj' in graph[get_id(e)] and r in graph[get_id(e)]['obj']:
        return [ee for ee in graph[get_id(e)]['obj'][r] if t in is_A(ee)]
    else:
        return None

def find(e,r):
    #return find({e},relation)
    if 'sub' in graph[get_id(e)] and r in graph[get_id(e)]['sub']:
        return list(graph[get_id(e)]['sub'][r])
    else:
        return None
    
def find_reverse(e,r):
    #return find({e},reverse(relation))
    if 'obj' in graph[get_id(e)] and r in graph[get_id(e)]['obj']:
        return list(graph[get_id(e)]['obj'][r])
    else:
        return None

def is_A(e):
    #return type of entity
    return type_dict[get_id(e)]

def is_All(t):
    # return entities which type is t
    return par_dict[get_id(t)]

def select_All(et, r, t):
    content = {}
    if r.startswith("-"):
        r = r[1:]
        if graph is not None and par_dict is not None:
            keys = par_dict[get_id(et)]
            for key in keys:
                if 'obj' in graph[get_id(key)] and r in graph[get_id(key)]['obj']:
                    val = [ee for ee in graph[get_id(key)]['obj'][r] if t in is_A(ee)]
                    content[key] = val
    else:
        if graph is not None and par_dict is not None:
            keys = par_dict[get_id(et)]
            for key in keys:
                if 'sub' in graph[get_id(key)] and r in graph[get_id(key)]['sub']:
                    val = [ee for ee in graph[get_id(key)]['sub'][r] if t in is_A(ee)]
                    content[key] = val
    return content

@app.route('/post', methods = ['POST'])
def post_res():
    response={}
    jsonpack = request.json
    if jsonpack['op']=="find":
        response['content']=find(jsonpack['sub'],jsonpack['pre'])
    elif jsonpack['op']=="find_reverse":
        response['content']=find_reverse(jsonpack['obj'],jsonpack['pre'])
    elif jsonpack['op']=="is_A":
        response['content']=is_A(jsonpack['entity'])
    ##########################################
    elif jsonpack['op']=="select":
        response['content']=select(jsonpack['sub'],jsonpack['pre'],jsonpack['obj'])
    elif jsonpack['op']=="select_All":
        response['content']=select_All(jsonpack['sub'],jsonpack['pre'],jsonpack['obj'])
    elif jsonpack['op']=="is_All":
        response['content']=is_All(jsonpack['type'])
    return jsonify(response)
    

if __name__ == '__main__':
    print("loading knowledge base...")
    global graph
    graph=pickle.load(open('/data/wuwei/data/wikidata.pkl','rb'))
    print("graph Load done!")
    global type_dict
    type_dict=pickle.load(open('/data/wuwei/data/child_par.pkl','rb'))
    print("type_dict Load done!",len(type_dict))
    global par_dict
    par_dict = pickle.load(open('/data/wuwei/data/par_child.pkl', 'rb'))
    print("par_dict Load done!")
    app.run(host='10.201.20.85', port=5000, use_debugger=True)
