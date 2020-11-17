# coding=utf-8
import os
import json
import pickle
import numpy as np
import datetime
import json
import msgpack
from flask import Flask, request, jsonify
app = Flask(__name__)


class Interpreter():
    def __init__(self, freebase_dir):
        self.map_program_to_func = {}
        self.map_program_to_func["gen_set1"] = self.execute_gen_set1
        self.map_program_to_func["gen_set1_e"] = self.execute_gen_e_set1
        self.map_program_to_func["gen_set2"] = self.execute_gen_set2
        self.map_program_to_func["gen_set1_date"] = self.execute_gen_set1
        self.map_program_to_func["select_oper_date_lt"] = self.execute_select_oper_date_lt
        self.map_program_to_func["select_oper_date_gt"] = self.execute_select_oper_date_gt
        self.map_program_to_func["gen_set2_dateconstrained"] = self.execute_gen_set2_dateconstrained
        self.map_program_to_func["set_oper_ints"] = self.execute_set_oper_ints
        self.map_program_to_func["joint"] = self.execute_joint
        self.map_program_to_func["none"] = self.execute_none
        self.map_program_to_func["terminate"] = self.execute_terminate

    # e, r包含在图谱中
    def is_kb_consistent(self, e, r):
        # print("find", e, r)
        if e in self.freebase_kb and r in self.freebase_kb[e]:
            # print("find", e, r, self.freebase_kb[e][r])
            return True
        else:
            return False

    # e,r,t 包含在图谱中
    def exist(self, e, r, t):
        if e in self.freebase_kb and r in self.freebase_kb[e] and t in self.freebase_kb[e][r]:
            return True
        else:
            return False

    # e,r,t 或者 t,r,e包含在图谱中
    def gen_exist(self, e, r, t):
        if t in self.freebase_kb and r in self.freebase_kb[t] and e in self.freebase_kb[t][r]:
            return 2
        elif self.exist(e, r, t):
            return 1
        else:
            return -1

    # 通过实体-关系 查找 三元组 类似select
    def execute_gen_set1(self, argument_value, argument_location):
        entity = argument_value[0]
        relation = argument_value[1]
        if entity is None or relation is None:
            return set([]), 1
        tuple_set = None
        if entity in self.freebase_kb and relation in self.freebase_kb[entity]:
            tuple_set = self.freebase_kb[entity][relation]
            # print("A1 select", entity, relation, tuple_set)
        return tuple_set, 0

    # 通过谓语宾语找主语
    def execute_gen_e_set1(self, argument_value, argument_location):
        relation = argument_value[0]
        type = argument_value[1]
        if relation is None or type is None:
            return [], 1
        entity_list = []
        for entity in self.freebase_kb:
            if relation in self.freebase_kb[entity] and type in self.freebase_kb[entity][relation]:
                entity_list.append(entity)
                # print("A2 select_e", entity, relation, tuple_set)
        # if self.is_kb_consistent(type, relation):
        #     tuple_set.append(self.freebase_kb[type][relation])
        #         # print("A2 select_e", entity, relation, tuple_set)
        return entity_list, 0

    # # 通过实体-关系 查找所有时间三元组
    # def execute_gen_set1_date(self, argument_value, argument_location):
    #     entity = argument_value[0]
    #     relation_date = argument_value[1]
    #     if entity is None or relation_date is None:
    #         return set([]), 1
    #     tuple_set = None
    #     if entity in self.freebase_kb and relation_date in self.freebase_kb[entity]:
    #         tuple_set = {d: entity for d in self.freebase_kb[entity][relation_date]}
    #     return tuple_set, 0

    # 2跳 参数1 [主语, 谓语1] 参数2 [谓语2] -- 通过主语谓语1得到宾语实体list 宾语实体下 如果有谓语2的关系 返回此实体-关系查询结果
    def execute_gen_set2(self, argument_value, argument_location):
        set_ent, _ = self.execute_gen_set1(argument_value, argument_location)
        relation = argument_value[2]
        if set_ent is None or relation is None:
            return set([]), 1
        tuple_set = None
        for e in set_ent:
            if e in self.freebase_kb and relation in self.freebase_kb[e]:
                if tuple_set is None:
                    tuple_set = set(self.freebase_kb[e][relation])
                else:
                    tuple_set.update(set(self.freebase_kb[e][relation]))
        return tuple_set, 0

    # tails中有没有和y同一年的
    def same_year(self, tails, y):
        for t in tails:
            t = self.convert_to_date(t).year
            if t == y:
                return True
        return False

    # 时间2跳 参数1-主语 参数2-宾语  返回宾语list结果 ->  参数3-关系 参数4-时间  参数5-时间 对象时间的年 一个2跳的关于时间的
    def execute_gen_set2_dateconstrained(self, argument_value, argument_location):
        set_ent, _ = self.execute_gen_set1(argument_value, argument_location)
        relation = argument_value[2]
        constr_rel_date = argument_value[3]
        constr_date = argument_value[4]
        if set_ent is None or relation is None or constr_rel_date is None or constr_date is None:
            return set([]), 1
        constr_year = constr_date.year
        tuple_set = None
        for e in set_ent:
            if e in self.freebase_kb and constr_rel_date in self.freebase_kb[e] and self.same_year(
                    self.freebase_kb[e][constr_rel_date], constr_year):
                if relation not in self.freebase_kb[e]:
                    continue
                if tuple_set is None:
                    tuple_set = set(self.freebase_kb[e][relation])
                else:
                    tuple_set.update(set(self.freebase_kb[e][relation]))
        return tuple_set, 0

    # 并
    def execute_set_oper_ints(self, argument_value, argument_location):
        set_ent1 = argument_value[0]
        set_ent2 = argument_value[1]
        if set_ent1 is None or set_ent2 is None:
            return set([]), 1
        set_ent_ints = set(set_ent1).intersection(set(set_ent2))
        flag = 0
        if argument_location is not None:
            argument_location1 = argument_location[0]
            argument_location2 = argument_location[1]
            if argument_location1 == argument_location2:
                flag = 0.1
        return set_ent_ints, flag

    # 转为日期
    def convert_to_date(self, x):
        if x.startswith('m.'):
            return None
        if 'T' in x:
            x = x.split('T')[0]
        if len(x.split('-')) == 1:
            x = x + '-01-01'
        elif len(x.split('-')) == 2:
            x = x + '-01'
        try:
            yyyy = int(x.split('-')[0])
            mm = int(x.split('-')[1])
            dd = int(x.split('-')[2])
            d = datetime.datetime(yyyy, mm, dd)
            return d
        except:
            return None

    # 小于等于date时间的集合
    def execute_select_oper_date_lt(self, e, r, date):
        if e is None or r is None or date is None:
            return set([]), 1
        e_list = list(e)
        right_date_e_list = []
        for e_item in e_list:
            if self.is_kb_consistent(e_item, r):
                e_item_date_list = self.freebase_kb[e_item][r]
                if len(e_item_date_list) > 0:
                    if self.convert_to_date(e_item_date_list[0]) is not None and self.convert_to_date(date) is not None:
                        if self.convert_to_date(e_item_date_list[0]) <= self.convert_to_date(date):
                            right_date_e_list.append(e_item)
            else:
                right_date_e_list.append(e_item)
        return right_date_e_list, 0

    # 大于等于date时间的集合
    def execute_select_oper_date_gt(self, e, r, date):
        if e is None or r is None or date is None:
            return set([]), 1
        e_list = list(e)
        right_date_e_list = []
        for e_item in e_list:
            if self.is_kb_consistent(e_item, r):
                e_item_date_list = self.freebase_kb[e_item][r]
                if len(e_item_date_list) > 0:
                    if self.convert_to_date(e_item_date_list[0]) is not None and self.convert_to_date(date) is not None:
                        if self.convert_to_date(e_item_date_list[0]) >= self.convert_to_date(date):
                            right_date_e_list.append(e_item)
            else:
                right_date_e_list.append(e_item)
        return right_date_e_list, 0

    def execute_none(self, argument_value, argument_location):
        return None, 0

    def execute_terminate(self, argument_value, argument_location):
        return None, 0

    # e, r
    def execute_joint(self, e, r, t):
        temp_set = set([])
        try:
            if isinstance(e, list):
                for entity in e:
                    if self.is_kb_consistent(entity, r):
                        # print("execute_joint", entity, r, self.freebase_kb[entity][r])
                        temp_set.update(set(self.freebase_kb[entity][r]))
                return list(temp_set), 0
            else:
                return list(temp_set), 1
        except:
            print("Some error occurs in execute_joint action!")
            return list(temp_set), 1

    # TODO: NOT THROUGHLY TESTED!
    # A4
    def get_joint_answer(self, e, r):
        temp_set = set([])
        try:
            if isinstance(e, list) and len(e) > 0 and r is not None:
                for entity in e:
                    if entity in self.freebase_kb and r in self.freebase_kb[entity]:
                        # print("execute_joint", entity, r, self.freebase_kb[entity][r])
                        temp_set.update(set(self.freebase_kb[entity][r]))
                return list(temp_set), 0
            else:
                return list(temp_set), 0
        except:
            print("Some error occurs in get_joint_answer action!")
            return list(temp_set), 1

    # TODO: NOT THROUGHLY TESTED!
    def get_filter_answer(self, e, r, t):
        temp_set = set([])
        e_list = []
        try:
            if isinstance(e, list) and len(e) > 0 and r is not None:
                if len(e) > 100:
                    for entity in e:
                        if self.exist(entity, r, t):
                            # temp_set.update((self.freebase_kb[entity][r]))
                            # new_e = self.freebase_kb[t][r]
                            e_list.append(entity)
                else:
                    for entity in e:
                        if self.gen_exist(entity, r, t) != -1:
                            # temp_set.update((self.freebase_kb[entity][r]))
                            # new_e = self.freebase_kb[t][r]
                            e_list.append(entity)

                    # if self.exist(entity, r, t):
                    #     e_list.append(entity)

                return e_list, 0
            else:
                return e_list, 1
        except:
            print("Some error occurs in get_joint_answer action!")
            return list(temp_set), 1

    def map_value(self, e, r):
        temp_set = {}
        for e_item in e:
            if self.is_kb_consistent(e_item, r):
                temp_set[e_item] = self.freebase_kb[e_item][r]
            # else:
            #     temp_set[e_item] = "NONE"
        return temp_set, 0

@app.route('/post', methods=['POST'])
def post_res():
    response = {}
    jsonpack = json.loads(request.json)
    if jsonpack['op'] == "find":
        response['content'] = interpreter.is_kb_consistent(jsonpack['sub'], jsonpack['pre'])
    elif jsonpack['op'] == "execute_gen_set1":
        response['content'] = interpreter.execute_gen_set1(jsonpack['sub_pre'], "")
    elif jsonpack['op'] == "execute_gen_e_set1":
        response['content'] = interpreter.execute_gen_e_set1(jsonpack['pre_type'], "")
    elif jsonpack['op'] == "execute_gen_set2":
        response['content'] = interpreter.execute_gen_set2(jsonpack['sub_pre1_pre2'], "")
    elif jsonpack['op'] == "joint":
        response['content'] = interpreter.execute_joint(jsonpack['e'], jsonpack['r'], jsonpack['t'])
    elif jsonpack['op'] == "get_joint_answer":
        response['content'] = interpreter.get_joint_answer(jsonpack['e'], jsonpack['r'])
    elif jsonpack['op'] == "exist":
        response['content'] = interpreter.exist(jsonpack['sub'], jsonpack['pre'], jsonpack['obj'])
    elif jsonpack['op'] == "get_filter_answer":
        response['content'] = interpreter.get_filter_answer(jsonpack['e'], jsonpack['r'], jsonpack['t'])
    elif jsonpack['op'] == "execute_select_oper_date_lt":
        response['content'] = interpreter.execute_select_oper_date_lt(jsonpack['e'], jsonpack['r'], jsonpack['date'])
    elif jsonpack['op'] == "execute_select_oper_date_gt":
        response['content'] = interpreter.execute_select_oper_date_gt(jsonpack['e'], jsonpack['r'], jsonpack['date'])
    elif jsonpack['op'] == "map_value":
        response['content'] = interpreter.map_value(jsonpack['e'], jsonpack['r'])

    # str2entity

    # elif jsonpack['op']=="find_reverse":
    #     response['content']=find_reverse(jsonpack['obj'],jsonpack['pre'])
    # elif jsonpack['op']=="is_A":
    #     response['content']=is_A(jsonpack['entity'])
    # elif jsonpack['op']=="select":
    #     response['content']=interpreter.execute_gen_set1(jsonpack['sub'],jsonpack['pre'],jsonpack['obj'])
    # elif jsonpack['op']=="select_All":
    #     response['content']=interpreter.execute_gen_set1(jsonpack['sub'],jsonpack['pre'],jsonpack['obj'])
    # elif jsonpack['op']=="is_All":
    #     response['content']=is_All(jsonpack['type'])
    return jsonify(response)

if __name__ == '__main__':
    print("loading knowledge base...")
    interpreter = Interpreter("")
    interpreter.freebase_kb = json.load(
        open('../../data/webqsp_data/webQSP_freebase_subgraph.json'))
    print("loading knowledge down, start the server")
    app.run(host='127.0.0.1', port=5003, use_debugger=True)
    # app.run(host='10.201.34.3', port=5001, use_debugger=True)

    # # local server
    # print("loading knowledge base...")
    # interpreter = Interpreter("")
    # interpreter.freebase_kb = json.load(
    #     open('../../data/webquestionssp/webQSP_freebase_subgraph.json'))
    # print("loading knowledge down, start the server")
    # app.run(host='127.0.0.1', port=5001, use_debugger=True)
