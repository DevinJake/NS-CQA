# -*- coding: utf-8 -*-
import json
import re

try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode
import requests


def get_id(idx):
    return int(idx[1:])


from flask import Flask, request, jsonify

app = Flask(__name__)
# Remote Server
# post_url = "http://10.201.34.3:5002/post"
# # local server
post_url = "http://127.0.0.1:5003/post"


class Symbolics_WebQSP():

    def __init__(self, seq, mode='online'):
        # local server
        if mode != 'online':
            print("loading local knowledge base...")
            self.freebase_kb = json.load(
                open('../../data/webquestionssp/webQSP_freebase_subgraph.json'))
            print("Loading knowledge is done, start the server...")
            app.run(host='127.0.0.1', port=5001, use_debugger=True)
        # remote server
        else:
            self.graph = None
            self.type_dict = None
        self.seq = seq
        self.answer = {}
        self.temp_variable_list = []  # to store temp variable
        self.temp_set = set([])
        self.temp_bool_dict = {}

    def executor(self):
        if len(self.seq) > 0:
            for symbolic in self.seq:
                # print("current answer:", self.answer)
                key = list(symbolic.keys())[0]
                if len(symbolic[key]) != 3:
                    continue
                e = symbolic[key][0].strip()
                r = symbolic[key][1].strip()
                t = str(symbolic[key][2]).strip()
                # The execution result from A1 is in dict format.
                # A1: Select(e，r，t)
                if ("A1" in symbolic):
                    try:
                        temp_result = self.select(e, r, t)
                        self.answer = temp_result
                        self.temp_bool_dict = temp_result
                    except:
                        print('ERROR! The action is Select(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A2: Select_e(e，r，t)
                elif ("A2" in symbolic):
                    try:
                        temp_result = self.select_e(e, r, t)
                        self.answer = temp_result
                        self.temp_bool_dict = temp_result
                    except:
                        print('ERROR! The action is Select_e(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A3: filter answer
                elif ("A3" in symbolic):
                    try:
                        # ?x ns:a ns:b
                        self.answer.update(self.filter_answer(e, r, t))
                    except:
                        print('ERROR! The action is filter_answer(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A4: Joint
                elif ("A4" in symbolic):
                    try:
                        self.answer.update(self.joint(e, r, t))
                    except:
                        print('ERROR! The action is joint_str(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A5: Filter not equal
                elif ("A5" in symbolic):
                    try:
                        self.answer = self.filter_not_equal(e, r, t)
                    except:
                        print('ERROR! The action is filter_not_equal(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A6: map_value
                elif ("A6" in symbolic):
                    try:
                        self.answer = self.map_value(e, r, t)
                    except:
                        print('ERROR! The action is map_value(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A7: order_value_limit: order e by value and get top n
                elif ("A7" in symbolic):
                    try:
                        self.answer = self.order_value_limit(e, r, t, False)
                    except:
                        print('ERROR! The action is order_value_limit(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A7: order_value_desc_limit: order desc e by value and get top n
                elif ("A8" in symbolic):
                    try:
                        self.answer = self.order_value_limit(e, r, t, True)
                    except:
                        print('ERROR! The action is order_value_desc_limit(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A9: Union(e，r，t)
                elif ("A9" in symbolic):
                    try:
                        self.answer.update(self.union(e, r, t))
                    except:
                        print('ERROR! The action is Inter(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                else:
                    print("wrong symbolic")
        return self.answer

    def print_answer(self):
        pass

    # A1
    def select(self, e, r, t):
        if e == "" or r == "" or t == "":
            return {}
        else:
            content = set([])
            try:
                json_pack = dict()
                json_pack['op'] = "execute_gen_set1"
                json_pack['sub_pre'] = [e, r]
                jsonpost = json.dumps(json_pack)
                # result_content = requests.post(post_url,json=json_pack)
                # print(result_content)
                # print(jsonpost)
                content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                if content is not None and content_result == 0:
                    content = set(content)
            except:
                print("ERROR for command: select_str(%s,%s,%s)" % (e, r, t))
            finally:
                if content is not None:
                    # Store records in set.
                    content = set(content)
                else:
                    content = set([])
                # A dict is returned whose key is the subject and whose value is set of entities.
                return {t: content}

    # A2
    def select_e(self, e, r, t):
        if e == "" or r == "" or t == "":
            return {}
        else:
            content = set([])
            try:
                json_pack = dict()
                json_pack['op'] = "execute_gen_e_set1"
                json_pack['pre_type'] = [r, t]
                jsonpost = json.dumps(json_pack)
                # result_content = requests.post(post_url,json=json_pack)
                # print(result_content)
                content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                if content is not None and content_result == 0:
                    content = set(content)
            except Exception as error:
                print("ERROR for command: select_e_str(%s,%s,%s)" % (e, r, t), error)
            finally:
                if content is not None:
                    # Store records in set.
                    content = set(content)
                else:
                    content = set([])
                # A dict is returned whose key is the subject and whose value is set of entities.
                return {e: content}

    # A3
    def filter_answer(self, e, r, t):
        intermediate_result = {}
        if e == "" or r == "" or t == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                # print ("start filter_answer")
                if "?" in e:
                    json_pack = dict()
                    json_pack['op'] = "get_filter_answer"
                    if e in self.answer:
                        json_pack['e'] = list(self.answer[e])
                    else:
                        json_pack['e'] = []
                    json_pack['r'] = r
                    json_pack['t'] = t
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    # print(content)
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {e: content}
            except:
                print("ERROR for command: filter_answer(%s,%s,%s)" % (e, r, t))
            finally:
                return intermediate_result

    # A4
    def joint(self, e, r, t):
        intermediate_result = {}
        if e == "" or r == "" or t == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if '?' in e and t != '?x':
                    json_pack = dict()
                    json_pack['op'] = "joint"
                    if e in self.answer:
                        json_pack['e'] = list(self.answer[e])
                    else:
                        json_pack['e'] = []
                    json_pack['r'] = r
                    json_pack['t'] = t
                    jsonpost = json.dumps(json_pack)
                    # result_content = requests.post(post_url,json=json_pack)
                    # print(result_content)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {t: content}
                if '?' in e and t == '?x':
                    # print(e, self.answer[e])
                    json_pack = dict()
                    json_pack['op'] = "get_joint_answer"
                    if e in self.answer:
                        json_pack['e'] = list(self.answer[e])
                    else:
                        json_pack['e'] = []
                    json_pack['r'] = r
                    jsonpost = json.dumps(json_pack)
                    # result_content = requests.post(post_url,json=json_pack)
                    # print(result_content)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {t: content}
            except:
                print("ERROR for command: joint_str(%s,%s,%s)" % (e, r, t))
            finally:
                return intermediate_result

    # A5
    def filter_not_equal(self, e, r, t):
        intermediate_result = {}
        if e == "" or t == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                # print("filter_not_equal")
                if e in self.answer:
                    answer_list = []
                    # print(t)
                    for answer_item in self.answer[e]:
                        if (answer_item != t):
                            answer_list.append(answer_item)
                    intermediate_result = {e: answer_list}
            except:
                print("ERROR for command: filter_not_equal(%s,%s,%s)" % (e, r, t))
            finally:
                return intermediate_result

    # A6
    def map_value(self, e, r, t):
        intermediate_result = {}
        if e == "" or r == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if "?" in e:
                    json_pack = dict()
                    json_pack['op'] = "map_value"
                    if e in self.answer:
                        json_pack['e'] = list(self.answer[e])
                    else:
                        json_pack['e'] = []
                    json_pack['r'] = r
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    # print(content)
                    if content is not None and content_result == 0:
                        content = content
                    else:
                        content = {}
                    intermediate_result = content
            except:
                print("ERROR for command: map_value(%s,%s,%s)" % (e, r, t))
            finally:
                return intermediate_result

    # A7 A8
    def order_value_limit(self, e, r, n, b_reverse):
        intermediate_result = {}
        if e == "" or r == "" or int(n) is None:
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                result_list = []
                e_list_dict = self.map_value(e, r, "")
                e_list = list(e_list_dict.items())

                if len(e_list) >= int(n):
                    e_list = sorted(e_list, key=lambda x: x[1], reverse=True)
                    e_list = e_list[:int(n)]

                    for v_key in e_list:
                        result_list.append(v_key[0])
                intermediate_result = {e: result_list}
            except:
                if b_reverse:
                    print("ERROR for command: order_value_limit(%s,%s,%s)" % (e, r, n))
                else:
                    print("ERROR for command: order_value_desc_limit(%s,%s,%s)" % (e, r, n))
            finally:
                return intermediate_result


    # A9
    def union(self, e, r, t):
        # print("A9:", e, r, t)
        intermediate_result = {}
        if e == "" or t == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if e in self.answer and t in self.answer:
                    union_set = set(self.answer[e]).union(set(self.answer[t]))
                    intermediate_result = {t: union_set}
                return {}
            except:
                print("ERROR for command: joint_str(%s,%s,%s)" % (e, r, t))
            finally:
                return intermediate_result

    # compare e by value for A7 and A8
    def compare_value(self, a, b):
        if a in self.answer and b in self.answer:
            return self.answer[a] > self.answer[b]
        return -1

    # A10
    def date_less_or_equal(self, e, r, date='2015-08-10'):
        intermediate_result = {}
        if e == "" or r == "" or date == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if "?" in e:
                    json_pack = dict()
                    json_pack['op'] = "execute_select_oper_date_lt"
                    if e in self.answer:
                        json_pack['e'] = list(self.answer[e])
                    else:
                        json_pack['e'] = []
                    json_pack['r'] = r
                    json_pack['date'] = date
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {e: content}
            except:
                print("ERROR for command: date_less_or_equal(%s, %s)" % (e, date))
            finally:
                return intermediate_result

    # A11
    def date_greater_or_equal(self, e, r, date='2015-08-10'):
        intermediate_result = {}
        if e == "" or r == "" or date == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if "?" in e:
                    json_pack = dict()
                    json_pack['op'] = "execute_select_oper_date_gt"
                    if e in self.answer:
                        json_pack['e'] = list(self.answer[e])
                    else:
                        json_pack['e'] = []
                    json_pack['r'] = r
                    json_pack['date'] = date
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {e: content}
            except:
                print("ERROR for command: date_less_or_equal(%s, %s)" % (e, date))
            finally:
                return intermediate_result

    def select_max_as(self, e, r, t):
        if e == "" or t == "" or r != "" or e not in self.answer:
            return {}
        max = -1
        for item in self.answer[e]:
            item_count = len(self.answer[item])
            if item_count > max:
                max = item_count
        return {t: set(max)}

    def select_all(self, et, r, t):
        # print("A2:", et, r, t)
        content = {}
        if et == "" or r == "" or t == "":
            return content
        if self.graph is not None and self.par_dict is not None:
            keys = self.par_dict[get_id(et)]
            for key in keys:
                if 'sub' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['sub']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['sub'][r] if self.is_A(ee) == t]
                elif 'obj' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['obj']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['obj'][r] if self.is_A(ee) == t]

                else:
                    content[key] = []
            return content
        else:
            json_pack = dict()
            json_pack['op'] = "select_All"
            json_pack['sub'] = et
            json_pack['pre'] = r
            json_pack['obj'] = t
            try:
                content_json = requests.post("http://10.201.34.3:5000/post", json=json_pack).json()
                if 'content' in content_json:
                    content = content_json['content']
            except:
                print("ERROR for command: select_all(%s,%s,%s)" % (et, r, t))
            # content = requests.post("http://127.0.0.1:5000/post", json=json_pack).json()['content']
            # for k, v in content.items():
            #   if len(v) == 0: content.pop(k)
            finally:
                if self.answer:
                    for k, v in self.answer.items():
                        # Combine the retrieved entities with existed retrieved entities related to same subject.
                        content.setdefault(k, []).extend(v)
                return content

    def is_bool(self, e):
        # print("A3: is_bool")
        if type(self.answer) == bool: return self.answer
        if self.temp_bool_dict == None: return False
        if type(self.temp_bool_dict) == dict:
            for key in self.temp_bool_dict:
                if (self.temp_bool_dict[key] != None and e in self.temp_bool_dict[key]):
                    return True
        return False

    def arg_min(self):
        # print("A4: arg_min")
        if not self.answer:
            return []
        if type(self.answer) != dict: return []
        minK = min(self.answer, key=lambda x: len(self.answer[x]))
        minN = len(self.answer[minK])
        min_set = [k for k in self.answer if len(self.answer[k]) == minN]
        self.temp_set = set(min_set)
        return min_set

    def arg_max(self):
        # print("A5: arg_max")
        if not self.answer:
            return []
        if type(self.answer) != dict: return []
        maxK = max(self.answer, key=lambda x: len(self.answer[x]))
        maxN = len(self.answer[maxK])
        return [k for k in self.answer if len(self.answer[k]) == maxN]

    def greater_than(self, e, r, t):
        content = self.answer
        if type(content) != dict: return []
        if e in content and not content[e] == None:
            N = len(content[e])
        else:
            N = 0
        return [k for k in self.answer if len(self.answer[k]) > N]

    # TODO: NOT TESTED!
    def less_than(self, e, r, t):
        content = self.answer
        if type(content) != dict: return []
        if e in content and not content[e] == None:
            N = len(content[e])
        else:
            N = 0
        return [k for k in self.answer if len(self.answer[k]) < N]

    # equal, or equal
    def filter_or_equal(self, e, r, t):
        if e in self.answer[e]:
            self.answer[e].add(t)
            return self.answer[e]
        else:
            return []

    def count(self, e=None):
        # print("A11:Count")
        try:
            # list or set
            if type(self.answer) == type([]) or type(self.answer) == type(set()):
                return len(self.answer)
            # dict
            if type(self.answer) == type({}):
                if e != '' and e:
                    if e not in self.answer and len(self.answer.keys()) == 1:
                        return len(self.answer.popitem())
                    elif e in self.answer:
                        return len(self.answer[e])
                else:
                    return len(self.answer.keys())
            # int
            if type(self.answer) == type(1):
                return self.answer
            else:
                return 0
        except:
            print("ERROR! THE ACTION IS count(%s)!" % e)
            return 0

    # TODO: NOT TESTED
    def at_least(self, N):
        # print("A12: at_least")
        # for k in list(self.answer):
        #     if len(self.answer[k]) <= int(N):
        #         self.answer.pop(k)
        # return self.answer
        answer_keys = []
        if type(self.answer) == dict:
            for k, v in self.answer.items():
                if len(v) >= int(N):
                    answer_keys.append(k)
        return answer_keys

    # TODO: NOT TESTED
    def at_most(self, N):
        # print("A13: at_most")
        answer_keys = []
        # for k, v in self.answer.items():
        #   if len(v) == 0: self.answer.pop(k)
        if type(self.answer) == dict:
            for k in list(self.answer):
                if len(self.answer[k]) <= int(N):
                    answer_keys.append(k)
        return answer_keys

    # TODO: NOT TESTED
    def equal(self, N):
        answer_keys = []
        if type(self.answer) == dict:
            for k, v in self.answer.items():
                # print k,len(v)
                if len(v) == int(N):
                    answer_keys.append(k)
        return answer_keys

    def both_a(self, e1, e2, r):
        intermediate_result = {}
        if e1 == "" or e2 == "" or r == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if e1 == 'ANSWER' and e2 != 'VARIABLE':
                    json_pack = dict()
                    json_pack['op'] = "both_a"
                    json_pack['e1'] = list(self.answer[e1])
                    json_pack['e2'] = list(self.answer[e2])
                    json_pack['r'] = r
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {'ANSWER': content}
            except:
                print("ERROR for command: both_a(%s,%s,%s)" % (e1, e2, r))
            finally:
                return intermediate_result


# action sequence
class Action():
    def __init__(self, action_type, e, r, t):
        self.action_type = action_type
        self.e = e
        self.r = r
        self.t = t

    def to_str(self):
        return "{{\'{0}\':[\'{1}\', \'{2}\', \'{3}\']}}".format(self.action_type, self.e, self.r, self.t)
