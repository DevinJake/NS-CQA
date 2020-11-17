# -*- coding: utf-8 -*-
# @Time    : 2019/1/18 14:52

try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode

import pickle
import requests
def get_id(idx):
    return int(idx[1:])

class Symbolics():

    def __init__(self, seq, mode='online'):
        if mode != 'online':
            print("loading knowledge base...")
            self.graph = pickle.load(open('/data/zjy/wikidata.pkl', 'rb'))
            self.type_dict = pickle.load(open('/data/zjy/type_kb.pkl', 'rb'))
            print("Load done!")
        else:
            self.graph = None
            self.type_dict = None

        self.seq = seq
        self.answer = {}
        self.temp_set = set([])
        self.temp_bool_dict = {}

    def executor(self):
        for symbolic in self.seq:
            key = list(symbolic.keys())[0]
            if len(symbolic[key]) != 3:
                continue
            e = symbolic[key][0].strip()
            r = symbolic[key][1].strip()
            t = symbolic[key][2].strip()
            # The execution result from A1 is in dict format.
            if ("A1" in symbolic):
                temp_result = self.select(e, r, t)
                self.answer = temp_result
                self.temp_bool_dict = temp_result
                self.print_answer()
            elif ("A2" in symbolic or "A16" in symbolic):
                self.answer = self.select_all(e, r, t)
                self.print_answer()
            elif ("A3" in symbolic):
                bool_temp_result = self.is_bool(e)
                if '|BOOL_RESULT|' in self.answer:
                    self.answer['|BOOL_RESULT|'].append(bool_temp_result)
                else:
                    temp = [bool_temp_result]
                    self.answer.setdefault('|BOOL_RESULT|', temp)
                self.print_answer()
            elif ("A4" in symbolic):
                self.answer = self.arg_min()
                self.print_answer()
            elif ("A5" in symbolic):
                self.answer = self.arg_max()
                self.print_answer()
            elif ("A6" in symbolic):
                self.answer = self.greater_than(e,r,t)
                self.print_answer()
            elif ("A7" in symbolic):
                self.answer = self.less_than(e,r,t)
                self.print_answer()
            elif ("A9" in symbolic):
                self.answer = self.union(e, r, t)
                self.print_answer()
            elif ("A8" in symbolic):
                self.answer = self.inter(e, r, t)
                self.print_answer()
            elif ("A10" in symbolic):
                self.answer = self.diff(e, r, t)
                self.print_answer()
            elif ("A11" in symbolic):
                self.answer = self.count(e)
                self.print_answer()
            elif ("A12" in symbolic):
                self.answer = self.at_least(e)
                self.print_answer()
            elif ("A13" in symbolic):
                self.answer = self.at_most(e)
                self.print_answer()
            elif ("A14" in symbolic):
                self.answer = self.equal(e)
                self.print_answer()
            elif ("A15" in symbolic):
                if r == "" and t == "":
                    self.answer = self.around(e)
                else:
                    self.answer = self.around(e,r,t)
                self.print_answer()
            elif ("A17" in symbolic):
                self.print_answer()
            else:
                print("wrong symbolic")

        return self.answer

    def is_A(self,e):
        #return type of entity
        if self.type_dict is not None:
            try:
                return self.type_dict[get_id(e)]
            except:
                return "empty"
        else:
            json_pack = dict()
            json_pack['op']="is_A"
            json_pack['entity']=e
            content = requests.post("http://10.201.34.3:5000/post", json=json_pack).json()['content']
            # content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']
            return content

    def select(self, e, r, t):
        if e == "":
            return {}
        if not ('Q' in e and 'Q' in t and 'P' in r):
            return {}
        if self.graph is not None:
            if 'sub' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['sub']:
                return {e:[ee for ee in self.graph[get_id(e)]['sub'][r] if self.is_A(ee) == t]}
            elif 'obj' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['obj']:
                return {e:[ee for ee in self.graph[get_id(e)]['obj'][r] if self.is_A(ee) == t]}
            else:
                return {}
        else:
            json_pack = dict()
            json_pack['op'] = "select"
            json_pack['sub'] = e
            json_pack['pre'] = r
            json_pack['obj'] = t
            # content = requests.post("http://127.0.0.1:5000/post", json=json_pack).json()['content']
            content = requests.post("http://10.201.34.3:5000/post", json=json_pack).json()['content']
            if content is not None:
                # Store records in set.
                content = set(content)
            else:
                content = set([])
            # A dict is returned whose key is the subject and whose value is set of entities.
            return {e:content}

    def select_all(self, et, r, t):
        #print("A2:", et, r, t)
        content = {}
        if et == "" or r =="" or t =="":
            return {}
        elif not ('Q' in et and 'Q' in t and 'P' in r):
            return {}
        if self.graph is not None and self.par_dict is not None:
            keys = self.par_dict[get_id(et)]
            for key in keys:
                if 'sub' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['sub']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['sub'][r] if self.is_A(ee) == t]
                elif 'obj' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['obj']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['obj'][r] if self.is_A(ee) == t]

                else:
                    content[key] = None
        else:
            json_pack = dict()
            json_pack['op'] = "select_All"
            json_pack['sub'] = et
            json_pack['pre'] = r
            json_pack['obj'] = t

        content = requests.post("http://10.201.34.3:5000/post", json=json_pack).json()['content']
        # content = requests.post("http://127.0.0.1:5000/post", json=json_pack).json()['content']
        # for k, v in content.items():
        #   if len(v) == 0: content.pop(k)

        if self.answer:
            for k, v in self.answer.items():
                # Combine the retrieved entities with existed retrieved entities related to same subject.
                content.setdefault(k, []).extend(v)
        return content

    def is_bool(self, e):
        # print("A3: is_bool")
        if type(self.answer) == bool: return True
        if self.temp_bool_dict == None: return False
        for key in self.temp_bool_dict:
            if (self.temp_bool_dict[key] != None and e in self.temp_bool_dict[key]):
                return True
        return False

    def arg_min(self):
        # print("A4: arg_min")
        if not self.answer:
            return None
        minK = min(self.answer, key=lambda x: len(self.answer[x]))
        minN = len(self.answer[minK])
        min_set = [k for k in self.answer if len(self.answer[k]) == minN]
        self.temp_set = set(min_set)
        return min_set

    def arg_max(self):
        # print("A5: arg_max")
        if not self.answer:
            return None
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

    def less_than(self, e, r, t):
        content = self.answer
        if type(content) != dict: return []
        if e in content and not content[e] == None:
            N = len(content[e])
        else:
            N = 0
        return [k for k in self.answer if len(self.answer[k]) < N]

    def union(self, e, r, t):
        #print("A9:", e, r, t)
        if e == "": return {}
        if t!="" and 'Q' not in t: return{}
        answer_dict = self.answer
        if type(answer_dict) == bool: return False
        if e in answer_dict and answer_dict[e]!=None:
            temp_set = self.select(e, r, t)
            if e in temp_set:
                answer_dict[e] = set(answer_dict[e]) | set(temp_set[e])
        else:
            answer_dict.update(self.select(e, r, t))

        # 进行 union 操作 todo 这里前面都和select部分一样 所以还是应该拆开？ union单独做 好处是union可以不止合并两个 字典里的都可以合并
        union_key = ""
        union_value = set([])
        for k, v in answer_dict.items():
            if v == None: v = []
            union_value = union_value | set(v)
        union_key = "|"
        answer_dict.clear()
        answer_dict[union_key] = list(set(union_value))

        return answer_dict

    def inter(self, e, r, t):
        #print("A8:", e, r, t)
        if e == "": return {}
        if not e.startswith("Q"): return {}
        answer_dict = self.answer
        if e in answer_dict and answer_dict[e]!=None:
            temp_set = self.select(e, r, t)
            if e in temp_set:
                answer_dict[e] = set(answer_dict[e]) & set(temp_set[e])
        else:
            s = self.select(e, r, t)
            answer_dict.update(s)

        # 进行 inter 类似 union
        inter_key = ""
        inter_value = set([])
        for k, v in answer_dict.items():
            if v == None: v = []
            if len(inter_value) > 0:
                inter_value = inter_value & set(v)
            else:
                inter_value = set(v)

        answer_dict.clear()
        inter_key = "&"
        answer_dict[inter_key] = list(set(inter_value))

        return answer_dict

    def diff(self, e, r, t):
        #print("A10:", e, r, t)
        if e == "": return {}
        answer_dict = self.answer
        if e in answer_dict and answer_dict[e]!=None:
            temp_set = self.select(e, r, t)
            if e in temp_set:
                answer_dict[e] = set(answer_dict[e]) - set(temp_set[e])
        else:
            answer_dict.update(self.select(e, r, t))
        # 进行 diff 操作 类似 union
        diff_key = ""
        diff_value = set([])
        for k, v in answer_dict.items():
            if v == None: v = []
            if k != e:
                diff_value.update(set(v))
        if(answer_dict[e]):
            diff_value = diff_value - set(answer_dict[e])

        answer_dict.clear()
        diff_key = "-"
        answer_dict[diff_key] = list(set(diff_value))

        return answer_dict

    def count(self,e= None):
        #print("A11:Count")
        if type(self.answer) == type([]):
            return len(self.answer)
        elif type(self.answer) != type({}):
            return 0
        elif e!='' and e and e in self.answer:
            if e not in self.answer and len(self.answer.keys()) == 1:
                return len(self.answer.popitem())
            return len(self.answer[e])
        else:
            return len(self.answer.keys()) if type(self.answer) == type({}) else 0

        return 0

    def at_least(self, N):
        # print("A12: at_least")
        # for k in list(self.answer):
        #     if len(self.answer[k]) <= int(N):
        #         self.answer.pop(k)
        # return self.answer
        answer_keys = []
        for k, v in self.answer.items():
            if len(v) >= int(N):
                answer_keys.append(k)
        return answer_keys

    def at_most(self, N):
        # print("A13: at_most")
        answer_keys = []
        # for k, v in self.answer.items():
        #   if len(v) == 0: self.answer.pop(k)
        for k in list(self.answer):
            if len(self.answer[k]) <= int(N):
                answer_keys.append(k)
        return answer_keys

    def equal(self, N):
        answer_keys = []
        for k, v in self.answer.items():
            #print k,len(v)
            if len(v) == int(N):
                answer_keys.append(k)
        return answer_keys

    def around(self,N,r=None,t=None):
        number = 0
        if(r!=None) and (t!=None):
            e = N
            dict_temp = self.select_all(e,r,t)
            number = len(dict_temp)
        elif N.isdigit():
            number = int(N)
        else:
            content = self.answer
            if type(content) == dict:
                if N in content and not content[N] == None:
                    number = len(content[N])
                else:
                    number = 0
        answer_keys = []
        if type(self.answer) == type({}):
            if number == 0 or 1< number <=5:
                for k, v in self.answer.items():
                    if abs(len(v)-int(number)) <= 1:
                        answer_keys.append(k)
            if number == 1:
                for k, v in self.answer.items():
                    if abs(len(v) - int(number)) < (int(number) * 0.6):
                        answer_keys.append(k)
            elif number > 5:
                for k, v in self.answer.items():
                    if abs(len(v)-int(number)) <= 5:
                        answer_keys.append(k)
            else:
                for k, v in self.answer.items():
                    # print k, len(v),abs(len(v)-int(N)),(int(N)/2)
                    if abs(len(v)-int(number)) < (int(number)*0.6):
                        answer_keys.append(k)
            self.temp_set = set(answer_keys)
        return answer_keys

    def EOQ(self):
        pass

    ########################
    def print_answer(self):
        pass
        # if(type(self.answer) == dict):
        #     for k,v in self.answer.items():
        #         #print self.item_data[k],": ",
        #         for value in v:
        #         #    print self.item_data[value], ",",
        #         print
        # elif(type(self.answer) == type([])):
        #     for a in self.answer:
        #         print self.item_data[a],
        #     print
        # else:
        #     if(self.answer in self.item_data):
        #         print self.answer,self.item_data[self.answer]
        #     else:
        #         print self.answer
    # print("----------------")

    def select_sparql(self, e, r, t):  # use sparql
        answer_dict = {}
        anser_values = []
        sparql = {"query": "SELECT ?river WHERE { \
                                            ?river wdt:" + r + " wd:" + e + ". \
                                            ?river wdt:P31  wd:" + t + ". \
                                       }",
                  "format": "json",
                  }
        # print sparql
        sparql = urlencode(sparql)
        # print sparql
        url = 'https://query.wikidata.org/sparql?' + sparql
        r = requests.get(url)
        # print r.json()["results"]
        for e in r.json()["results"]["bindings"]:
            entity = e["river"]["value"].split("/")[-1]
            anser_values.append(entity)
        answer_dict[e] = anser_values

        return answer_dict

if __name__ == "__main__":
    print("Building knowledge base....")
    kb = Symbolics(None,'online')
    # for e in kb.find('Q2619632', 'P138'):
    #     print(e,kb.is_A(e))
    # 'e' is the keys of returned dictionary.
    result_dict = kb.select('Q2619632', 'P138', 'Q355304')
    for e in result_dict:
        # 'val' is the set corresponding to the key.
        val = result_dict[e]
        for v in val:
            print(v, kb.is_A(v))

