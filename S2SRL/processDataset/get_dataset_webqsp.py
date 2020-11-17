# -*- coding: utf-8 -*-
import json
import sys
sys.path.insert(0, '../SymbolicExecutor/')
from symbolics_webqsp import Symbolics_WebQSP
from symbolics_webqsp_novar import Symbolics_WebQSP_novar
from itertools import islice
sys.path.insert(0, '../../S2SRL/')
from libbots import data, model, utils

b_print = True
LINE_SIZE = 100000
special_counting_characters = {'-','|','&'}
special_characters = {'(',')','-','|','&'}
mask_path = '../../data/webqsp_data/test/'
mask_path_RL = '../../data/webqsp_data/test/'
# action sequence
class Action():
    def __init__(self, action_type, e, r, t):
        self.action_type = action_type
        self.e = e
        self.r = r
        self.t = t

    def to_str(self):
        return "{{\'{0}\':[\'{1}\', \'{2}\', \'{3}\']}}".format(self.action_type, self.e, self.r, self.t)

class Qapair(object):
    def __init__(self, question, answer, sparql):
        self.question = question
        self.answer = answer
        self.sparql = sparql

    def obj_2_json(obj):
        return {
            "question": obj.question,
            "answer": obj.answer,
            "sparql": obj.sparql,
        }

class WebQSP(object):
    def __init__(self, id, question, action_sequence_list, entity, relation, type, entity_mask, relation_mask, type_mask, mask_action_sequence_list, answerlist, input_str, response_entities, orig_response_str, action_string, int_mask):
        self.id = id
        self.question = question
        self.action_sequence_list = action_sequence_list
        self.entity = entity
        self.relation = relation
        self.type = type
        self.entity_mask = entity_mask
        self.relation_mask = relation_mask
        self.type_mask = type_mask
        self.mask_action_sequence_list = mask_action_sequence_list
        self.answerlist = answerlist
        self.input_str = input_str
        self.response_entities = response_entities
        self.orig_response_str = orig_response_str
        self.action_string = action_string
        self.int_mask = int_mask

    def obj_2_rljson(obj):
        return {
            obj.id: {
                "question": obj.question,
                "action_sequence_list": obj.action_sequence_list,
                "entity": obj.entity,
                "relation": obj.relation,
                "type": obj.type,
                "entity_mask": obj.entity_mask,
                "relation_mask": obj.relation_mask,
                "type_mask": obj.type_mask,
                "answers": obj.answerlist,
                "input": obj.input_str,
                "response_entities": obj.response_entities,
                "response_bools": [],
                "orig_response": obj.orig_response_str,
                "pseudo_gold_program": obj.action_string,
                "int_mask": obj.int_mask
            }
        }

    def obj_2_s2sjson(obj):
        return {
            obj.id: {
                "question": obj.question,
                "action_sequence_list": obj.action_sequence_list,
                "entity": obj.entity,
                "relation": obj.relation,
                "type": obj.type,
                "entity_mask": obj.entity_mask,
                "relation_mask": obj.relation_mask,
                "type_mask": obj.type_mask,
                "mask_action_sequence_list": obj.mask_action_sequence_list,
                "answers": obj.answerlist,
                "input": obj.input_str,
                "response_entities": obj.response_entities,
                "orig_response_str": obj.orig_response_str,
                "pseudo_gold_program": obj.action_string,
                "int_mask": obj.int_mask
            }
        }

def getViriableNameByIndex(sparql_str, n):
    return "?x"

# parse sparql in dataset to action sequence
def processSparql(sparql_str, id="empty", constraint_list=[]):
        sparql_list = []
        untreated_list = sparql_str.split("\n")
        answer_keys = []
        index = -1
        sparql_str_type = "simple1"
        has_filter = False
        has_datetime = False

        for untreated_str in untreated_list:
            index += 1
            action_type = "A1"
            s = ""
            r = ""
            t = ""

            # remove note
            note_index = untreated_str.find("#")
            if note_index != -1:
                untreated_str = untreated_str[0:note_index]
            # remove /t
            untreated_str = untreated_str.replace("\t", "")
            untreated_str = untreated_str.strip()
            if untreated_str == '':
                continue

            if "UNION" == untreated_str:
                if b_print:
                    print("has union", id)
                sparql_str_type = "union"
            if "FILTER" in untreated_str:
                has_filter = True
            if "xsd:datetime" in untreated_str:
                has_datetime = True

            if "FILTER(NOT EXISTS" in untreated_str:
                FILTER_time_list = untreated_str.split(" ")
                filter_relation = ""
                for FILTER_time_item in FILTER_time_list:
                    if FILTER_time_item.startswith('ns:'):
                        filter_relation = FILTER_time_item.replace("ns:", "")
                        if index + 2 < len(untreated_list):
                            to_find_date_op_str = untreated_list[index + 2]
                            if "<=" in to_find_date_op_str:
                                action_type = "A10"
                                s = "?y"
                                r = filter_relation
                                tofind_date_list = to_find_date_op_str.split("\"")
                                if len(tofind_date_list) == 3:
                                    t = tofind_date_list[1]
                                    action_item = Action(action_type, s, r, t)
                                    if isValidAction(action_item):
                                        sparql_list.append(action_item)
                            elif ">=" in to_find_date_op_str:
                                action_type = "A11"
                                s = "?y"
                                r = filter_relation
                                tofind_date_list = to_find_date_op_str.split("\"")
                                if len(tofind_date_list) == 3:
                                    t = tofind_date_list[1]
                                    action_item = Action(action_type, s, r, t)
                                    if isValidAction(action_item):
                                        sparql_list.append(action_item)

            if untreated_str.startswith("SELECT"):  # find answer key
                for item in untreated_str.split(" "):
                    if "?" in item:
                        answer_keys.append(item.replace(" ", ""))
            elif untreated_str.startswith("PREFIX") or "langMatches" in untreated_str:
                # ignore
                pass
            elif untreated_str.startswith("FILTER (?x != ns:"):
                # filter not equal
                action_type = "A5"
                s = "?x"
                # s = "ANSWER"
                t = untreated_str.replace("FILTER (?x != ns:", "").replace(")", "").replace(" ", "")
                action_item = Action(action_type, s, r, t)
                if isValidAction(action_item):
                    sparql_list.append(action_item)
            elif untreated_str.count("?") == 2 and ("FILTER" not in untreated_str or "EXISTS" not in untreated_str):
                action_type = "A4"  # joint
                triple_list = untreated_str.split(" ")
                if len(triple_list) == 4:
                    s = triple_list[0].replace("ns:", "")
                    r = triple_list[1].replace("ns:", "")
                    t = triple_list[2].replace("ns:", "")
                    if s != "" and r != "" and t != "":
                        b_find_order_limit = False
                        to_find_order_index = -1
                        for to_find_order in untreated_list:
                            to_find_order_index += 1
                            if "ORDER BY" in to_find_order:
                                start_index = to_find_order.find("?")
                                if start_index != -1:
                                    end_index = to_find_order.find(")", start_index) if (action_type == "A8" or action_type == "A4")else len(
                                        to_find_order)
                                    if end_index != -1:
                                        var_name = to_find_order[start_index:end_index]
                                        if to_find_order_index < len(to_find_order):
                                            if untreated_list[to_find_order_index + 1].startswith("LIMIT "):
                                                limit_n = int(untreated_list[to_find_order_index + 1].replace("LIMIT ", ""))
                                                for to_find_var in untreated_list:
                                                    if var_name in to_find_var:
                                                        var_list = to_find_var.strip().split(" ")
                                                        relative_var = var_list[0]
                                                        relative_r = var_list[1].replace("ns:", "")
                                                        if var_name == t:
                                                            b_find_order_limit = True
                                                            action_type = "A8" if "ORDER BY DESC" in to_find_order else "A7"
                                                            action_item = Action(action_type, relative_var, relative_r, limit_n)
                                                            if isValidAction(action_item):
                                                                sparql_list.append(action_item)
                        if not b_find_order_limit:
                            action_item = Action(action_type, s, r, t)
                            if isValidAction(action_item):
                                sparql_list.append(action_item)
            elif untreated_str.count("?") == 1 and untreated_str.startswith("ns:"):
                # base action: select
                action_type = "A1"
                triple_list = untreated_str.split(" ")
                if len(triple_list) == 4:
                    s = triple_list[0].replace("ns:", "")
                    r = triple_list[1].replace("ns:", "")
                    t = triple_list[2].replace("ns:", "")
                    if s != "" and r != "" and t != "":
                        action_item = Action(action_type, s, r, t)
                        if isValidAction(action_item):
                            sparql_list.append(action_item)
            elif untreated_str.count("?") == 1 and untreated_str.startswith("?"):
                # ?x ns:a ns:b
                # if have e,  A3 : filter variable: find sub set fits the bill
                # if don't have e, A1_3 :find e
                action_type = "A3"
                triple_list = untreated_str.split(" ")
                if True:
                    s = triple_list[0].replace("ns:", "")
                    r = triple_list[1].replace("ns:", "")
                    t = triple_list[2].replace("ns:", "")
                    if s != "" and r != "" and t != "":
                        # for action in sparql_list:
                        #     if action.e == s or action.t == s:  # already has variable
                        #         action_type = "A4"
                        # special for webqsp: swap s and t ,A3->A1 "6" means single action_seq
                        # print(len(untreated_list), "length of untreated_list")
                        # if len(untreated_list) == 6:
                        #     action_type = "A1"
                        #     temp = s
                        #     s = t
                        #     t = temp
                        action_item = Action(action_type, s, r, t)
                        if isValidAction(action_item):
                            sparql_list.append(action_item)
                # print(action_item)


        # reorder list
        reorder_sparql_list = reorder(sparql_list, answer_keys)
        # reorder_sparql_list = sparql_list
        # for astr in reorder_sparql_list:
        #     print(astr.to_str())

        old_sqarql_list = []
        for item in reorder_sparql_list:
            seqset = {}
            seqlist = []
            # seqlist.append(item.e)
            # seqlist.append(item.r)
            # seqlist.append(item.t)
            if str(item.e) != '' and '?' not in str(item.e):
                seqlist.append(item.e)
            if str(item.r) != '' and '?' not in str(item.r):
                seqlist.append(item.r)
            if str(item.t) != '' and '?' not in str(item.t):
                seqlist.append(item.t)
            seqset[item.action_type] = seqlist
            old_sqarql_list.append(seqset)

        if has_filter and has_datetime:
            sparql_str_type = "filter_date"
        return old_sqarql_list, sparql_str_type

# parse sparql for value type answer
def processSparql_value(sparql_str, id="empty"):
        sparql_list = []
        untreated_list = sparql_str.split("\n")
        answer_keys = []
        value_len = len(untreated_list)
        if value_len == 8:
            value_str = untreated_list[5]
            value_str_list = value_str.split(" ")
            if len(value_str_list) == 4 and value_str_list[2] == '?x' and value_str_list[3] == '.':
                s = value_str_list[0]
                r = value_str_list[1]
                return [s, r]
        return []

def getsimple_seq(seq):
    return seq

def isValidAction(action_item):
    e = action_item.e
    t = action_item.t
    str_t = str(action_item.t)
    # return (e.startswith("m.") or e.startswith("?"))\
    #        and (str_t.startswith("m.") or str_t.startswith("?") or str_t == "" or isinstance(t, int))
    return True

def reorder(sparql_list, answer_keys):
    reorder_sparql_list = []
    for key in answer_keys:
        add_next_variable(sparql_list, key, reorder_sparql_list)
    reorder_sparql_list.reverse()
    return reorder_sparql_list

def add_next_variable(sparql_list, variable_key, reorder_sparql_list):
    for action_item in sparql_list:
        # filter_not_equal is always the last action
        if action_item.action_type == "A5":
            reorder_sparql_list.append(action_item)
            sparql_list.remove(action_item)
            break

    if variable_key == "":
        return

    variable_sql_list = []
    for sql in sparql_list:
        if variable_key == sql.e or variable_key == sql.t:
            variable_sql_list.append(sql)

    if len(variable_sql_list) == 0:
        return
    if len(variable_sql_list) == 1:
        cur_action = variable_sql_list[0]
        if cur_action == "?x" and "?" not in cur_action.t and cur_action.action_type == "A3":
            cur_action.action_type = "A1"
            e = cur_action.e
            t = cur_action.t
            cur_action.e = t
            cur_action.t = e
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)
            return

    next_variable = ""

    for sql in variable_sql_list:
        if sql.action_type == "A8" or sql.action_type == "A7":
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    for sql in variable_sql_list:
        if sql.action_type == "A9":
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    for sql in variable_sql_list:
        if sql.action_type == "A10" or sql.action_type == "A11":
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    for sql in variable_sql_list:
        if sql.action_type == "A3":
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    for sql in variable_sql_list:
        if sql.action_type == "A4":
            # next_variable
            next_variable = sql.e
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    for sql in variable_sql_list:
        if sql.action_type == "A2":
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    for sql in variable_sql_list:
        if sql.action_type == "A1":
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    return add_next_variable(sparql_list, next_variable, reorder_sparql_list)

w_1 = 0.2
def calc_01_reward(answer, true_answer):
    true_reward = 0.0
    try:
        if len(true_answer) == 0:
            if len(answer) == 0:
                return 1.0
            else:
                return w_1
        else:
            right_count = 0
            for e in answer:
                if e in true_answer:
                    right_count += 1
            return float(right_count) / float(len(true_answer))
    except:
        return true_reward

w_1 = 0.2
def calc_01_reward_type(res_answer, true_answer, type = "f1"):
    true_reward = 0.0
    prec = 0.0
    res_answer = set(res_answer)
    true_answer = set(true_answer)
    intersec = set(res_answer).intersection(set(true_answer))
    if len(true_answer) == 0:
        return 0.0
    if type == "jaccard":
        union = set([])
        union.update(res_answer)
        union.update(true_answer)
        true_reward = float(len(intersec)) / float(len(union))
    elif type == "recall":
        true_reward = float(len(intersec)) / float(len(true_answer))
    elif type == "f1":
        if len(res_answer) == 0:
            prec = 0.0
        else:
            prec = float(len(intersec)) / float(len(res_answer))
        rec = float(len(intersec)) / float(len(true_answer))
        if prec == 0 and rec == 0:
            true_reward = 0.0
        else:
            true_reward = (2.0 * prec * rec) / (prec + rec)
    elif type == "precision":
        if len(res_answer) == 0:
            prec = 0.0
        else:
            prec = float(len(intersec)) / float(len(res_answer))
            true_reward = prec
    return true_reward


def process_webqsp_RL():
    # Load WebQuestions Semantic Parses
    WebQSPList = []
    WebQSPList_Correct = []
    to_handle_list = []
    to_test_by_hand_list = []
    WebQSPList_Incorrect = []
    no_gold_answer = []
    AnswerType_Value_idlist = []
    to_add_list = []
    to_skip_list = []
    result_list = []
    no_x_list = []
    parse_error_list = []
    json_errorlist = []
    true_count = 0
    errorlist = []
    has_datetime_list = []
    has_union_list = []

    final_data_RL_train = {}
    final_data_RL_test = {}
    final_data_seq2seq_train = {}
    final_data_seq2seq_test = {}
    final_data_seq2seq = {}

    with open("WebQSPList_Correct.json", "r", encoding='UTF-8') as correct_list:
        WebQSPList_Correct = json.load(correct_list)
    with open("WebQSP.train.json", "r", encoding='UTF-8') as webQaTrain:
        load_dictTrain = json.load(webQaTrain)
    with open("WebQSP.test.json", "r", encoding='UTF-8') as webQaTest:
        load_dictTest = json.load(webQaTest)
    with open("to_handle.json", "r", encoding='UTF-8') as to_handle_file:
    # with open("human_mark.json", "r", encoding='UTF-8') as to_handle_file:
        to_handle_list = json.load(to_handle_file)
    with open("to_test_by_hand.json", "r", encoding='UTF-8') as to_test_by_hand_file:
        to_test_by_hand_list = json.load(to_test_by_hand_file)
    # with open("errorlist.json", "r", encoding='UTF-8') as errors_file:
    #     errorlist_file_list = json.load(errors_file)

    train_questions = load_dictTrain["Questions"]
    test_questions = load_dictTest["Questions"]
    all_questions = train_questions + test_questions
    small_questions = train_questions[0:50] + test_questions[0:50]

    process_questions = all_questions
    # total rewards
    total_reward = 0
    test_count = 0
    total_reward_jaccard = 0
    total_reward_precision = 0
    total_reward_recall = 0

    all_count = 0
    for parse_q in process_questions:
        question = parse_q["ProcessedQuestion"]
        for q in parse_q["Parses"]:
            id = q["ParseId"]
            if b_print:
                print(id)
            all_count += 1
            sparql = q["Sparql"]
            reward = 0.0
            answerList = q["Answers"]
            if len(answerList) == 0:
                print(id, "no gold answer")
                no_gold_answer.append(id)
            Answers = []
            orig_response_list = []
            for an in answerList:
                Answers.append(an['AnswerArgument'])
                orig_response_list.append(an['AnswerArgument'])
            answer_type = 'NONE'
            if len(answerList) > 0:
                answer_type = answerList[0]['AnswerType']
            response_entities = Answers
            orig_response_str = ', '.join(orig_response_list)

            constraint_list = q['Constraints']
            for constraint in constraint_list:
                Operator = constraint['Operator']
                ArgumentType = constraint['ArgumentType']
                Argument = constraint['Argument']
                EntityName = constraint['EntityName']
                SourceNodeIndex = constraint['SourceNodeIndex']
                NodePredicate = constraint['NodePredicate']
                ValueType = constraint['ValueType']
                if Operator == "LessOrEqual":
                    s = 1

                if Operator == "GreaterOrEqual":
                    s = 1

                # "Operator": "Equal",
                # "ArgumentType": "Entity",
                # "Argument": "m.04ztj",
                # "EntityName": "Marriage",
                # "SourceNodeIndex": 0,
                # "NodePredicate": "people.marriage.type_of_union",
                # "ValueType": "String"

                # "Operator": "LessOrEqual",
                # "ArgumentType": "Value",
                # "Argument": "2015-08-10",
                # "EntityName": "",
                # "SourceNodeIndex": 0,
                # "NodePredicate": "sports.sports_team_roster.from",
                # # "ValueType": "DateTime"

            if answer_type == "Value":
                AnswerType_Value_idlist.append(id)
                sr_list = processSparql_value(sparql)
                if len(sr_list) == 2:
                    for an in Answers:
                        to_add_list.append(str(sr_list[0] + '**' + sr_list[1] + '**' + an))

            # assert answer_type == "Entity"
            # continue
            # if id == "WebQTest-475.P0":  # test one
            if True:  # test all
                # test seq
                true_answer = Answers
                test_sparql = sparql
                sparql_str_type = ""
                # if id not in errorlist_file_list:
                #     continue
                if False:
                # if id in to_handle_list:
                    seq = to_handle_list[id]["action_sequence_list"]
                else:
                    seq, sparql_str_type = processSparql(test_sparql, id, constraint_list)
                    seq = getsimple_seq(seq)
                if sparql_str_type == "UNION":
                    has_union_list.append(id)
                elif sparql_str_type == "filter_date":
                    has_datetime_list.append(id)

                if b_print:
                    print(seq)
                symbolic_exe = Symbolics_WebQSP_novar(seq, tid=id)
                answer = symbolic_exe.executor()
                if b_print:
                    print("answer: ", answer)
                    print("true_answer: ", true_answer)
                try:
                    # key = "?x"
                    key = "ANSWER"
                    if key in answer or True:
                        res_answer = answer[key] if key in answer else []
                        reward = calc_01_reward_type(res_answer, true_answer, "f1")
                        if b_print:
                            print(id, reward)
                        if reward != 1.0:
                            result_list.append({id: [seq, reward]})
                            result_list.append({id + "_res_answer": list(res_answer)})
                            result_list.append({id + "_true_answer": list(true_answer)})
                            errorlist.append(id)

                        reward_jaccard = calc_01_reward_type(res_answer, true_answer, "jaccard")
                        reward_recall = calc_01_reward_type(res_answer, true_answer, "recall")
                        reward_precision = calc_01_reward_type(res_answer, true_answer, "precision")
                        test_count += 1
                        if reward <= 1.0:
                            # if get right answer, generate action sequence
                            true_count += 1
                            entity = set()
                            relation = set()
                            types = set()
                            e_index = 1
                            r_index = 1
                            t_index = 1
                            for srt in seq:
                                for k, v in srt.items():
                                    if k in ['A1']:
                                        if len(v) > 0:
                                            if v[0] != "" and v[0] not in entity:
                                                entity.add(v[0])
                                        if len(v) > 1:
                                            if v[1] != "" and v[1] not in relation:
                                                relation.add(v[1])
                                    elif k in ['A2', 'A3', 'A7', 'A8', 'A10', 'A11']:
                                        if len(v) > 0:
                                            if v[0] != "" and v[0] not in relation:
                                                relation.add(v[0])
                                        if len(v) > 1:
                                            if v[1] != "" and v[1] not in types:
                                                types.add(v[1])
                                    elif k in ['A4']:
                                        if len(v) > 0:
                                            if v[0] != "":
                                                relation.add(v[0])
                                    elif k in ['A5']:
                                        if len(v) > 0:
                                            if v[0] != "":
                                                types.add(v[0])

                                    # if v[2] != "" and v[2] not in entity:
                                    #     entity.add(v[2])
                                    # if len(v) > 2:
                                    #     if v[2] != "" and v[2] not in types:
                                    #         types.add(v[2])
                            entity = list(entity)
                            relation = list(relation)
                            types = list(types)
                            # types.sort(reverse=True)
                            # entity.sort(reverse=True)
                            entity_mask = dict()
                            relation_mask = dict()
                            type_mask = dict()
                            # try:
                            #     entity = sorted(entity)
                            # finally:
                            #     entity = entity
                            # relation = sorted(relation)
                            # type = sorted(type)
                            for e in entity:
                                dict_entity = {e: "ENTITY{0}".format(e_index)}
                                entity_mask.update(dict_entity)
                                e_index += 1
                            for r in relation:
                                dict_relation = {r: "RELATION{0}".format(r_index)}
                                relation_mask.update(dict_relation)
                                r_index += 1
                            for t in types:
                                dict_type = {t: "TYPE{0}".format(t_index)}
                                type_mask.update(dict_type)
                                t_index += 1
                            mask_action_sequence_list = []

                            for srt in seq:
                                mask_set = {}
                                masklist = []
                                a_mask = ""
                                e_mask = ""
                                r_mask = ""
                                t_mask = ""
                                e_mask_key = ""
                                r_mask_key = ""
                                t_mask_key = ""
                                for k, v in srt.items():
                                    a_mask = k
                                    if k in ['A1']:
                                        e_mask_key = v[0]
                                        r_mask_key = v[1]
                                    elif k in ['A2', 'A3', 'A7', 'A8', 'A10', 'A11']:
                                        r_mask_key = v[0]
                                        t_mask_key = v[1]
                                    elif k in ['A4']:
                                        r_mask_key = v[0]
                                    elif k in ['A5']:
                                        t_mask_key = v[0]
                                    #
                                    # if len(v) > 0:
                                    #     e_mask_key = v[0]
                                    # if len(v) > 1:
                                    #     r_mask_key = v[1]
                                    # t_mask_key = v[2]
                                    e_mask = entity_mask[e_mask_key] if e_mask_key != "" else ""
                                    r_mask = relation_mask[r_mask_key] if r_mask_key != "" else ""
                                    t_mask = type_mask[t_mask_key] if t_mask_key != "" and t_mask_key in type_mask else ""
                                if a_mask != "":
                                    if e_mask != "":
                                        masklist.append(e_mask)
                                    if r_mask != "":
                                        masklist.append(r_mask)
                                    if t_mask != "":
                                        masklist.append(t_mask)
                                    mask_set = {a_mask: masklist}
                                    mask_action_sequence_list.append(mask_set)

                            if id != "" and question != "" and seq != "":
                                question_string = '<E> '
                                if len(entity_mask) > 0:
                                    for entity_key, entity_value in entity_mask.items():
                                        if str(entity_value) != '':
                                            question_string += str(entity_value) + ' '
                                question_string += '</E> <R> '
                                if len(relation_mask) > 0:
                                    for relation_key, relation_value in relation_mask.items():
                                        if str(relation_value) != '':
                                            question_string += str(relation_value) + ' '
                                question_string += '</R> <T> '
                                if len(type_mask) > 0:
                                    for type_key, type_value in type_mask.items():
                                        if str(type_value) != '':
                                            question_string += str(type_value) + ' '
                                question_string += '</T> '

                                question_token = str(question).lower().replace('?', '')
                                question_token = question_token.replace(',', ' ')
                                question_token = question_token.replace(':', ' ')
                                question_token = question_token.replace('(', ' ')
                                question_token = question_token.replace(')', ' ')
                                question_token = question_token.replace('"', ' ')
                                question_token = question_token.strip()
                                question_string += question_token
                                question_string = question_string.strip()

                                action_string = ''
                                try:
                                    actions = eval(str(seq))
                                except SyntaxError:
                                    pass
                                if len(actions) > 0:
                                    action_list = actions
                                    action = eval(str(action_list))
                                    for action_dict in action_list:
                                        for temp_key, temp_value in action_dict.items():
                                            action_string += temp_key + ' ( '
                                            for token in temp_value:
                                                # if '-' in token:
                                                #     token = '- ' + token.replace('-','')
                                                if "?" not in token:
                                                    mask_token = ''
                                                    if token in entity_mask:
                                                        mask_token = entity_mask[token]
                                                    elif token in relation_mask:
                                                        mask_token = relation_mask[token]
                                                    elif token in type_mask:
                                                        mask_token = type_mask[token]
                                                    action_string += str(mask_token) + ' '
                                            action_string += ') '

                                int_mask = {}

                                correct_item = WebQSP(id, question, seq, entity, relation, types, entity_mask,
                                                      relation_mask, type_mask, mask_action_sequence_list,
                                                      answerList, question_string, response_entities, orig_response_str, action_string, int_mask)
                            # print(question)
                            # print(answer)
                            WebQSPList_Correct.append(id)
                            if id.startswith('WebQTrn'):
                                final_data_seq2seq_train.update(correct_item.obj_2_s2sjson())
                                final_data_RL_train.update(correct_item.obj_2_rljson())
                            else:
                                final_data_seq2seq_test.update(correct_item.obj_2_s2sjson())
                                final_data_RL_test.update(correct_item.obj_2_rljson())
                            final_data_seq2seq.update(correct_item.obj_2_s2sjson())
                        else:
                            if b_print:
                                print('incorrect!', reward)
                                print(" ")
                            WebQSPList_Incorrect.append(id)
                            errorlist.append(id)
                            json_errorlist.append(q)

                        total_reward += reward
                        total_reward_jaccard += reward_jaccard
                        total_reward_recall += reward_recall
                        total_reward_precision += reward_precision
                    else:
                        no_x_list.append({id:str(seq)})
                except Exception as exception:
                    print(exception)
                    parse_error_list.append(id)
                    pass

    questions_count = len(process_questions)
    print('all_count', all_count)
    print('questions_count', questions_count)

    mean_reward_jaccard = total_reward_jaccard / questions_count
    mean_reward_recall = total_reward_recall / questions_count
    mean_reward_precision = total_reward_precision / questions_count
    mean_reward = total_reward / questions_count
    print("mean_reward_jaccard: ", mean_reward_jaccard)
    print("mean_reward_recall: ", mean_reward_recall)
    print("mean_reward_precision: ", mean_reward_precision)
    print("mean_reward_f1: ", mean_reward)
    print("{0} pairs correct".format(true_count))
    print("errorlist", errorlist)
    print("has_union_list", has_union_list)
    print("has_datetime_list", has_datetime_list)
    print("parse_error_list", parse_error_list)


    # not x
    jsondata = json.dumps(no_x_list, indent=1)
    fileObject = open('no_x_list.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # parse_error_list
    jsondata = json.dumps(parse_error_list, indent=1)
    fileObject = open('parse_error_list.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # result_list
    jsondata = json.dumps(result_list, indent=1)
    fileObject = open('result_list.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # result_list
    jsondata = json.dumps(errorlist, indent=1)
    fileObject = open('errorlist.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # no_gold_answer
    jsondata = json.dumps(no_gold_answer, indent=1)
    fileObject = open('no_gold_answer.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # AnswerType_Value_idlist
    jsondata = json.dumps(AnswerType_Value_idlist, indent=1)
    fileObject = open('AnswerType_Value_idlist.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # to_add_list
    jsondata = json.dumps(to_add_list, indent=1)
    fileObject = open('to_add_list.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # jsondata = json.dumps(WebQSPList_Correct, indent=1, default=WebQSP.obj_2_json)
    jsondata = json.dumps(WebQSPList_Correct, indent=1)
    fileObject = open('WebQSPList_Correct1.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # final_data_RL
    jsondata = json.dumps(final_data_RL_train, indent=1)
    fileObject = open(mask_path_RL + 'final_webqsp_train_RL.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    jsondata = json.dumps(final_data_RL_test, indent=1)
    fileObject = open(mask_path_RL + 'final_webqsp_test_RL.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # final_data_seq2seq
    jsondata = json.dumps(final_data_seq2seq_train, indent=1)
    fileObject = open(mask_path_RL + 'final_data_seq2seq_train.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # final_data_seq2seq
    jsondata = json.dumps(final_data_seq2seq_test, indent=1)
    fileObject = open(mask_path_RL + 'final_data_seq2seq_test.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()

    # final_data_
    jsondata = json.dumps(final_data_seq2seq, indent=1)
    fileObject = open(mask_path_RL + 'final_data_seq2seq.json', 'w')
    fileObject.write(jsondata)
    fileObject.close()


# Get training data for sequence2sequence.
def getTrainingDatasetForPytorch_seq2seq_webqsp():
    fwTrainQ = open(mask_path + 'PT_train.question', 'w', encoding="UTF-8")
    fwTrainA = open(mask_path + 'PT_train.action', 'w', encoding="UTF-8")
    fwTestQ = open(mask_path + 'PT_test.question', 'w', encoding="UTF-8")
    fwTestA = open(mask_path + 'PT_test.action', 'w', encoding="UTF-8")
    fwQuestionDic = open(mask_path + 'dic_py.question', 'w', encoding="UTF-8")
    fwActionDic = open(mask_path + 'dic_py.action', 'w', encoding="UTF-8")
    questionSet = set()
    actionSet = set()
    with open("../../data/webqsp_data/test/final_data_seq2seq.json", 'r', encoding="UTF-8") as load_f:
        count = 1
        train_action_string_list, test_action_string_list, train_question_string_list, test_question_string_list = list(), list(), list(), list()
        dict_list = list()
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            try:
                actions = eval(str(value['mask_action_sequence_list']))
            except SyntaxError:
                pass
            if len(actions) > 0:
                count += 1
                action_string = ''
                action = actions
                action = eval(str(action))
                for dict in action:
                    for temp_key, temp_value in dict.items():
                        action_string += temp_key + ' ( '
                        for token in temp_value:
                            # if '-' in token:
                            #     token = '- ' + token.replace('-','')
                            if "?" not in token:
                                action_string += str(token) + ' '
                        action_string += ') '
                question_string = '<E> '


                # entities = value['entity_mask']
                # if len(entities) > 0:
                #     for entity_key, entity_value in entities.items():
                #         if str(entity_value) != '' and '?' not in str(entity_value):
                #             question_string += str(entity_value) + ' '
                # question_string += '</E> <R> '
                # relations = value['relation_mask']
                # if len(relations) > 0:
                #     for relation_key, relation_value in relations.items():
                #         if str(relation_value) != '' and '?' not in str(relation_value):
                #             question_string += str(relation_value) + ' '
                # question_string += '</R> <T> '
                # types = value['type_mask']
                # if len(types) > 0:
                #     for type_key, type_value in types.items():
                #         if str(type_value) != '' and '?' not in str(type_value):
                #             question_string += str(type_value) + ' '

                entities = value['entity_mask']
                if len(entities) > 0:
                    for entity_key, entity_value in entities.items():
                        if str(entity_value) != '' and '?' not in str(entity_key):
                            question_string += str(entity_value) + ' '
                question_string += '</E> <R> '
                relations = value['relation_mask']
                if len(relations) > 0:
                    for relation_key, relation_value in relations.items():
                        if str(relation_value) != '' and '?' not in str(relation_key):
                            question_string += str(relation_value) + ' '
                question_string += '</R> <T> '
                types = value['type_mask']
                if len(types) > 0:
                    for type_key, type_value in types.items():
                        if str(type_value) != '' and '?' not in str(type_key):
                            question_string += str(type_value) + ' '


                question_string += '</T> '
                question_token = str(value['question']).lower().replace('?', '')
                question_token = question_token.replace(',', ' ')
                question_token = question_token.replace(':', ' ')
                question_token = question_token.replace('(', ' ')
                question_token = question_token.replace(')', ' ')
                question_token = question_token.replace('"', ' ')
                question_token = question_token.strip()
                question_string += question_token
                question_string = question_string.strip() + '\n'

                action_string = action_string.strip() + '\n'
                action_tokens = action_string.strip().split(' ')
                action_tokens_set = set(action_tokens)
                actionSet = actionSet.union(action_tokens_set)

                question_tokens = question_string.strip().split(' ')
                question_tokens_set = set(question_tokens)
                questionSet = questionSet.union(question_tokens_set)

                dict_temp = {}
                dict_temp.setdefault('q', str(key) + ' ' + question_string)
                dict_temp.setdefault('a', str(key) + ' ' + action_string)
                dict_list.append(dict_temp)

    # train_size = int(len(dict_list) * 0.95)
    train_size = int(len(dict_list))
    for i, item in enumerate(dict_list):
        if item.get('a').startswith('WebQTrn'):
            train_action_string_list.append(item.get('a'))
            train_question_string_list.append(item.get('q'))
        else:
            test_action_string_list.append(item.get('a'))
            test_question_string_list.append(item.get('q'))
    fwTrainQ.writelines(train_question_string_list)
    fwTrainA.writelines(train_action_string_list)
    fwTestQ.writelines(test_question_string_list)
    fwTestA.writelines(test_action_string_list)
    fwTrainQ.close()
    fwTrainA.close()
    fwTestQ.close()
    fwTestA.close()

    questionList = list()
    for item in questionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            questionList.append(temp)
    actionList = list()
    for item in actionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            actionList.append(temp)
    fwQuestionDic.writelines(questionList)
    fwActionDic.writelines(actionList)
    print("Getting webqsp seq2seq process Dataset is done!")

# Get training data for sequence2sequence.
def getTrainingDatasetForPytorch_seq2seq_webqsp_novar():
    fwTrainQ = open(mask_path + 'PT_train.question', 'w', encoding="UTF-8")
    fwTrainA = open(mask_path + 'PT_train.action', 'w', encoding="UTF-8")
    fwTestQ = open(mask_path + 'PT_test.question', 'w', encoding="UTF-8")
    fwTestA = open(mask_path + 'PT_test.action', 'w', encoding="UTF-8")
    fwQuestionDic = open(mask_path + 'dic_py.question', 'w', encoding="UTF-8")
    fwActionDic = open(mask_path + 'dic_py.action', 'w', encoding="UTF-8")
    questionSet = set()
    actionSet = set()
    with open("../../data/webqsp_data/test/final_data_seq2seq.json", 'r', encoding="UTF-8") as load_f:
        count = 1
        train_action_string_list, test_action_string_list, train_question_string_list, test_question_string_list = list(), list(), list(), list()
        dict_list = list()
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            try:
                actions = eval(str(value['mask_action_sequence_list']))
            except SyntaxError:
                pass
            if len(actions) > 0:
                count += 1
                action_string = ''
                action = actions
                action = eval(str(action))
                for dict in action:
                    for temp_key, temp_value in dict.items():
                        action_string += temp_key + ' ( '
                        for token in temp_value:
                            # if '-' in token:
                            #     token = '- ' + token.replace('-','')
                            if "?" not in token:
                                action_string += str(token) + ' '
                        action_string += ') '
                question_string = '<E> '


                # entities = value['entity_mask']
                # if len(entities) > 0:
                #     for entity_key, entity_value in entities.items():
                #         if str(entity_value) != '' and '?' not in str(entity_value):
                #             question_string += str(entity_value) + ' '
                # question_string += '</E> <R> '
                # relations = value['relation_mask']
                # if len(relations) > 0:
                #     for relation_key, relation_value in relations.items():
                #         if str(relation_value) != '' and '?' not in str(relation_value):
                #             question_string += str(relation_value) + ' '
                # question_string += '</R> <T> '
                # types = value['type_mask']
                # if len(types) > 0:
                #     for type_key, type_value in types.items():
                #         if str(type_value) != '' and '?' not in str(type_value):
                #             question_string += str(type_value) + ' '

                entities = value['entity_mask']
                if len(entities) > 0:
                    for entity_key, entity_value in entities.items():
                        if str(entity_value) != '' and '?' not in str(entity_key):
                            question_string += str(entity_value) + ' '
                question_string += '</E> <R> '
                relations = value['relation_mask']
                if len(relations) > 0:
                    for relation_key, relation_value in relations.items():
                        if str(relation_value) != '' and '?' not in str(relation_key):
                            question_string += str(relation_value) + ' '
                question_string += '</R> <T> '
                types = value['type_mask']
                if len(types) > 0:
                    for type_key, type_value in types.items():
                        if str(type_value) != '' and '?' not in str(type_key):
                            question_string += str(type_value) + ' '


                question_string += '</T> '
                question_token = str(value['question']).lower().replace('?', '')
                question_token = question_token.replace(',', ' ')
                question_token = question_token.replace(':', ' ')
                question_token = question_token.replace('(', ' ')
                question_token = question_token.replace(')', ' ')
                question_token = question_token.replace('"', ' ')
                question_token = question_token.strip()
                question_string += question_token
                question_string = question_string.strip() + '\n'

                action_string = action_string.strip() + '\n'
                action_tokens = action_string.strip().split(' ')
                action_tokens_set = set(action_tokens)
                actionSet = actionSet.union(action_tokens_set)

                question_tokens = question_string.strip().split(' ')
                question_tokens_set = set(question_tokens)
                questionSet = questionSet.union(question_tokens_set)

                dict_temp = {}
                dict_temp.setdefault('q', str(key) + ' ' + question_string)
                dict_temp.setdefault('a', str(key) + ' ' + action_string)
                dict_list.append(dict_temp)

    # train_size = int(len(dict_list) * 0.95)
    train_size = int(len(dict_list))
    for i, item in enumerate(dict_list):
        if item.get('a').startswith('WebQTrn'):
            train_action_string_list.append(item.get('a'))
            train_question_string_list.append(item.get('q'))
        else:
            test_action_string_list.append(item.get('a'))
            test_question_string_list.append(item.get('q'))
    fwTrainQ.writelines(train_question_string_list)
    fwTrainA.writelines(train_action_string_list)
    fwTestQ.writelines(test_question_string_list)
    fwTestA.writelines(test_action_string_list)
    fwTrainQ.close()
    fwTrainA.close()
    fwTestQ.close()
    fwTestA.close()

    questionList = list()
    for item in questionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            questionList.append(temp)
    actionList = list()
    for item in actionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            actionList.append(temp)
    fwQuestionDic.writelines(questionList)
    fwActionDic.writelines(actionList)
    print("Getting webqsp seq2seq process Dataset is done!")

def getShareVocabularyForWebQSP():
    questionVocab = set()
    actionVocab = set()
    actionVocab_list = list()
    with open(mask_path + 'dic_py.question', 'r', encoding="UTF-8") as infile1, \
            open(mask_path + 'dic_rl.question', 'r', encoding="UTF-8") as infile2, \
            open(mask_path + 'dic_rl_tr.question', 'r', encoding="UTF-8") as infile3:
        count = 0
        while True:
            lines_gen = list(islice(infile1, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                questionVocab.add(token)
                count = count + 1
            print(count)
        count = 0
        while True:
            lines_gen = list(islice(infile2, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                questionVocab.add(token)
                count = count + 1
            print(count)
        count = 0
        while True:
            lines_gen = list(islice(infile3, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                questionVocab.add(token)
                count = count + 1
            print(count)
    with open(mask_path + 'dic_py.action', 'r', encoding="UTF-8") as infile1, open(mask_path + 'dic_rl.action', 'r', encoding="UTF-8") as infile2:
        count = 0
        while True:
            lines_gen = list(islice(infile1, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                actionVocab.add(token)
                count = count + 1
            print(count)
        count = 0
        while True:
            lines_gen = list(islice(infile2, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                token = line.strip()
                actionVocab.add(token)
                count = count + 1
            print(count)
    action_size = 0
    for word in actionVocab:
        if word not in questionVocab and word not in special_characters:
            actionVocab_list.append(word)
            action_size += 1
        elif word in special_characters:
            actionVocab_list.append(word)
            action_size += 1
            if word in questionVocab:
                questionVocab.remove(word)
    questionVocab_list = list(questionVocab)
    share_vocab_list = actionVocab_list + questionVocab_list
    for i in range(len(share_vocab_list)):
        share_vocab_list[i] = share_vocab_list[i] + '\n'
    fw = open(mask_path_RL + 'share.webqsp.question', 'w', encoding="UTF-8")
    fw.writelines(share_vocab_list)
    fw.close()
    print("Writing SHARE VOCAB is done!")
    return action_size

# Get training data for REINFORCE (using annotation instead of denotation as reward).
def getTrainingDatasetForRlWebQSP():
    fwTrainQ = open(mask_path + 'RL_train.question', 'w', encoding="UTF-8")
    fwTrainA = open(mask_path + 'RL_train.action', 'w', encoding="UTF-8")
    fwTestQ = open(mask_path + 'RL_test.question', 'w', encoding="UTF-8")
    fwTestA = open(mask_path + 'RL_test.action', 'w', encoding="UTF-8")
    fwQuestionDic = open(mask_path + 'dic_rl.question', 'w', encoding="UTF-8")
    fwActionDic = open(mask_path + 'dic_rl.action', 'w', encoding="UTF-8")
    fwNoaction = open(mask_path + 'no_action_question.txt', 'w', encoding="UTF-8")
    no_action_question_list = list()
    questionSet = set()
    actionSet = set()
    with open("../../data/webqsp_data/test/final_data_seq2seq.json", 'r', encoding="UTF-8") as load_f:
        count = 1
        train_action_string_list, test_action_string_list, train_question_string_list, test_question_string_list = list(), list(), list(), list()
        dict_list = list()
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            try:
                actions = eval(str(value['mask_action_sequence_list']))
            except SyntaxError:
                pass
            if len(actions) > 0:
                action_string_list = list()
                count += 1
                for action in actions:
                    action_string = ''
                    for temp_key, temp_value in action.items():
                        action_string += temp_key + ' ( '
                        for token in temp_value:
                            if '-' in token:
                                token = '- ' + token.replace('-','')
                            action_string += str(token) + ' '
                        action_string += ') '
                    action_string = action_string.strip() + '\n'
                    action_string_list.append(action_string)
                question_string = '<E> '
                entities = value['entity_mask']
                if len(entities) > 0:
                    for entity_key, entity_value in entities.items():
                        if str(entity_value) != '':
                            question_string += str(entity_value) + ' '
                question_string += '</E> <R> '
                relations = value['relation_mask']
                if len(relations) > 0:
                    for relation_key, relation_value in relations.items():
                        if str(relation_value) !='':
                            question_string += str(relation_value) + ' '
                question_string += '</R> <T> '
                types = value['type_mask']
                if len(types) > 0:
                    for type_key, type_value in types.items():
                        if str(type_value) !='':
                            question_string += str(type_value) + ' '
                question_string += '</T> '
                question_token = str(value['question']).lower().replace('?', '')
                question_token = question_token.replace(',', ' ')
                question_token = question_token.replace(':', ' ')
                question_token = question_token.replace('(', ' ')
                question_token = question_token.replace(')', ' ')
                question_token = question_token.replace('"', ' ')
                question_token = question_token.strip()
                question_string += question_token
                question_string = question_string.strip() + '\n'

                question_tokens = question_string.strip().split(' ')
                question_tokens_set = set(question_tokens)
                questionSet = questionSet.union(question_tokens_set)

                action_withkey_list = list()
                if len(action_string_list) > 0:
                    for action_string in action_string_list:
                        action_tokens = action_string.strip().split(' ')
                        action_tokens_set = set(action_tokens)
                        actionSet = actionSet.union(action_tokens_set)
                        action_withkey_list.append(str(key) + ' ' + action_string)

                dict_temp = {}
                dict_temp.setdefault('q', str(key) + ' ' + question_string)
                dict_temp.setdefault('a', action_withkey_list)
                dict_list.append(dict_temp)
            elif len(actions) == 0:
                no_action_question_list.append(str(key) + ' ' + str(value['question']).lower().replace('?', '').strip() + '\n')

    train_size = int(len(dict_list))
    # train_size = int(len(dict_list) * 0.95)
    for i, item in enumerate(dict_list):
        if item.get('a')[0].startswith('WebQTrn'):
            for action_string in item.get('a'):
                train_action_string_list.append(action_string)
            train_question_string_list.append(item.get('q'))
        else:
            for action_string in item.get('a'):
                test_action_string_list.append(action_string)
            test_question_string_list.append(item.get('q'))
    fwTrainQ.writelines(train_question_string_list)
    fwTrainA.writelines(train_action_string_list)
    fwTestQ.writelines(test_question_string_list)
    fwTestA.writelines(test_action_string_list)
    questionList = list()
    for item in questionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            questionList.append(temp)
    actionList = list()
    for item in actionSet:
        temp = str(item) + '\n'
        if temp != '\n':
            actionList.append(temp)
    fwQuestionDic.writelines(questionList)
    fwActionDic.writelines(actionList)
    fwNoaction.writelines(no_action_question_list)
    fwTrainQ.close()
    fwTrainA.close()
    fwTestQ.close()
    fwTestA.close()
    fwQuestionDic.close()
    fwActionDic.close()
    fwNoaction.close()
    print ("Getting RL processDataset is done!")

def testInputData():
    # # List of (question, {question information and answer}) pairs, the training pairs are in format of 1:1.
    MAX_TOKENS = 40
    DIC_PATH = mask_path_RL + 'share.webqsp.question'
    TRAIN_QUESTION_ANSWER_PATH = mask_path_RL + 'final_webqsp_train_RL.json'
    phrase_pairs, emb_dict = data.load_RL_data_TR(TRAIN_QUESTION_ANSWER_PATH, DIC_PATH, MAX_TOKENS)
    print("Obtained {0} phrase pairs with {1} uniq words from {2}.".format(len(phrase_pairs), len(emb_dict), TRAIN_QUESTION_ANSWER_PATH))

if __name__ == "__main__":
    print("start process webqsp dataset")
    process_webqsp_RL()   # dataset to mask


    # getTrainingDatasetForPytorch_seq2seq_webqsp_novar() # PT.train
    #
    # getTrainingDatasetForRlWebQSP()
    # getShareVocabularyForWebQSP() # share.question
    # testInputData()

    # getTrainingDatasetForPytorch_seq2seq_webqsp() # PT.train