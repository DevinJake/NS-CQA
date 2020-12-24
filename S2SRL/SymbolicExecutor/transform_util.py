# -*- coding: utf-8 -*-
# @Time    : 2019/9/1 23:36
# @Author  : Devin Hua
# Function: transforming.

# Transform boolean results into string format.
def transformBooleanToString(list):
    temp_set = set()
    if len(list) == 0:
        return ''
    else:
        for i, item in enumerate(list):
            if item == True:
                list[i] = "YES"
                temp_set.add(list[i])
            elif item == False:
                list[i] = "NO"
                temp_set.add(list[i])
            else:
                return ''
    if len(temp_set) == 1:
        return temp_set.pop()
    if len(temp_set) > 1:
        return ((' and '.join(list)).strip() + ' respectively')

# Transform action sequence ['A2', '(', 'Q5058355', 'P361', 'Q5058355', ')', 'A2', '(', 'Q5058355', 'P361', 'Q5058355', ')', 'A15', '(', 'Q22329858', ')'] into list.
def list2dict(list):
    final_list = []
    temp_list = []
    new_list = []
    action_list = []
    left_count, right_count, action_count = 0, 0, 0
    for a in list:
        if a.startswith("A"):
            action_count+=1
            action_list.append(a)
        if (a == "("):
            new_list = []
            left_count+=1
            continue
        if (a == ")"):
            right_count+=1
            if ("-" in new_list and new_list[-1] != "-"):
                new_list[new_list.index("-") + 1] = "-" + new_list[new_list.index("-") + 1]
                new_list.remove("-")
            if (new_list == []):
                new_list = ["", "", ""]
            if (len(new_list) == 1):
                new_list = [new_list[0], "", ""]
            if ("&" in new_list):
                new_list = ["&", "", ""]
            if ("-" in new_list):
                new_list = ["-", "", ""]
            if ("|" in new_list):
                new_list = ["|", "", ""]
            temp_list.append(new_list)
            # To handle the error when action sequence is like 'A1 (Q1,P1,Q2) A2 Q3,P2,Q4)'.
            new_list = []
            continue
        if not a.startswith("A"):
            if a.startswith("E"):  a = "Q17"
            if a.startswith("T"):  a = "Q17"
            new_list.append(a)

    # To handle the error when action sequence is like 'A1 Q1,P1,Q2) A2(Q3,P2,Q4', 'A1(Q1,P1,Q2 A2(Q3,P2,Q4)'.
    number_list = [left_count, right_count, len(action_list), len(temp_list)]
    set_temp = set(number_list)
    # The value of multiple numbers is same.
    if len(set_temp) == 1:
        for action, parameter_temp in zip(action_list, temp_list):
            final_list.append({action: parameter_temp})
    # print("final_list", final_list)
    return final_list

def list2dict_webqsp(list):
    #print("list", list)
    final_list = []
    temp_list = []
    new_list = []
    for a in list:
        if (a == "("):
            new_list = []
            continue
        if (a == ")"):
            if ("-" in new_list):
                new_list[new_list.index("-") + 1] = "-" + new_list[new_list.index("-") + 1]
                new_list.remove("-")
            if (new_list == []):
                new_list = ["", "", ""]
            if (len(new_list) == 1):
                new_list = [new_list[0], "", ""]
            if ("&" in new_list):
                new_list = ["&", "", ""]
            if ("-" in new_list):
                new_list = ["-", "", ""]
            if ("|" in new_list):
                new_list = ["|", "", ""]
            temp_list.append(new_list)
            continue
        if not a.startswith("A"):
            # if a.startswith("E"):  a = "Q17"
            # if a.startswith("T"):  a = "Q17"
            new_list.append(a)

    i = 0
    for a in list:
        if (a.startswith("A")):
            if i < len(temp_list):
                no_empty_list = []
                for item in temp_list[i]:
                    if item != '':
                        no_empty_list.append(item)
                final_list.append({a: no_empty_list})
                # temp_dict[a] = temp_list[i]
                i += 1

    return final_list