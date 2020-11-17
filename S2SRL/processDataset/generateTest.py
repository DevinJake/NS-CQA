# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 23:44
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io
import json
import re
import fnmatch
import os
from Preprocess.load_qadata import *
from itertools import islice
LINE_SIZE = 100000

def getQuestionAndContextints():
    contextIntPath = '../../data/demoqa2/context_ints_testfull.txt'
    lines = list()
    with open(contextIntPath, 'r', encoding="UTF-8") as infile:
        count = 0
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                lines.append(line.strip())
            count = count + 1
            print(count)
    count = 0
    q_ints = dict()
    while count < len(lines):
        line = str(lines[count])
        if line.startswith('context_utterance'):
            q_utterance = ' '.join(line.split(' ')[1:]).strip()
            q_ints[q_utterance] = ''
            count += 1
            if count < len(lines) and str(lines[count]).startswith('context_ints'):
                int_tokens = ' '.join(str(lines[count]).split(' ')[1:]).strip()
                if int_tokens != '' and int_tokens.strip().isdigit():
                    q_ints[q_utterance] = str(int(int_tokens))
            count += 1
    return q_ints

# Add context-int information with the test dataset.
def testGenerate():
    q_ints = getQuestionAndContextints()
    with open("../../data/auto_QA_data/CSQA_ANNOTATIONS_test_INT.json", 'w', encoding="UTF-8") as test_json, open("../../data/auto_QA_data/CSQA_ANNOTATIONS_test.json", 'r', encoding="UTF-8") as load_f:
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            q_string = value['question'].strip()
            if q_string in q_ints:
                value['context_ints'] = q_ints[q_string]
            else:
                value['context_ints'] = ''
                print('Question not in q_ints is: %s' % (key + ': ' + q_string))
            int_maskID = {}
            if value['context_ints'] != '':
                int_maskID[value['context_ints']] = 'INT'
            value['int_mask'] = int_maskID
        test_json.writelines(json.dumps(load_dict, indent=1, ensure_ascii=False, sort_keys=False))
        print("Writing JSON is done!")


if __name__ == "__main__":
    testGenerate()
