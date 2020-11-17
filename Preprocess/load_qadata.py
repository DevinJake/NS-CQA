# -*- coding: utf-8 -*-
# @Time    : 2019/1/20 14:50

import re

import fnmatch
import os


def load_qadata(qa_dir):
    """
    :param qa_dir: 预处理后的qa数据的文件夹，eg：/home/zhangjingyao/preprocessed_data_10k/test
    :return: 这个文件夹下面，问答数据的字典。最外层是序号: QA_1_
    """
    print("begin_load_qadata")
    qa_set = {}
    # os.walk: generates the file names in a directory tree by walking the tree.
    # default: top, which is used to yield 3-tuples,
    # i.e., (dirpath, dirnames, filenames) for each directory rooted at directory
    for root, dirnames, filenames in os.walk(qa_dir):
        if(dirnames == []):
            qa_id = root[root.rfind("_")+1:]
            qa_dict ={}
            for filename in fnmatch.filter(filenames, '*.txt'):
                pattern = re.compile('QA_\d+_')
                # re.sub: substitute the pattern with "" in filename.
                keystr = re.sub(pattern,"", filename).replace(".txt","")
                qa_dict[keystr] = open(root+"/"+filename).readlines()
            qa_set[qa_id] = qa_dict
    print("load_qadata_success")
    return qa_set

# Get the question-answer pairs grouped by the category of the questions.
def getQA_by_state_py3(qa_set):
    total = 0
    qmap = {}
    for key, qafile in qa_set.items():
        if not "context" in qafile: continue
        for qid in range(len(qafile["context"])):
            # Get all contents of one specific question-answer pair.
            q = {k: v[qid] for k, v in qafile.items()}
            # State is used to indicate the category of the question.
            if not "state" in q: continue
            total+=1
            # Analyzing the question-answer pairs.
            # qstring = q["context_utterance"]
            qstate = q["state"]
            # qrelations = q["context_relations"].split('|')
            if(qstate in qmap):
                qmap[qstate].append(q)
            else:
                qmap[qstate] = [q]
    print(total)
    return qmap

# Get the question-answer pairs grouped by the category of the questions.
def getQA_by_state(qa_set):
    total = 0
    qmap = {}
    # dict. itervalues(): Returns an iterator over the dictionary’s values.
    # The usage in python2.
    for qafile in qa_set.itervalues():
        if not "context" in qafile: continue
        for qid in range(len(qafile["context"])):
            # Get all contents of one specific question-answer pair.
            q = {k: v[qid] for k, v in qafile.items()}
            # State is used to indicate the category of the question.
            if not "state" in q: continue
            total+=1
            # Analyzing the question-answer pairs.
            # qstring = q["context_utterance"]
            qstate = q["state"]
            # qrelations = q["context_relations"].split('|')
            if(qmap.has_key(qstate)):
                qmap[qstate].append(q)
            else:
                qmap[qstate] = [q]
    print(total)
    return qmap

def count_qa_relation(qa):
    return len(qa['context_relations'].split('|'))

def count_qa_entity(qa):
    return len(qa['context_entities'].split('|'))

def count_qa_etype(qa):
    return len(qa['context_types'].split('|'))

def return_qa_type(qa):
    if 'or' in qa['context_utterance']:
        return 'union'
    elif 'and' in qa['context_utterance']:
        return 'inter'
    elif "not" in qa['context_utterance']:
        return 'diff'
    else:
        return 'None'

def return_N_type(qa):
    if 'atleast' in qa['context_utterance']:
        return 'at_least'
    elif 'atmost' in qa['context_utterance']:
        return 'at_most'
    elif 'approxi' in qa['context_utterance'] or 'around' in qa['context_utterance']:
        return 'approxi'
    elif 'exactly' in qa['context_utterance']:
        return 'equal'

    elif 'more' in qa['context_utterance'] or 'greater' in qa['context_utterance']:
        return 'more'
    elif 'less' in qa['context_utterance'] or 'lesser' in qa['context_utterance']:
        return 'less'

    elif 'max' in qa['context_utterance']:
        return 'max'
    elif 'min' in qa['context_utterance']:
        return 'min'

    else:
        return 'None'

countDict = {}
for root, dirnames, filenames in os.walk('/home/zhangjingyao/demo/'):
    # print filenames
    for f in filenames:
        countDict[f] = 0

def print_context_int_py3(qa,root,fname):
    fname1 = root + fname + '.txt'
    with open(fname1, 'a+') as f:
        f.write('context_utterance ' + str(qa['context_utterance']))
        f.write('context_ints ' + str(qa['context_ints']))
    # fname1 = root + fname + '_onlyints.txt'
    # with open(fname1, 'a+') as f:
    #     s = str(qa['context_ints']).strip()
    #     if s != '':
    #         f.write(s+'\n')

def print_qa_py3(qa,root,fname):
    fname = root+fname+'.txt'
    #if(os.path.exists(fname) and os.path.getsize(fname)>102400): return

    with open(fname,'a+') as f:
        # Move file pointer to the start of the file.
        f.seek(0)
        flag = 0
        lines = f.readlines()
        for line in lines:
            if line.startswith("state"):
                flag += 1
            # if flag == 20:
            #     return
        f.write(str(flag)+"\n")
        f.write('state: ' + str(qa['state']))
        f.write('context_utterance: ' + str(qa['context_utterance']))
        f.write('context_relations: ' + str(qa['context_relations']))
        f.write('context_entities: ' + str(qa['context_entities']))
        f.write('context_types: ' + str(qa['context_types']))
        f.write('context: ' + str(qa['context']))
        f.write('orig_response: ' + str(qa['orig_response']))
        f.write('response_entities: ' + str(qa['response_entities']))
        f.write('----------------------------\n')
        f.write('SYMBOLIC:\n')
        f.write('CODE:\n')
        f.write('----------------------------\n')

def print_qa(qa,root,fname):
    countDict[fname+'.txt']+=1
    fname = root+fname+'.txt'
    #if(os.path.exists(fname) and os.path.getsize(fname)>102400): return

    f = open(fname,"a+")
    flag = 0
    for line in f.readlines():
        if line.startswith("state"):
            flag += 1
        # if flag == 20:
        #     return
    print (f, flag)
    print (f, 'state:',qa['state'],)
    print (f, "context_utterance:",qa['context_utterance'],)
    print (f, 'context_relations:',qa['context_relations'],)
    print (f, 'context_entities:',qa['context_entities'],)
    print (f, 'context_types:',qa['context_types'],)
    print (f, 'context:',qa['context'],)
    print (f, 'orig_response:',qa['orig_response'],)
    print (f, 'response_entities:',qa['response_entities'],)
    print (f, "----------------------------")
    print (f, "SYMBOLIC:\n")
    print (f, "----------------------------")
    print (f, "CODE:\n")
    print (f, "----------------------------")

def get_trainingset_for_annotation():
    root = "../data/demoqa2/"
    qa_set = load_qadata("../data/official_downloaded_data/10k/train_10k")
    qa_map = getQA_by_state_py3(qa_set)
    # for qa in qa_map['Quantitative Reasoning (Count) (All)\n']:
    #     print_qa_py3(qa, root, 'train_count_all')
    # for qa in qa_map['Quantitative Reasoning (All)\n']:
    #     print_qa_py3(qa, root, 'train_quanti_all')
    for qa in qa_map['Verification (Boolean) (All)\n']:
        print_qa_py3(qa, root, 'train_bool_all')

# Get question and context_ints info for training dataset.
def get_context_ints_trainingset(root, path):
    qa_set = load_qadata(path)
    qa_map = getQA_by_state_py3(qa_set)
    type = 'context_ints_' + path.split('_')[-1]
    count = 0
    for k, v in qa_map.items():
        for qa in v:
            print_context_int_py3(qa, root, type)
            count += 1
            if count % 10000 == 0:
                print(count)


if __name__ == "__main__":
    get_trainingset_for_annotation()
    # root = "../data/demoqa2/"
    # path = "../data/official_downloaded_data/10k/train_10k"
    # get_context_ints_trainingset(root, path)
    #
    # path = "../data/official_downloaded_data/944k/train_944k"
    # get_context_ints_trainingset(root, path)

    # path = "../data/official_downloaded_data/156k_testfull"
    # get_context_ints_trainingset(root, path)

    '''
    qa_set = load_qadata("/data/zjy/valid_full")
    qa_map = getQA_by_state(qa_set)


    # root = '/home/zhangjingyao/demo/'
    # for fname in os.listdir(root):
    #     f = open(root+fname,"w+")
    #     f.truncate()

    logicalcount = 0
    for qa in qa_map['Logical Reasoning (All)\n']:
        logicalcount+=1
        # if count_qa_relation(qa) == 1 and return_qa_type(qa) == 'union':
        #     print_qa(qa, root, 'logical_union_single')
        # elif count_qa_relation(qa) == 1 and return_qa_type(qa) == 'inter':
        #     print_qa(qa, root, 'logical_inter_single')
        # elif count_qa_relation(qa) == 1 and return_qa_type(qa) == 'diff':
        #     print_qa(qa, root, 'logical_diff_single')
        # elif count_qa_relation(qa) > 1:
        #     print_qa(qa, root, 'logical_above_multi')
        # else:
        #     print_qa(qa, root, 'logical_else')

    vericount = 0
    for qa in qa_map['Verification (Boolean) (All)\n']:
        vericount+=1
        # print_qa(qa, root, 'verifi_bool_single')

    qcount_count = 0
    for qa in qa_map['Quantitative Reasoning (Count) (All)\n']:
        qcount_count+=1
        # if return_qa_type(qa) == 'diff': print(return_qa_type(qa)),count_qa_entity(qa) ,qa['context']
        # if count_qa_etype(qa) == 2  and return_N_type(qa) in ['at_least','at_most','approxi','equal']:
        #     print_qa(qa, root, 'quant_countOver_single_et')
        # elif count_qa_etype(qa) > 2  and return_N_type(qa) in ['at_least','at_most','approxi','equal']:
        #     print_qa(qa, root, 'quant_countOver_multi_et')
        # elif count_qa_etype(qa) == 1 and count_qa_entity(qa) == 1:
        #     print_qa(qa, root, 'quant_count_single_et')
        # elif count_qa_etype(qa) > 1 and count_qa_entity(qa) == 1:
        #     print_qa(qa, root, 'quant_count_multi_et')
        # elif count_qa_entity(qa) > 1 and return_qa_type(qa) in ['union', 'inter']:
        #     print_qa(qa, root, 'quant_count_logical')
        # elif count_qa_entity(qa) > 1 and return_qa_type(qa) in ['diff']:
        #     print_qa(qa, root, 'quant_count_logical_diff')
        # else:
        #     print_qa(qa, root, 'count_else')

    q_count = 0
    for qa in qa_map['Quantitative Reasoning (All)\n']:
        q_count+=1
        # print_qa(qa, root, 'verifi_bool_single&multi')
        # if count_qa_etype(qa) == 2 and return_N_type(qa) in ['max','min']:
        #     print_qa(qa, root, 'quant_minmax_single_et')
        # elif count_qa_etype(qa) > 2  and return_N_type(qa) in ['max','min']:
        #     print_qa(qa, root, 'quant_minmax_multi_et')
        # elif count_qa_etype(qa) == 2 and return_N_type(qa) in ['at_least','at_most','approxi','equal']:
        #     print_qa(qa, root, 'quant_atleast_single_et')
        # elif count_qa_etype(qa) > 2 and return_N_type(qa) in ['at_least','at_most','approxi','equal']:
        #     print_qa(qa, root, 'quant_atleast_multi_et')
        # else:
        #     print_qa(qa, root, 'quanti_else')
    ccount_count = 0
    for qa in qa_map['Comparative Reasoning (Count) (All)\n']:
        ccount_count+=1
        # if count_qa_etype(qa) == 2 and return_N_type(qa) in ['more', 'less']:
        #     print_qa(qa, root, 'comp_countOver_more_single_et')
        # elif count_qa_etype(qa) > 2 and return_N_type(qa) in ['more', 'less']:
        #     print_qa(qa, root, 'comp_countOver_more_multi_et')
        # else:
        #     print_qa(qa, root, 'Comparative_Count_else')
    c_count = 0
    for qa in qa_map['Comparative Reasoning (All)\n']:
        c_count +=1
        # if count_qa_etype(qa) == 2 and return_N_type(qa) in ['more', 'less']:
        #     print_qa(qa, root, 'comp__more_single_et')
        # elif count_qa_etype(qa) > 2 and return_N_type(qa) in ['more', 'less']:
        #     print_qa(qa, root, 'comp__more_multi_et')
        # else:
        #     print_qa(qa, root, 'Comparative_else')

    simplecount = 0
    for qa in qa_map['Simple Question (Direct)\n']:
        simplecount+=1

    print("Simple::Comparative::ComparativeCount::Quantitative::QuantitativeCount::Verification::Logical")
    print(simplecount,c_count,ccount_count,q_count,qcount_count,vericount,logicalcount)
    print("total_val:",simplecount+c_count+ccount_count+q_count+qcount_count+vericount+logicalcount)
    total = 0
    for k,v in countDict.items():
        # print k,v
        total+=v
    # print "total", total
    '''
