import collections
import os
import sys
import logging
import itertools
import pickle
import json
import torch

from . import cornell

UNKNOWN_TOKEN = '#UNK'
BEGIN_TOKEN = "#BEG"
END_TOKEN = "#END"
MAX_TOKENS = 30
MIN_TOKEN_FEQ = 1
SHUFFLE_SEED = 1987
LINE_SIZE = 50000

EMB_DICT_NAME = "emb_dict.dat"
EMB_NAME = "emb.npy"

log = logging.getLogger("data")

from itertools import islice


def save_emb_dict(dir_name, emb_dict):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "wb") as fd:
        pickle.dump(emb_dict, fd)


def load_emb_dict(dir_name):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "rb") as fd:
        return pickle.load(fd)


def encode_words(words, emb_dict):
    """
    Convert list of words into list of embeddings indices, adding our tokens
    :param words: list of strings
    :param emb_dict: embeddings dictionary
    :return: list of IDs
    """
    res = [emb_dict[BEGIN_TOKEN]]
    unk_idx = emb_dict[UNKNOWN_TOKEN]
    for w in words:
        idx = emb_dict.get(w.lower(), unk_idx)
        res.append(idx)
    res.append(emb_dict[END_TOKEN])
    return res


def encode_words_for_retriever(words, emb_dict):
    """
    Convert list of words into list of embeddings indices, adding our tokens
    :param words: list of strings
    :param emb_dict: embeddings dictionary
    :return: list of IDs
    """
    res = []
    unk_idx = emb_dict[UNKNOWN_TOKEN]
    for w in words:
        idx = emb_dict.get(w.lower(), unk_idx)
        res.append(idx)
    return res


def encode_phrase_pairs(phrase_pairs, emb_dict, filter_unknows=True):
    """
    Convert list of phrase pairs to training data
    :param phrase_pairs: list of (phrase, phrase)
    :param emb_dict: embeddings dictionary (word -> id)
    :return: list of tuples ([input_id_seq], [output_id_seq])
    """
    unk_token = emb_dict[UNKNOWN_TOKEN]
    result = []
    for p1, p2 in phrase_pairs:
        p = encode_words(p1, emb_dict), encode_words(p2, emb_dict)
        '''It is not correct to exclude the sample with 'UNK' from the dataset.'''
        # if unk_token in p[0] or unk_token in p[1]:
        #     continue
        result.append(p)
    return result


def encode_phrase_pairs_RLTR(phrase_pairs, emb_dict, filter_unknows=True):
    """
    Convert list of phrase pairs to training data
    :param phrase_pairs: list of (phrase, phrase)
    :param emb_dict: embeddings dictionary (word -> id)
    :return: list of tuples ([input_id_seq], [output_id_seq])
    """
    unk_token = emb_dict[UNKNOWN_TOKEN]
    result = []
    for p1, p2 in phrase_pairs:
        p = encode_words(p1, emb_dict), p2
        # STAR: It is incorrect to exclude the sample with 'UNK' from the dataset.
        # if unk_token in p[0] or unk_token in p[1]:
        #     continue
        result.append(p)
    return result


# Change token list into token tuple.
def group_train_data_RLTR(training_data):
    groups = []
    for p1, p2 in training_data:
        l = tuple(p1)
        temp = (l, p2)
        groups.append(temp)
    return list(groups)


# Change token list into token tuple.
def group_train_data_RLTR_for_support(training_data):
    groups = {}
    for p1, p2 in training_data:
        groups.setdefault(p2['qid'], (tuple(p1), p2))
    return groups


def group_train_data(training_data):
    """
    Group training pairs by first phrase
    :param training_data: list of (seq1, seq2) pairs
    :return: list of (seq1, [seq*]) pairs

    这里的defaultdict(function_factory)构建的是一个类似dictionary的对象，
    其中keys的值，自行确定赋值，但是values的类型，是function_factory的类实例，而且具有默认值。
    比如defaultdict(int)则创建一个类似dictionary对象，里面任何的values都是int的实例，
    而且就算是一个不存在的key, d[key] 也有一个默认值，这个默认值是int()的默认值0.
    一般用法为：
    d = collections.defaultdict(list)
    for k, v in s:
        d[k].append(v)
    """
    groups = collections.defaultdict(list)
    for p1, p2 in training_data:
        # 取出key为tuple(p1)的value；
        # If no value is assigned to the key, the default value (in this case empty list) is assigned to the key.
        l = groups[tuple(p1)]
        # 将p2挂在value后面，完成grouping操作；
        l.append(p2)
    return list(groups.items())


def group_train_data_one_to_one(training_data):
    """
    Group training_data to one-to-one format.
    """
    temp_list = list()
    for p1, p2 in training_data:
        p2_list = list()
        p1 = tuple(p1)
        p2_list.append(p2)
        temp = (p1, p2_list)
        temp_list.append(temp)
    return temp_list


def iterate_batches(data, batch_size):
    assert isinstance(data, list)
    assert isinstance(batch_size, int)

    ofs = 0
    while True:
        batch = data[ofs * batch_size:(ofs + 1) * batch_size]
        # STAR: Why the length of a batch can not be one?
        # if len(batch) <= 1:
        if len(batch) < 1:
            break
        yield batch
        ofs += 1


def get_RL_question_action_list(qpath, apath):
    qdict = {}
    adict = {}
    with open(qpath, 'r', encoding="UTF-8") as infile:
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                qid = str(line).strip().split(' ')[0]
                tokens = str(line).strip().split(' ')[1:]
                qdict.setdefault(qid, tokens)

    with open(apath, 'r', encoding="UTF-8") as infile:
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                qid = str(line).strip().split(' ')[0]
                tokens = str(line).strip().split(' ')[1:]
                if qid not in adict:
                    action_list = list()
                    action_list.append(tokens)
                    adict.setdefault(qid, action_list)
                else:
                    action_list = adict.get(qid)
                    action_list.append(tokens)
                    adict.setdefault(qid, action_list)
    return qdict, adict


def get_question_token_list(path):
    with open(path, 'r', encoding="UTF-8") as infile:
        token_list = list()
        count = 0
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                tokens = str(line).strip().split(' ')[1:]
                token_list.append(tokens)
                count = count + 1
                # print(count)
    return token_list


def get_action_token_list(path):
    with open(path, 'r', encoding="UTF-8") as infile:
        token_list = list()
        count = 0
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                tokens = str(line).strip().split(' ')[1:]
                token_list.append(tokens)
                count = count + 1
                # print(count)
    return token_list


def get_vocab(path):
    with open(path, 'r', encoding="UTF-8") as infile:
        token_list = list()
        count = 0
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                tokens = str(line).strip().lower()
                token_list.append(tokens)
                count = count + 1
                # print(count)
    return token_list


# Suppose the genre_filter is comedy and other parameters are as default.
def load_data(genre_filter, max_tokens=MAX_TOKENS, min_token_freq=MIN_TOKEN_FEQ):
    dialogues = cornell.load_dialogues(genre_filter=genre_filter)
    if not dialogues:
        log.error("No dialogues found, exit!")
        sys.exit()
    log.info("Loaded %d dialogues with %d phrases, generating training pairs",
             len(dialogues), sum(map(len, dialogues)))
    # 将dataset中的对话前一句和后一句组合成一个training pair，作为输入和输出;
    phrase_pairs = dialogues_to_pairs(dialogues, max_tokens=max_tokens)
    log.info("Counting freq of words...")
    # get all words;
    word_counts = collections.Counter()
    for dial in dialogues:
        for p in dial:
            word_counts.update(p)
    freq_set = set(map(lambda p: p[0], filter(lambda p: p[1] >= min_token_freq, word_counts.items())))
    log.info("Data has %d uniq words, %d of them occur more than %d",
             len(word_counts), len(freq_set), min_token_freq)
    phrase_dict = phrase_pairs_dict(phrase_pairs, freq_set)
    return phrase_pairs, phrase_dict


def load_data_from_existing_data(QUESTION_PATH, ACTION_PATH, DIC_PATH, max_tokens=None):
    """
    Convert dialogues to training pairs of phrases
    :param dialogues:
    :param max_tokens: limit of tokens in both question and reply
    :return: list of (phrase, phrase) pairs
    """
    question_list = get_question_token_list(QUESTION_PATH)
    action_list = get_action_token_list(ACTION_PATH)
    vocab_list = get_vocab(DIC_PATH)

    result = []
    if len(question_list) == len(action_list):
        for i in range(len(question_list)):
            if max_tokens is None or (len(question_list[i]) <= max_tokens and len(action_list[i]) <= max_tokens):
                result.append((question_list[i], action_list[i]))

    res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
    next_id = 3
    for w in vocab_list:
        if w not in res:
            res[w] = next_id
            next_id += 1
    return result, res


def load_RL_data(QUESTION_PATH, ACTION_PATH, DIC_PATH, max_tokens=None):
    qdict, adict = get_RL_question_action_list(QUESTION_PATH, ACTION_PATH)
    vocab_list = get_vocab(DIC_PATH)
    result = []
    if len(qdict) == len(adict):
        for qid, q in qdict.items():
            if max_tokens is None or len(q) <= max_tokens:
                if qid in adict:
                    if len(adict.get(qid)) > 0:
                        for action in adict.get(qid):
                            if max_tokens is None or len(action) <= max_tokens:
                                result.append((q, action))

    res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
    next_id = 3
    for w in vocab_list:
        if w not in res:
            res[w] = next_id
            next_id += 1
    return result, res


def load_RL_data_TR(QUESTION_PATH, DIC_PATH=None, max_tokens=None, NSM=False):
    result = []
    with open(QUESTION_PATH, 'r', encoding="UTF-8") as load_f:
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            length = len(str(value['input']).strip().split(' '))
            if 'entity_mask' in value and 'relation_mask' in value and 'type_mask' in value and 'response_bools' in value and 'response_entities' in value and 'orig_response' in value and 'question' in value and length <= max_tokens:
                if not NSM:
                    question_info = {'qid': key, 'entity_mask': value['entity_mask'],
                                     'relation_mask': value['relation_mask'],
                                     'type_mask': value['type_mask'], 'response_bools': value['response_bools'],
                                     'response_entities': value['response_entities'],
                                     'orig_response': value['orig_response']}
                else:
                    question_info = {'qid': key, 'entity_mask': value['entity_mask'],
                                     'relation_mask': value['relation_mask'],
                                     'type_mask': value['type_mask'], 'response_bools': value['response_bools'],
                                     'response_entities': value['response_entities'],
                                     'orig_response': value['orig_response'],
                                     'pseudo_gold_program': value['pseudo_gold_program']}
                result.append((str(value['input']).strip().split(' '), question_info))
            else:
                continue
    if (DIC_PATH != None):
        res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
        vocab_list = get_vocab(DIC_PATH)
        next_id = 3
        for w in vocab_list:
            if w not in res:
                res[w] = next_id
                next_id += 1
        return result, res
    else:
        return result


def load_RL_data_TR_INT(QUESTION_PATH, DIC_PATH=None, max_tokens=None, NSM=False):
    result = []
    with open(QUESTION_PATH, 'r', encoding="UTF-8") as load_f:
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            length = len(str(value['input']).strip().split(' '))
            if 'entity_mask' in value and 'relation_mask' in value and 'type_mask' in value and 'response_bools' in value and 'response_entities' in value and 'orig_response' in value and 'question' in value and 'int_mask' in value and length <= max_tokens:
                if not NSM:
                    question_info = {'qid': key, 'entity_mask': value['entity_mask'],
                                     'relation_mask': value['relation_mask'],
                                     'type_mask': value['type_mask'], 'response_bools': value['response_bools'],
                                     'response_entities': value['response_entities'],
                                     'orig_response': value['orig_response'], 'int_mask': value['int_mask']}
                else:
                    question_info = {'qid': key, 'entity_mask': value['entity_mask'],
                                     'relation_mask': value['relation_mask'],
                                     'type_mask': value['type_mask'], 'response_bools': value['response_bools'],
                                     'response_entities': value['response_entities'],
                                     'orig_response': value['orig_response'], 'int_mask': value['int_mask'],
                                     'pseudo_gold_program': value['pseudo_gold_program']}

                result.append((str(value['input']).strip().split(' '), question_info))
            else:
                continue
    if (DIC_PATH != None):
        res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
        vocab_list = get_vocab(DIC_PATH)
        next_id = 3
        for w in vocab_list:
            if w not in res:
                res[w] = next_id
                next_id += 1
        return result, res
    else:
        return result


def load_data_MAML(QUESTION_PATH, DIC_PATH=None, max_tokens=None):
    result = []
    with open(QUESTION_PATH, 'r', encoding="UTF-8") as load_f:
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            length = len(str(value['input']).strip().split(' '))
            if max_tokens is None or length <= max_tokens:
                if 'entity_mask' in value and 'relation_mask' in value and 'type_mask' in value and 'response_bools' in value and 'response_entities' in value and 'orig_response' in value and 'question' in value:
                    question_info = {'qid': key, 'entity_mask': value['entity_mask'],
                                     'relation_mask': value['relation_mask'],
                                     'type_mask': value['type_mask'], 'response_bools': value['response_bools'],
                                     'response_entities': value['response_entities'],
                                     'orig_response': value['orig_response'],
                                     'entity': value['entity'], 'relation': value['relation'], 'type': value['type'],
                                     'question': value['question']}
                    result.append((str(value['input']).strip().split(' '), question_info))
            else:
                continue
    if (DIC_PATH != None):
        res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
        vocab_list = get_vocab(DIC_PATH)
        next_id = 3
        for w in vocab_list:
            if w not in res:
                res[w] = next_id
                next_id += 1
        return result, res
    else:
        return result


# TODO: unify load_data_MAML_TEST with load_data_MAML with using 'response_bools';
def load_data_MAML_TEST(QUESTION_PATH, DIC_PATH=None, max_tokens=None):
    result = []
    with open(QUESTION_PATH, 'r', encoding="UTF-8") as load_f:
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            length = len(str(value['input']).strip().split(' '))
            if max_tokens is None or length <= max_tokens:
                if 'entity_mask' in value and 'relation_mask' in value and 'type_mask' in value and 'response_entities' in value and 'orig_response' in value and 'question' in value:
                    question_info = {'qid': key, 'entity_mask': value['entity_mask'],
                                     'relation_mask': value['relation_mask'],
                                     'type_mask': value['type_mask'], 'response_entities': value['response_entities'],
                                     'orig_response': value['orig_response'], 'entity': value['entity'],
                                     'relation': value['relation'], 'type': value['type'],
                                     'question': value['question']}
                    result.append((str(value['input']).strip().split(' '), question_info))
            else:
                continue
    if (DIC_PATH != None):
        res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
        vocab_list = get_vocab(DIC_PATH)
        next_id = 3
        for w in vocab_list:
            if w not in res:
                res[w] = next_id
                next_id += 1
        return result, res
    else:
        return result


def load_dict(DIC_PATH=None):
    if (DIC_PATH != None):
        res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
        vocab_list = get_vocab(DIC_PATH)
        next_id = 3
        for w in vocab_list:
            if w not in res:
                res[w] = next_id
                next_id += 1
        return res
    else:
        return {}


def phrase_pairs_dict(phrase_pairs, freq_set):
    """
    Return the dict of words in the dialogues mapped to their IDs
    :param phrase_pairs: list of (phrase, phrase) pairs
    :return: dict
    """
    res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
    next_id = 3
    for p1, p2 in phrase_pairs:
        for w in map(str.lower, itertools.chain(p1, p2)):
            if w not in res and w in freq_set:
                res[w] = next_id
                next_id += 1
    return res


# 将dataset中的对话前一句和后一句组合成一个training pair;
def dialogues_to_pairs(dialogues, max_tokens=None):
    """
    Convert dialogues to training pairs of phrases
    :param dialogues:
    :param max_tokens: limit of tokens in both question and reply
    :return: list of (phrase, phrase) pairs
    """
    result = []
    for dial in dialogues:
        prev_phrase = None
        for phrase in dial:
            if prev_phrase is not None:
                if max_tokens is None or (len(prev_phrase) <= max_tokens and len(phrase) <= max_tokens):
                    result.append((prev_phrase, phrase))
            prev_phrase = phrase
    return result


def decode_words(indices, rev_emb_dict):
    return [rev_emb_dict.get(idx, UNKNOWN_TOKEN) for idx in indices]


def trim_tokens_seq(tokens, end_token):
    res = []
    for t in tokens:
        res.append(t)
        if t == end_token:
            break
    return res


def split_train_test(data, train_ratio=0.90):
    count = int(len(data) * train_ratio)
    return data[:count], data[count:]


def get944k(path):
    with open(path, "r", encoding='UTF-8') as CSQA_List:
        dict944k = json.load(CSQA_List)
    return dict944k


def get_webqsp(path):
    with open(path, "r", encoding='UTF-8') as WEBQSP_List:
        dict_webqsp = json.load(WEBQSP_List)
    return dict_webqsp


def get_docID_indices(order_list):
    did_indices = {}
    d_list = []
    next_id = 0
    for w in order_list:
        if not len(w) < 1:
            docID, document = list(w.items())[0]
            if docID not in did_indices:
                did_indices[docID] = next_id
                d_list.append(document)
                next_id += 1
    return did_indices, d_list


def get_ordered_docID_document(filepath):
    with open(filepath, 'r', encoding="UTF-8") as load_f:
        return (json.load(load_f))


def load_json(QUESTION_PATH):
    with open(QUESTION_PATH, 'r', encoding="UTF-8") as load_f:
        load_dict = json.load(load_f)
        return load_dict


def get_qid_question_pairs(filepath):
    pair = {}
    pair_list = get_ordered_docID_document(filepath)
    for temp in pair_list:
        docID, document = list(temp.items())[0]
        pair[docID] = document
    return pair


def get_question_embedding(question, emb_dict, net):
    question_token = question.lower().replace('?', '')
    question_token = question_token.replace(',', ' ')
    question_token = question_token.replace(':', ' ')
    question_token = question_token.replace('(', ' ')
    question_token = question_token.replace(')', ' ')
    question_token = question_token.replace('"', ' ')
    question_token = question_token.strip().split()
    question_token_indices = [emb_dict['#UNK'] if token not in emb_dict else emb_dict[token] for token in
                              question_token]
    question_token_embeddings = net.emb(torch.tensor(question_token_indices, requires_grad=False).cuda())
    question_embeddings = torch.mean(question_token_embeddings, 0).view(1, -1)
    question_embeddings = torch.tensor(question_embeddings.tolist(), requires_grad=False).cuda()
    return question_embeddings
