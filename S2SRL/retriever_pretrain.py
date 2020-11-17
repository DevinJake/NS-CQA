import torch.nn as nn
import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from datetime import datetime
from statistics import mean
from libbots import adabound, data, model, metalearner, retriever_module

MAX_TOKENS = 40
MAX_MAP = 1000000
DIC_PATH = '../data/auto_QA_data/share.question'
DIC_PATH_INT = '../data/auto_QA_data/share_INT.question'
SAVES_DIR = '../data/saves/new_retriever'
QID_RANGE = '../data/auto_QA_data/944k_rangeDict.json'
ORDERED_QID_QUESTION_DICT = '../data/auto_QA_data/CSQA_result_question_type_count944k_orderlist.json'
TRAIN_QUESTION_ANSWER_PATH = '../data/auto_QA_data/mask_even_1.0%/RL_train_TR_new_2k.question'
TRAIN_944K_QUESTION_ANSWER_PATH = '../data/auto_QA_data/CSQA_DENOTATIONS_full_944K.json'
DICT_944K = '../data/auto_QA_data/CSQA_result_question_type_944K.json'
DICT_944K_WEAK = '../data/auto_QA_data/CSQA_result_question_type_count944K.json'
POSITIVE_Q_DOCS = '../data/auto_QA_data/retriever_question_documents_pair.json'
QTYPE_DOC_RANGE = '../data/auto_QA_data/944k_rangeDict.json'
TRAINING_SAMPLE_DICT = '../data/auto_QA_data/retriever_training_samples.json'

def get_document_embedding(doc_list, emb_dict, net):
    d_embed_list = []
    for i, doc in enumerate(doc_list):
        # Get question tokens:
        question_token = doc.lower().replace('?', '')
        question_token = question_token.replace(',', ' ')
        question_token = question_token.replace(':', ' ')
        question_token = question_token.replace('(', ' ')
        question_token = question_token.replace(')', ' ')
        question_token = question_token.replace('"', ' ')
        question_token = question_token.strip().split()
        question_token_indices = [emb_dict['#UNK'] if token not in emb_dict else emb_dict[token] for token in question_token]
        question_token_embeddings = net.emb(torch.tensor(question_token_indices, requires_grad=False).cuda())
        # The average of the token embeddings is used to represent the embedding of the query.
        question_embeddings = torch.mean(question_token_embeddings, 0).tolist()
        d_embed_list.append(question_embeddings)
        if i%10000 ==0:
            print('Transformed %d*10k embeddings!' %(i/10000))
    return d_embed_list

def initialize_document_embedding(int_flag=True, w2v=300, file_path=''):
    device = 'cuda'
    # Dict: word token -> ID.
    if not int_flag:
        emb_dict = data.load_dict(DIC_PATH=DIC_PATH)
    else:
        emb_dict = data.load_dict(DIC_PATH=DIC_PATH_INT)
    ordered_docID_doc_list = data.get_ordered_docID_document(ORDERED_QID_QUESTION_DICT)
    docID_dict, doc_list = data.get_docID_indices(ordered_docID_doc_list)
    # Index -> qid.
    rev_docID_dict = {id: doc for doc, id in docID_dict.items()}

    net = retriever_module.RetrieverModel(emb_size=w2v, dict_size=len(docID_dict), EMBED_FLAG=False,
                         device='cuda').to('cuda')
    net.cuda()
    net.zero_grad()
    # temp_param_dict = get_net_parameter(net)

    # Get trained wording embeddings.
    path = file_path
    net1 = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE,
                             LSTM_FLAG=True, ATT_FLAG=False, EMBED_FLAG=False).to(device)
    net1.load_state_dict(torch.load(path))
    doc_embedding_list = get_document_embedding(doc_list, emb_dict, net1)
    # Add padding vector.
    doc_embedding_list.append([0.0] * model.EMBEDDING_DIM)
    doc_embedding_tensor = torch.tensor(doc_embedding_list).cuda()
    net.document_emb.weight.data = doc_embedding_tensor.clone().detach()
    # temp_param_dict1 = get_net_parameter(net)

    MAP_for_queries = 1.0
    epoch = 0
    isExists = os.path.exists(SAVES_DIR)
    if not isExists:
        os.makedirs(SAVES_DIR)

    # os.makedirs(SAVES_DIR, exist_ok=True)
    # torch.save(net.state_dict(), os.path.join(SAVES_DIR, "initial_epoch_%03d_%.3f.dat" % (epoch, MAP_for_queries)))
    torch.save(net.state_dict(), os.path.join(SAVES_DIR, "initial_epoch_%03d_%.3f.dat" % (epoch, MAP_for_queries)))

def establish_positive_question_documents_pair(MAX_TOKENS, int_flag = True):
    # Dict: word token -> ID.
    docID_dict, _ = data.get_docID_indices(data.get_ordered_docID_document(ORDERED_QID_QUESTION_DICT))
    # Index -> qid.
    rev_docID_dict = {id: doc for doc, id in docID_dict.items()}
    # # List of (question, {question information and answer}) pairs, the training pairs are in format of 1:1.
    if not int_flag:
        phrase_pairs, emb_dict = data.load_data_MAML(TRAIN_QUESTION_ANSWER_PATH, DIC_PATH, MAX_TOKENS)
    else:
        phrase_pairs, emb_dict = data.load_data_MAML(TRAIN_QUESTION_ANSWER_PATH, DIC_PATH_INT, MAX_TOKENS)
    print("Obtained %d phrase pairs with %d uniq words from %s." %(len(phrase_pairs), len(emb_dict),
             TRAIN_QUESTION_ANSWER_PATH))
    phrase_pairs_944K = data.load_data_MAML(TRAIN_944K_QUESTION_ANSWER_PATH, max_tokens=MAX_TOKENS)
    print("Obtained %d phrase pairs from %s." %(len(phrase_pairs_944K), TRAIN_944K_QUESTION_ANSWER_PATH))

    # Transform token into index in dictionary.
    train_data = data.encode_phrase_pairs_RLTR(phrase_pairs, emb_dict)
    # # list of (seq1, [seq*]) pairs，把训练对做成1：N的形式；
    # train_data = data.group_train_data(train_data)
    train_data = data.group_train_data_RLTR(train_data)
    train_data_944K = data.encode_phrase_pairs_RLTR(phrase_pairs_944K, emb_dict)
    train_data_944K = data.group_train_data_RLTR_for_support(train_data_944K)

    dict944k = data.get944k(DICT_944K)
    print("Reading dict944k from %s is done. %d pairs in dict944k." %(DICT_944K, len(dict944k)))
    dict944k_weak = data.get944k(DICT_944K_WEAK)
    print("Reading dict944k_weak from %s is done. %d pairs in dict944k_weak" %(DICT_944K_WEAK, len(dict944k_weak)))

    metaLearner = metalearner.MetaLearner(samples=5,train_data_support_944K=train_data_944K, dict=dict944k,
                                          dict_weak=dict944k_weak, steps=5, weak_flag=True)

    question_doctments_pair_list = {}
    idx = 0
    for temp_batch in data.iterate_batches(train_data, 1):
        task = temp_batch[0]
        if len(task) == 2 and 'qid' in task[1]:
            # print("Task %s is training..." %(str(task[1]['qid'])))
            # Establish support set.
            support_set = metaLearner.establish_support_set(task, metaLearner.steps, metaLearner.weak_flag, metaLearner.train_data_support_944K)
            documents = []
            if len(support_set) > 0:
                for support_sample in support_set:
                    if len(support_sample) == 2 and 'qid' in support_sample[1]:
                        documents.append(support_sample[1]['qid'])
            else:
                print('task %s has no support set!' %(str(task[1]['qid'])))
                documents.append(task[1]['qid'])
            question_doctments_pair_list[task[1]['qid']] = documents
            if idx % 100 == 0:
                print(idx)
            idx += 1
        else:
            print('task has no qid or len(task)!=2:')
            print(task)
    fw = open('../data/auto_QA_data/retriever_question_documents_pair.json', 'w', encoding="UTF-8")
    fw.writelines(json.dumps(question_doctments_pair_list, indent=1, ensure_ascii=False))
    fw.close()
    print('Writing retriever_question_documents_pair.json is done!')

def AnalyzeQuestion(question_info):
    typelist = ['Simple Question (Direct)_',
                'Verification (Boolean) (All)_',
                'Quantitative Reasoning (Count) (All)_',
                'Logical Reasoning (All)_',
                'Comparative Reasoning (Count) (All)_',
                'Quantitative Reasoning (All)_',
                'Comparative Reasoning (All)_'
                ]
    typelist_for_test = ['SimpleQuestion(Direct)',
                              'Verification(Boolean)(All)',
                              'QuantitativeReasoning(Count)(All)',
                              'LogicalReasoning(All)',
                              'ComparativeReasoning(Count)(All)',
                              'QuantitativeReasoning(All)',
                              'ComparativeReasoning(All)'
                              ]
    type_map = {'SimpleQuestion(Direct)': 'Simple Question (Direct)_',
           'Verification(Boolean)(All)': 'Verification (Boolean) (All)_',
           'QuantitativeReasoning(Count)(All)': 'Quantitative Reasoning (Count) (All)_',
           'LogicalReasoning(All)': 'Logical Reasoning (All)_',
           'ComparativeReasoning(Count)(All)': 'Comparative Reasoning (Count) (All)_',
           'QuantitativeReasoning(All)': 'Quantitative Reasoning (All)_',
           'ComparativeReasoning(All)': 'Comparative Reasoning (All)_'}
    type_name = 'NOTYPE'
    for typei in typelist:
        if typei in question_info['qid']:
            type_name = typei
            break
    if type_name == 'NOTYPE':
        for typei in typelist_for_test:
            if typei in question_info['qid']:
                type_name = type_map[typei]
                break
    entity_count = len(question_info['entity']) if 'entity' in question_info else 0
    relation_count = len(question_info['relation']) if 'relation' in question_info else 0
    type_count = len(question_info['type']) if 'type' in question_info else 0
    question = question_info['question'] if 'question' in question_info else 'NOQUESTION'
    key_weak = '{0}{1}_{2}_{3}'.format(type_name, entity_count, relation_count, type_count)
    return key_weak, question, question_info['qid']

def generate_training_samples(int_flag=True):
    training_sample_dict = {}
    docID_dict, _ = data.get_docID_indices(data.get_ordered_docID_document(ORDERED_QID_QUESTION_DICT))
    positive_q_docs_pair = data.load_json(POSITIVE_Q_DOCS)
    qtype_docs_range = data.load_json(QTYPE_DOC_RANGE)
    if not int_flag:
        phrase_pairs, _ = data.load_data_MAML(TRAIN_QUESTION_ANSWER_PATH, DIC_PATH, MAX_TOKENS)
    else:
        phrase_pairs, _ = data.load_data_MAML(TRAIN_QUESTION_ANSWER_PATH, DIC_PATH_INT, MAX_TOKENS)
    print("Obtained %d phrase pairs from %s." % (len(phrase_pairs), TRAIN_QUESTION_ANSWER_PATH))
    for question in phrase_pairs:
        if len(question) == 2 and 'qid' in question[1]:
            key_weak, _, query_qid = AnalyzeQuestion(question[1])
            query_index = docID_dict[query_qid]
            if key_weak in qtype_docs_range:
                document_range = (qtype_docs_range[key_weak]['start'], qtype_docs_range[key_weak]['end'])
            else:
                document_range = (0, len(docID_dict))
            positive_document_list = [docID_dict[doc] for doc in positive_q_docs_pair[query_qid]]
            training_sample_dict[query_qid] = {'query_index': query_index, 'document_range': document_range, 'positive_document_list': positive_document_list}
    fw = open('../data/auto_QA_data/retriever_training_samples.json', 'w', encoding="UTF-8")
    fw.writelines(json.dumps(training_sample_dict, indent=1, ensure_ascii=False))
    fw.close()
    print('Writing retriever_training_samples.json is done!')

def retriever_training(epoches, RETRIEVER_EMBED_FLAG=True, query_embedding=True, w2v=300, seq2seq_word_embedding_path='', int_flag=True):
    ''' One instance of the retriever training samples:
        query_index = [800000, 0, 2, 100000, 400000, 600000]
        document_range = [(700000, 944000),
                          (1, 10),
                          (10, 300000),
                          (10, 300000),
                          (300000, 500000),
                          (500000, 700000)]
        positive_document_list =
        [[700001-700000, 700002-700000, 900000-700000, 910000-700000, 944000-2-700000],
        [2, 3],
        [13009-10, 34555-10, 234-10, 6789-10, 300000-1-10],
        [11-10, 16-10, 111111-10, 222222-10, 222223-10],
        [320000-300000, 330000-300000, 340000-300000, 350000-300000, 360000-300000],
        [600007-500000, 610007-500000, 620007-500000, 630007-500000, 690007-500000]]'''

    retriever_path = '../data/saves/retriever/initial_epoch_000_1.000.dat'
    device = 'cuda'
    learning_rate = 0.01
    # TODO: get dataset with INT mask;
    # TODO: get similar docs with INT mask;
    docID_dict, _ = data.get_docID_indices(data.get_ordered_docID_document(ORDERED_QID_QUESTION_DICT))
    # Index -> qid.
    rev_docID_dict = {id: doc for doc, id in docID_dict.items()}
    training_samples = data.load_json(TRAINING_SAMPLE_DICT)

    net = retriever_module.RetrieverModel(emb_size=w2v, dict_size=len(docID_dict), EMBED_FLAG=RETRIEVER_EMBED_FLAG, device=device).to(device)
    net.load_state_dict(torch.load(retriever_path))
    net.zero_grad()
    # temp_param_dict = get_net_parameter(net)
    # retriever_optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)
    # retriever_optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, eps=1e-3)
    retriever_optimizer = adabound.AdaBound(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, final_lr=0.1)
    # temp_param_dict = get_net_parameter(net)

    emb_dict = None
    net1 = None
    qid_question_pair = {}
    if query_embedding:
        if not int_flag:
            emb_dict = data.load_dict(DIC_PATH=DIC_PATH)
        else:
            emb_dict = data.load_dict(DIC_PATH=DIC_PATH_INT)
        # Get trained wording embeddings.
        path = seq2seq_word_embedding_path
        net1 = model.PhraseModel(emb_size=w2v, dict_size=len(emb_dict),
                                 hid_size=model.HIDDEN_STATE_SIZE,
                                 LSTM_FLAG=True, ATT_FLAG=False, EMBED_FLAG=False).to(device)
        net1.load_state_dict(torch.load(path))
        qid_question_pair = data.get_qid_question_pairs(ORDERED_QID_QUESTION_DICT)

    max_value = MAX_MAP
    MAP_for_queries = MAX_MAP

    for i in range(epoches):
        print('Epoch %d is training......' %(i))
        # count= 0
        for key, value in training_samples.items():
            retriever_optimizer.zero_grad()
            net.zero_grad()
            if query_embedding:
                if key in qid_question_pair:
                    question_tokens = qid_question_pair[key]
                else:
                    print("ERROR! NO SUCH QUESTION: %s!" %(str(key)))
                    continue
                query_tensor = data.get_question_embedding(question_tokens, emb_dict, net1)
            else:
                query_tensor = torch.tensor(net.pack_input(value['query_index']).tolist(), requires_grad=False).cuda()
            document_range = (value['document_range'][0], value['document_range'][1])
            logsoftmax_output, _, _ = net(query_tensor, document_range)
            logsoftmax_output = logsoftmax_output.cuda()
            positive_document_list = [k-value['document_range'][0] for k in value['positive_document_list']]
            possitive_logsoftmax_output = torch.stack([logsoftmax_output[k] for k in positive_document_list])
            # todo: sum or mean?
            loss_policy_v = -possitive_logsoftmax_output.mean()
            loss_policy_v = loss_policy_v.cuda()
            loss_policy_v.backward()
            retriever_optimizer.step()
            # temp_param_dict = get_net_parameter(net)
            # if count%100==0:
            #     print('     Epoch %d, %d samples have been trained.' %(i, count))
            # count+=1

        # Record trained parameters.
        if i % 1 == 0:
            MAP_list=[]
            for j in range(int(len(training_samples)/40)):
                random.seed(datetime.now())
                key, value = random.choice(list(training_samples.items()))
                if query_embedding:
                    question_tokens = qid_question_pair[key]
                    query_tensor = data.get_question_embedding(question_tokens, emb_dict, net1)
                else:
                    query_tensor = torch.tensor(net.pack_input(value['query_index']).tolist(), requires_grad=False).cuda()
                document_range = (value['document_range'][0], value['document_range'][1])
                logsoftmax_output, _, _ = net(query_tensor, document_range)
                order = net.calculate_rank(logsoftmax_output.tolist())
                positive_document_list = [k - value['document_range'][0] for k in value['positive_document_list']]
                orders = [order[k] for k in positive_document_list]
                MAP = mean(orders)
                MAP_list.append(MAP)
            MAP_for_queries = mean(MAP_list)
            print('------------------------------------------------------')
            print('Epoch %d, MAP_for_queries: %f' % (i, MAP_for_queries))
            print('------------------------------------------------------')
            if MAP_for_queries < max_value:
                max_value = MAP_for_queries
                if MAP_for_queries < 500:
                    isExists = os.path.exists(SAVES_DIR)
                    if not isExists:
                        os.makedirs(SAVES_DIR)
                    torch.save(net.state_dict(), os.path.join(SAVES_DIR, "new_AdaBound_epoch_%03d_%.3f.dat" % (i, MAP_for_queries)))
                    print('Save the state_dict: %s' % (str(i) + ' ' + str(MAP_for_queries)))
        if MAP_for_queries < 10:
            break

if __name__ == "__main__":
    epoches = 300
    w2v = 50
    seq2seq_word_embedding_path = '../data/saves/maml_att=0_newdata2k_reptile_1task/reptile_epoch_020_0.784_0.741.dat'
    # initialize_document_embedding(True, w2v, path)
    # establish_positive_question_documents_pair(MAX_TOKENS, True)
    # generate_training_samples(True)
    # If query_embedding is true, using the sum of word embedding to represent the questions.
    # If query_embedding is false, using the document_emb, which is stored in the retriever model,
    # to represent the questions.
    # If RETRIEVER_EMBED_FLAG is true, optimizing document_emb when training the retriever.
    # If RETRIEVER_EMBED_FLAG is false, document_emb is fixed when training.
    # retriever_training(epoches, RETRIEVER_EMBED_FLAG=True, query_embedding=True, w2v=w2v, seq2seq_word_embedding_path=seq2seq_word_embedding_path, int_flag=True)
    retriever_training(epoches, RETRIEVER_EMBED_FLAG=True, query_embedding=True, w2v=w2v,
                       seq2seq_word_embedding_path=seq2seq_word_embedding_path, int_flag=False)