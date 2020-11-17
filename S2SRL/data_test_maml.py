# !/usr/bin/env python3
# The file is used to predict the action sequences for full-data test dataset.
import argparse
import logging
import sys

from libbots import data, model, utils, metalearner

import torch
log = logging.getLogger("data_test")

DIC_PATH = '../data/auto_QA_data/share.question'
TRAIN_944K_QUESTION_ANSWER_PATH = '../data/auto_QA_data/CSQA_DENOTATIONS_full_944K.json'
DICT_944K = '../data/auto_QA_data/CSQA_result_question_type_944K.json'
DICT_944K_WEAK = '../data/auto_QA_data/CSQA_result_question_type_count944K.json'
MAX_TOKENS = 40

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    # # command line parameters for final test
    # sys.argv = ['data_test.py', '-m=bleu_0.984_09.dat', '-p=final', '--n=rl_even']
    # command line parameters for final test (subset data)
    # Args for 1st-order maml.
    sys.argv = ['data_test_maml.py', '-m=epoch_025_0.741_0.710.dat', '-p=sample_final_maml', '--n=maml_newdata2k_1storder_1e-3', '--cuda', '-s=5', '-a=0', '--att=0', '--lstm=1', '--fast-lr=1e-3', '--meta-lr=1e-4', '--steps=5', '--batches=1', '--weak=1', '--embed-grad']
    # Args for reptile.
    # sys.argv = ['data_test_maml.py', '-m=epoch_019_0.767_0.741.dat', '-p=sample_final_maml',
    #             '--n=maml_att=0_newdata2k_reptile_random_retriever', '--cuda', '-s=5', '-a=0', '--att=0', '--lstm=1',
    #             '--fast-lr=1e-4', '--meta-lr=1e-4', '--steps=5', '--batches=1', '--weak=1', '--embed-grad', '--retriever-random']
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", required=True,
    #                     help="Category to use for training. Empty string to train on full processDataset")
    parser.add_argument("-m", "--model", required=True, help="Model name to load")
    parser.add_argument("-p", "--pred", required=True, help="the test processDataset format, " \
                                                            "py is one-to-one (one sentence with one reference), rl is one-to-many")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    # Choose the function to compute reward (0-1 or adaptive reward).
    # If a = true, 1 or yes, the adaptive reward is used. Otherwise 0-1 reward is used.
    parser.add_argument("--att", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using attention mechanism in seq2seq")
    parser.add_argument("--lstm", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using LSTM mechanism in seq2seq")
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda")
    # Number of decoding samples.
    parser.add_argument("-s", "--samples", type=int, default=4, help="Count of samples in prob mode")
    # The action='store_true' means once the parameter is assigned a value, the action is to mark it as 'True';
    # If there is no value of the parameter, the value is assigned as 'False'.
    # Conversely, if action is 'store_false', if the parameter has a value, the parameter is viewed as 'False'.
    parser.add_argument('--first-order', action='store_true', help='use the first-order approximation of MAML')
    parser.add_argument('--fast-lr', type=float, default=0.0001,
                        help='learning rate for the 1-step gradient update of MAML')
    parser.add_argument('--meta-lr', type=float, default=0.0001,
                        help='learning rate for the meta optimization')
    parser.add_argument('--steps', type=int, default=5, help='steps in inner loop of MAML')
    parser.add_argument('--batches', type=int, default=5, help='tasks of a batch in outer loop of MAML')
    # If weak is true, it means when searching for support set, the questions with same number of E/R/T nut different relation will be retrieved if the questions in this pattern is less than the number of steps.
    parser.add_argument("--weak", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using weak mode to search for support set")
    # If a = true, 1 or yes, the adaptive reward is used. Otherwise 0-1 reward is used.
    parser.add_argument("-a", "--adaptive", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="0-1 or adaptive reward")
    # If false, the embedding tensors in the model do not need to be trained.
    parser.add_argument('--embed-grad', action='store_false', help='fix embeddings when training')
    parser.add_argument('--retriever-random', action='store_true', help='randomly get support set for the retriever')
    parser.add_argument("--MonteCarlo", action='store_true', default=False,
                        help="using Monte Carlo algorithm for REINFORCE")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    log.info("Device info: %s", str(device))

    PREDICT_PATH = '../data/saves/' + str(args.name) + '/' + str(args.pred) + '_predict.actions'
    fwPredict = open(PREDICT_PATH, 'w', encoding="UTF-8")

    TEST_QUESTION_PATH = '../data/auto_QA_data/mask_test/' + str(args.pred).upper() + '_test.question'
    log.info("Open: %s", '../data/auto_QA_data/mask_test/' + str(args.pred).upper() + '_test.question')

    phrase_pairs, emb_dict = data.load_data_MAML_TEST(QUESTION_PATH=TEST_QUESTION_PATH, DIC_PATH=DIC_PATH)
    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    phrase_pairs_944K = data.load_data_MAML(TRAIN_944K_QUESTION_ANSWER_PATH, max_tokens=MAX_TOKENS)
    log.info("Obtained %d phrase pairs from %s.", len(phrase_pairs_944K), TRAIN_944K_QUESTION_ANSWER_PATH)

    if args.retriever_random:
        log.info("Using random support set for test.")

    end_token = emb_dict[data.END_TOKEN]
    # Transform token into index in dictionary.
    test_data = data.encode_phrase_pairs_RLTR(phrase_pairs, emb_dict)
    # # list of (seq1, [seq*]) pairs，把训练对做成1：N的形式；
    # train_data = data.group_train_data(train_data)
    test_data = data.group_train_data_RLTR(test_data)

    train_data_944K = data.encode_phrase_pairs_RLTR(phrase_pairs_944K, emb_dict)
    train_data_944K = data.group_train_data_RLTR_for_support(train_data_944K)
    dict944k = data.get944k(DICT_944K)
    log.info("Reading dict944k from %s is done. %d pairs in dict944k.", DICT_944K, len(dict944k))
    dict944k_weak = data.get944k(DICT_944K_WEAK)
    log.info("Reading dict944k_weak from %s is done. %d pairs in dict944k_weak", DICT_944K_WEAK, len(dict944k_weak))

    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE,
                            LSTM_FLAG=args.lstm, ATT_FLAG=args.att, EMBED_FLAG=args.embed_grad).to(device)

    model_path = '../data/saves/' + str(args.name) + '/' + str(args.model)
    net.load_state_dict((torch.load(model_path)))
    log.info("Model loaded from %s, continue testing in MAML first-order mode...", model_path)

    # BEGIN token
    beg_token = torch.LongTensor([emb_dict[data.BEGIN_TOKEN]]).to(device)
    beg_token = beg_token.cuda()

    metaLearner = metalearner.MetaLearner(net, device=device, beg_token=beg_token, end_token=end_token,
                                          adaptive=args.adaptive, samples=args.samples,
                                          train_data_support_944K=train_data_944K, rev_emb_dict=rev_emb_dict,
                                          first_order=args.first_order, fast_lr=args.fast_lr,
                                          meta_optimizer_lr=args.meta_lr, dial_shown=False, dict=dict944k,
                                          dict_weak=dict944k_weak, steps=args.steps, weak_flag=args.weak)
    log.info("Meta-learner: %d inner steps, %f inner learning rate, "
             "%d outer steps, %f outer learning rate, using weak mode:%s"
             % (args.steps, args.fast_lr, args.batches, args.meta_lr, str(args.weak)))

    seq_count = 0
    correct_count = 0
    sum_bleu = 0.0

    test_dataset_count = 0
    token_string_list = list()
    refer_string_list = list()
    batch_count = 0
    # seq_1是輸入，targets是references，可能有多個；

    # The dict stores the initial parameters in the modules.
    old_param_dict = metaLearner.get_net_named_parameter()

    for test_task in test_data:
        batch_count += 1
        # Batch is represented for a batch of tasks in MAML.
        # In each task, a batch of support set is established.
        token_string = metaLearner.first_order_sampleForTest(test_task, old_param_dict=old_param_dict, random=args.retriever_random, mc=args.MonteCarlo)

        test_dataset_count += 1
        # log.info("%d PREDICT: %s", test_dataset_count, token_string)
        token_string_list.append(str(test_task[1]['qid']) + ': ' + token_string+'\n')
        if test_dataset_count % 100 == 0:
            print (test_dataset_count)

    fwPredict.writelines(token_string_list)
    fwPredict.close()
    log.info("Writing to file %s is done!", PREDICT_PATH)