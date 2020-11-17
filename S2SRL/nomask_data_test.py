# !/usr/bin/env python3
# The file is used to predict the action sequences for full-data test dataset.
import argparse
import logging
import sys

from libbots import data, model, utils

import torch

log = logging.getLogger("nomask_data_test")

DIC_PATH = '../data/auto_QA_data/nomask_share.question'

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    # # command line parameters for final test
    # sys.argv = ['data_test.py', '-m=bleu_0.984_09.dat', '-p=final', '--n=rl_even']
    # command line parameters for final test (subset data)
    sys.argv = ['nomask_data_test.py', '-m=bleu_0.984_09.dat', '-p=sample_final', '--n=nomask_rl_even']

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", required=True,
    #                     help="Category to use for training. Empty string to train on full processDataset")
    parser.add_argument("-m", "--model", required=True, help="Model name to load")
    parser.add_argument("-p", "--pred", required=True, help="the test processDataset format, " \
                                                            "py is one-to-one (one sentence with one reference), rl is one-to-many")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    PREDICT_PATH = '../data/saves/' + str(args.name) + '/' + str(args.pred) + '_predict.actions'
    fwPredict = open(PREDICT_PATH, 'w', encoding="UTF-8")
    REFER_PATH = '../data/saves/' + str(args.name) + '/' + str(args.pred) + '_refer.actions'
    fwRefer = open(REFER_PATH, 'w', encoding="UTF-8")

    phrase_pairs, emb_dict = [], list()
    TEST_QUESTION_PATH = '../data/auto_QA_data/nomask_test/' + str(args.pred).upper() + '_test.question'
    log.info("Open: %s", '../data/auto_QA_data/nomask_test/' + str(args.pred).upper() + '_test.question')
    TEST_ACTION_PATH = '../data/auto_QA_data/nomask_test/' + str(args.pred).upper() + '_test.action'
    log.info("Open: %s", '../data/auto_QA_data/nomask_test/' + str(args.pred).upper() + '_test.action')
    if args.pred == 'pt' or 'final' in args.pred:
        phrase_pairs, emb_dict = data.load_data_from_existing_data(TEST_QUESTION_PATH, TEST_ACTION_PATH, DIC_PATH)
    elif args.pred == 'rl':
        phrase_pairs, emb_dict = data.load_RL_data(TEST_QUESTION_PATH, TEST_ACTION_PATH, DIC_PATH)
    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    if args.pred == 'rl':
        train_data = data.group_train_data(train_data)
    else:
        train_data = data.group_train_data_one_to_one(train_data)
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE)
    net = net.cuda()
    model_path = '../data/saves/' + str(args.name) + '/' + str(args.model)
    net.load_state_dict((torch.load(model_path)))
    end_token = emb_dict[data.END_TOKEN]

    seq_count = 0
    correct_count = 0
    sum_bleu = 0.0

    test_dataset_count = 0
    token_string_list = list()
    refer_string_list = list()
    # seq_1是輸入，targets是references，可能有多個；
    for seq_1, targets in train_data:
        test_dataset_count += 1
        input_seq = net.pack_input(seq_1, net.emb)
        enc = net.encode(input_seq)
        _, tokens = net.decode_chain_argmax(enc, input_seq.data[0:1],
                                            seq_len=data.MAX_TOKENS, stop_at_token=end_token)
        references = [seq[1:] for seq in targets]
        # references = [seq[1:] if seq[1:] != '' else ['NONE'] for seq in targets]
        token_string, reference_string = '', ''
        for token in tokens:
            if token in rev_emb_dict and rev_emb_dict.get(token)!= '#END':
                token_string += str(rev_emb_dict.get(token)).upper() + ' '
        token_string = token_string.strip()
        # log.info("%d PREDICT: %s", test_dataset_count, token_string)
        token_string_list.append(str(test_dataset_count) + ': ' + token_string+'\n')
        if test_dataset_count % 1000 == 0:
            print (test_dataset_count)

        flag = False
        for reference in references:
            reference_string = ''
            for token in reference:
                if token in rev_emb_dict and rev_emb_dict.get(token)!= '#END':
                    reference_string += str(rev_emb_dict.get(token)).upper() + ' '
            reference_string = reference_string.strip()
            if token_string == reference_string:
                flag = True
            # log.info("%d REFER: %s", test_dataset_count, reference_string)
            refer_string_list.append(str(test_dataset_count) + ': ' + reference_string + '\n')

        bleu = utils.calc_bleu_many(tokens, references)
        if flag == True:
            correct_count += 1
        # log.info("%d bleu: %f", test_dataset_count, bleu)
        sum_bleu += bleu
        seq_count += 1
    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    log.info("Processed %d phrases, mean BLEU = %.4f", seq_count, sum_bleu / seq_count)
    log.info("Processed %d phrases, correctness = %.4f", seq_count, correct_count / seq_count)
    fwPredict.writelines(token_string_list)
    fwPredict.close()
    fwRefer.writelines(refer_string_list)
    fwRefer.close()
    log.info("Writing to file %s is done!", PREDICT_PATH)
    log.info("Writing to file %s is done!", REFER_PATH)
