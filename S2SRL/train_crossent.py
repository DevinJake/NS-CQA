#!/usr/bin/env python3
import os
import random
import argparse
import logging
import numpy as np
import sys
from tensorboardX import SummaryWriter
import time
from libbots import data, model, utils

import torch
import torch.optim as optim
import torch.nn.functional as F

SAVES_DIR = "../data/saves"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHES = 100
MAX_TOKENS = 40
MAX_TOKENS_INT = 43

log = logging.getLogger("train")

TEACHER_PROB = 1.0

TRAIN_QUESTION_PATH = '../data/auto_QA_data/mask_even/PT_train.question'
TRAIN_ACTION_PATH = '../data/auto_QA_data/mask_even/PT_train.action'
DIC_PATH = '../data/auto_QA_data/share.question'
TRAIN_QUESTION_PATH_INT = '../data/auto_QA_data/mask_even_1.0%/PT_train_INT.question'
TRAIN_ACTION_PATH_INT = '../data/auto_QA_data/mask_even_1.0%/PT_train_INT.action'
DIC_PATH_INT = '../data/auto_QA_data/share_INT.question'
# DIC_PATH_INT = '../data/auto_QA_data/share_944K_INT.question'

TRAIN_QUESTION_PATH_WEBQSP = '../data/webqsp_data/mask/PT_train.question'
TRAIN_ACTION_PATH_WEBQSP = '../data/webqsp_data/mask/PT_train.action'
DIC_PATH_WEBQSP = '../data/webqsp_data/share.webqsp.question'
TRAIN_QUESTION_PATH_INT_WEBQSP = '../data/webqsp_data/mask/PT_train.question'
TRAIN_ACTION_PATH_INT_WEBQSP = '../data/webqsp_data/mask/PT_train.action'
DIC_PATH_INT_WEBQSP = '../data/webqsp_data/share.webqsp.question'

def run_test(test_data, net, end_token, device="cuda"):
    bleu_sum = 0.0
    bleu_count = 0
    for p1, p2 in test_data:
        input_seq = net.pack_input(p1, net.emb, device)
        # enc = net.encode(input_seq)
        context, enc = net.encode_context(input_seq)
        # Return logits (N*outputvocab), res_tokens (1*N)
        # Always use the first token in input sequence, which is '#BEG' as the initial input of decoder.
        # The maximum length of the output is defined in class libbots.data.
        _, tokens = net.decode_chain_argmax(enc, input_seq.data[0:1],
                                            seq_len=data.MAX_TOKENS,
                                            context=context[0],
                                            stop_at_token=end_token)
        bleu_sum += utils.calc_bleu(tokens, p2[1:])
        bleu_count += 1
    return bleu_sum / bleu_count

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    # command line parameters
    sys.argv = ['train_crossent.py',
                '--cuda',
                '-d=csqa',
                '--n=crossent_even_1%_att=0_withINT',
                '--att=1',
                '--lstm=1',
                '--int',
                '-w2v=300']

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", required=True, help="Category to use for training. "
    #                                                   "Empty string to train on full processDataset")
    parser.add_argument("--cuda", action='store_true', default=False,
                        help="Enable cuda")
    parser.add_argument("-d", "--dataset", default="csqa", help="Name of the dataset")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    # Choose the function to compute reward (0-1 or adaptive reward).
    # If a = true, 1 or yes, the adaptive reward is used. Otherwise 0-1 reward is used.
    parser.add_argument("--att", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using attention mechanism in seq2seq")
    parser.add_argument("--lstm", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using LSTM mechanism in seq2seq")
    # If false, the embedding tensors in the model do not need to be trained.
    parser.add_argument('--embed-grad', action='store_false', help='the embeddings would not be optimized when training')
    parser.add_argument('--int', action='store_true', help='training model with INT mask information')
    parser.add_argument("-w2v", "--word_dimension", type=int, default=50, help="The dimension of the word embeddings")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    log.info("Device info: %s", str(device))

    saves_path = os.path.join(SAVES_DIR, args.name)
    isExists = os.path.exists(saves_path)
    if not isExists:
        os.makedirs(saves_path)
    # saves_path = os.path.join(SAVES_DIR, args.name)
    # os.makedirs(saves_path, exist_ok=True)

    # To get the input-output pairs and the relevant dictionary.
    if not args.int:
        log.info("Training model without INT mask information...")
        if args.dataset == "csqa":
            phrase_pairs, emb_dict = data.load_data_from_existing_data(TRAIN_QUESTION_PATH, TRAIN_ACTION_PATH, DIC_PATH, MAX_TOKENS)
        else:
            phrase_pairs, emb_dict = data.load_data_from_existing_data(TRAIN_QUESTION_PATH_WEBQSP, TRAIN_ACTION_PATH_WEBQSP, DIC_PATH_WEBQSP, MAX_TOKENS)


    if args.int:
        log.info("Training model with INT mask information...")
        if args.dataset == "csqa":
            phrase_pairs, emb_dict = data.load_data_from_existing_data(TRAIN_QUESTION_PATH_INT, TRAIN_ACTION_PATH_INT, DIC_PATH_INT, MAX_TOKENS_INT)
        else:
            phrase_pairs, emb_dict = data.load_data_from_existing_data(TRAIN_QUESTION_PATH_INT_WEBQSP, TRAIN_ACTION_PATH_INT_WEBQSP, DIC_PATH_INT_WEBQSP, MAX_TOKENS_INT)

    # Index -> word.
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}
    log.info("Obtained %d phrase pairs with %d uniq words from %s and %s.",
             len(phrase_pairs), len(emb_dict), TRAIN_QUESTION_PATH, TRAIN_ACTION_PATH)
    data.save_emb_dict(saves_path, emb_dict)
    end_token = emb_dict[data.END_TOKEN]
    # 将tokens转换为emb_dict中的indices;
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    log.info("Training data converted, got %d samples", len(train_data))
    train_data, test_data = data.split_train_test(train_data)
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))
    if args.att:
        log.info("Using attention mechanism to train the SEQ2SEQ model...")
    else:
        log.info("Train the SEQ2SEQ model without attention mechanism...")
    if args.lstm:
        log.info("Using LSTM mechanism to train the SEQ2SEQ model...")
    else:
        log.info("Using RNN mechanism to train the SEQ2SEQ model...")

    net = model.PhraseModel(emb_size=args.word_dimension, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE, LSTM_FLAG=args.lstm, ATT_FLAG=args.att).to(device)
    # 转到cuda
    net.cuda()
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)

    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    best_bleu = None

    time_start = time.time()

    for epoch in range(MAX_EPOCHES):
        losses = []
        bleu_sum = 0.0
        bleu_count = 0
        dial_shown = False
        random.shuffle(train_data)

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        # model.train() tells your model that you are training the model.
        # So effectively layers like dropout, batchnorm etc.
        # which behave different on the train and test procedures know
        # what is going on and hence can behave accordingly.
        net.train()

        for batch in data.iterate_batches(train_data, BATCH_SIZE):
            optimiser.zero_grad()
            # input_idx：一个batch的输入句子的tokens对应的ID矩阵；
            # output_idx：一个batch的输出句子的tokens对应的ID矩阵；
            ''' 猜测input_seq：一个batch输入的所有tokens的embedding，大小为358*50；
            tensor([[-1.0363, -1.6041,  0.1451,  ..., -1.0645,  0.2387,  1.2934],
        [-1.0363, -1.6041,  0.1451,  ..., -1.0645,  0.2387,  1.2934],
        [-1.0363, -1.6041,  0.1451,  ..., -1.0645,  0.2387,  1.2934],
        ...,
        [ 0.5198, -0.3963,  1.4022,  ...,  1.0182,  0.2710, -1.5520],
        [ 2.1937, -0.5535, -0.9000,  ..., -0.1032,  0.3514, -1.2759],
        [-0.8078,  0.1575,  1.1064,  ...,  0.1365,  0.4121, -0.4211]],
       device='cuda:0')'''
            input_seq, out_seq_list, _, out_idx = net.pack_batch(batch, net.emb, device)
            # net.encode calls nn.LSTM by which the forward function is called to run the neural network.
            # enc is a batch of last time step's hidden state outputted by encoder.
            # enc = net.encode(input_seq)
            context, enc = net.encode_context(input_seq)

            net_results = []
            net_targets = []
            for idx, out_seq in enumerate(out_seq_list):
                ref_indices = out_idx[idx][1:]
                # Get the last step's hidden state and cell state of encoder for the idx-th element in a batch.
                enc_item = net.get_encoded_item(enc, idx)
                # Using teacher forcing to train the model.
                if random.random() < TEACHER_PROB:
                    context_temp = context[idx]
                    r = net.decode_teacher(enc_item, out_seq, context[idx])
                    blue_temp = net.seq_bleu(r, ref_indices)
                    bleu_sum += blue_temp
                    # Get predicted tokens.
                    seq = torch.max(r.data, dim=1)[1]
                    seq = seq.cpu().numpy()
                # argmax做训练；
                else:
                    r, seq = net.decode_chain_argmax(enc_item, out_seq.data[0:1],
                                                     len(ref_indices), context[idx])
                    blue_temp = utils.calc_bleu(seq, ref_indices)
                    bleu_sum += blue_temp
                net_results.append(r)
                net_targets.extend(ref_indices)
                bleu_count += 1

                if not dial_shown:
                    # data.decode_words transform IDs to tokens.
                    ref_words = [utils.untokenize(data.decode_words(ref_indices, rev_emb_dict))]
                    log.info("Reference: %s", " ~~|~~ ".join(ref_words))
                    log.info("Predicted: %s, bleu=%.4f", utils.untokenize(data.decode_words(seq, rev_emb_dict)), blue_temp)
                    dial_shown = True
            results_v = torch.cat(net_results)
            results_v = results_v.cuda()
            targets_v = torch.LongTensor(net_targets).to(device)
            targets_v = targets_v.cuda()
            loss_v = F.cross_entropy(results_v, targets_v)
            loss_v = loss_v.cuda()
            loss_v.backward()
            optimiser.step()

            losses.append(loss_v.item())
        bleu = bleu_sum / bleu_count
        bleu_test = run_test(test_data, net, end_token, device)
        log.info("Epoch %d: mean loss %.3f, mean BLEU %.3f, test BLEU %.3f",
                 epoch, np.mean(losses), bleu, bleu_test)
        writer.add_scalar("loss", np.mean(losses), epoch)
        writer.add_scalar("bleu", bleu, epoch)
        writer.add_scalar("bleu_test", bleu_test, epoch)
        if best_bleu is None or best_bleu < bleu_test:
            if best_bleu is not None:
                out_name = os.path.join(saves_path, "pre_bleu_%.3f_%02d.dat" %
                                        (bleu_test, epoch))
                torch.save(net.state_dict(), out_name)
                log.info("Best BLEU updated %.3f", bleu_test)
            best_bleu = bleu_test

        if epoch % 10 == 0:
            out_name = os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" %
                                    (epoch, bleu, bleu_test))
            torch.save(net.state_dict(), out_name)
        print ("------------------Epoch " + str(epoch) + ": training is over.------------------")

    time_end = time.time()
    log.info("Training time is %.3fs." % (time_end - time_start))
    print("Training time is %.3fs." % (time_end - time_start))

    writer.close()
