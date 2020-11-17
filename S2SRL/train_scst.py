#!/usr/bin/env python3
import os
import sys
import random
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter

from libbots import data, model, utils

import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import ptan

SAVES_DIR = "../data/saves"

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHES = 40
MAX_TOKENS = 40
TRAIN_RATIO = 0.985

TRAIN_QUESTION_PATH = '../data/auto_QA_data/mask_even/RL_train.question'
TRAIN_ACTION_PATH = '../data/auto_QA_data/mask_even/RL_train.action'
DIC_PATH = '../data/auto_QA_data/share.question'

log = logging.getLogger("train")

# Calculate bleus for samples in test dataset.
def run_test(test_data, net, end_token, device="cuda"):
    bleu_sum = 0.0
    bleu_count = 0
    # p1 is one sentence, p2 is sentence list.
    for p1, p2 in test_data:
        # Transform sentence to padded embeddings.
        input_seq = net.pack_input(p1, net.emb, device)
        # Get hidden states from encoder.
        # enc = net.encode(input_seq)
        context, enc = net.encode_context(input_seq)
        # Decode sequence by feeding predicted token to the net again. Act greedily.
        # Return N*outputvocab, N output token indices.
        _, tokens = net.decode_chain_argmax(enc, input_seq.data[0:1], seq_len=data.MAX_TOKENS,
                                            context = context[0], stop_at_token=end_token)
        ref_indices = [
            # Remove #BEG from sentence.
            indices[1:]
            for indices in p2
        ]
        # BEG is not included in tokens.
        # Accept several reference sentences and return the one with the best score.
        bleu_sum += utils.calc_bleu_many(tokens, ref_indices)
        bleu_count += 1
    return bleu_sum / bleu_count


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    # # command line parameters
    # sys.argv = ['train_crossent.py', '--cuda', '-l=../data/saves/crossent/pre_bleu_0.942_18.dat', '-n=rl']

    # Read optimized parameters from pre-training SEQ2SEQ.
    sys.argv = ['train_crossent.py', '--cuda', '-l=../data/saves/1crossent_even_1%/pre_bleu_0.786_02.dat', '-n=1rl_even_1%', '--att=0', '--lstm=1']

    # # Read optimized parameters from pre-training RL.
    # sys.argv = ['train_crossent.py', '--cuda', '-l=../data/saves/rl_even_1%/bleu_0.995_03.dat', '-n=rl_even_1%']
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", required=True, help="Category to use for training. Empty string to train on full processDataset")
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-l", "--load", required=True, help="Load the s2s model whereby continue training the RL mode")
    # Number of decoding samples.
    parser.add_argument("--samples", type=int, default=4, help="Count of samples in prob mode")
    parser.add_argument("--disable-skip", default=False, action='store_true', help="Disable skipping of samples with high argmax BLEU")
    # Choose the function to compute reward (0-1 or adaptive reward).
    # If a = true, 1 or yes, the adaptive reward is used. Otherwise 0-1 reward is used.
    parser.add_argument("--att", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using attention mechanism in seq2seq")
    parser.add_argument("--lstm", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using LSTM mechanism in seq2seq")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    log.info("Device info: %s", str(device))

    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    # phrase_pairs, emb_dict = data.load_data('comedy')
    # List of (seq1, [seq*]) pairs, the training pairs are in format of 1:N.
    phrase_pairs, emb_dict = data.load_RL_data(TRAIN_QUESTION_PATH, TRAIN_ACTION_PATH, DIC_PATH, MAX_TOKENS)
    log.info("Obtained %d phrase pairs with %d uniq words from %s and %s.", len(phrase_pairs), len(emb_dict), TRAIN_QUESTION_PATH, TRAIN_ACTION_PATH)
    data.save_emb_dict(saves_path, emb_dict)
    end_token = emb_dict[data.END_TOKEN]
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    # list of (seq1, [seq*]) pairs，把训练对做成1：N的形式；
    train_data = data.group_train_data(train_data)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    train_data, test_data = data.split_train_test(train_data, TRAIN_RATIO)
    log.info("Training data converted, got %d samples", len(train_data))
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))
    if (args.att):
        log.info("Using attention mechanism to train the SEQ2SEQ model...")
    else:
        log.info("Train the SEQ2SEQ model without attention mechanism...")
    if (args.lstm):
        log.info("Using LSTM mechanism to train the SEQ2SEQ model...")
    else:
        log.info("Using RNN mechanism to train the SEQ2SEQ model...")

    # Index -> word.
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}
    # PhraseModel.__init__() to establish a LSTM model.
    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE, LSTM_FLAG=args.lstm, ATT_FLAG=args.att).to(device)
    # Using cuda.
    net.cuda()
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)
    # Load the pre-trained seq2seq model.
    net.load_state_dict(torch.load(args.load))
    log.info("Model loaded from %s, continue training in RL mode...", args.load)

    # BEGIN token
    beg_token = torch.LongTensor([emb_dict[data.BEGIN_TOKEN]]).to(device)
    beg_token = beg_token.cuda()

    # TBMeanTracker (TensorBoard value tracker):
    # allows to batch fixed amount of historical values and write their mean into TB
    with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
        optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
        batch_idx = 0
        best_bleu = None

        time_start = time.time()

        # Loop in epoches.
        for epoch in range(MAX_EPOCHES):
            random.shuffle(train_data)
            dial_shown = False

            total_samples = 0
            skipped_samples = 0
            bleus_argmax = []
            bleus_sample = []

            for batch in data.iterate_batches(train_data, BATCH_SIZE):
                batch_idx += 1
                # Each batch conduct one gradient upweight.
                optimiser.zero_grad()
                # input_seq: the padded and embedded batch-sized input sequence.
                # input_batch: the token ID matrix of batch-sized input sequence. Each row is corresponding to one input sentence.
                # output_batch: the token ID matrix of batch-sized output sequences. Each row is corresponding to a list of several output sentences.
                input_seq, input_batch, output_batch = net.pack_batch_no_out(batch, net.emb, device)
                input_seq = input_seq.cuda()
                # Get (two-layer) hidden state of encoder of samples in batch.
                # enc = net.encode(input_seq)
                context, enc = net.encode_context(input_seq)

                net_policies = []
                net_actions = []
                net_advantages = []
                # Transform ID to embedding.
                beg_embedding = net.emb(beg_token)
                beg_embedding = beg_embedding.cuda()

                for idx, inp_idx in enumerate(input_batch):
                    total_samples += 1
                    # Get IDs of reference sequences' tokens corresponding to idx-th input sequence in batch.
                    ref_indices = [
                        indices[1:]
                        for indices in output_batch[idx]
                    ]
                    # Get the (two-layer) hidden state of encoder of idx-th input sequence in batch.
                    item_enc = net.get_encoded_item(enc, idx)
                    # 'r_argmax' is the list of out_logits list and 'actions' is the list of output tokens.
                    # The output tokens are generated greedily by using chain_argmax (using last setp's output token as current input token).
                    r_argmax, actions = net.decode_chain_argmax(item_enc, beg_embedding, data.MAX_TOKENS,
                                                                context[idx], stop_at_token=end_token)
                    # Get the highest BLEU score as baseline used in self-critic.
                    argmax_bleu = utils.calc_bleu_many(actions, ref_indices)
                    bleus_argmax.append(argmax_bleu)

                    # In this case, the BLEU score is so high that it is not needed to train such case with RL.
                    if not args.disable_skip and argmax_bleu > 0.99:
                        skipped_samples += 1
                        continue

                    # In one epoch, when model is optimized for the first time, the optimized result is displayed here.
                    # After that, all samples in this epoch don't display anymore.
                    if not dial_shown:
                        # data.decode_words transform IDs to tokens.
                        log.info("Input: %s", utils.untokenize(data.decode_words(inp_idx, rev_emb_dict)))
                        ref_words = [utils.untokenize(data.decode_words(ref, rev_emb_dict)) for ref in ref_indices]
                        log.info("Refer: %s", " ~~|~~ ".join(ref_words))
                        log.info("Argmax: %s, bleu=%.4f", utils.untokenize(data.decode_words(actions, rev_emb_dict)),
                                 argmax_bleu)

                    for _ in range(args.samples):
                        # 'r_sample' is the list of out_logits list and 'actions' is the list of output tokens.
                        # The output tokens are sampled following probabilitis by using chain_sampling.
                        r_sample, actions = net.decode_chain_sampling(item_enc, beg_embedding, data.MAX_TOKENS, context[idx], stop_at_token=end_token)
                        sample_bleu = utils.calc_bleu_many(actions, ref_indices)

                        if not dial_shown:
                            log.info("Sample: %s, bleu=%.4f", utils.untokenize(data.decode_words(actions, rev_emb_dict)), sample_bleu)

                        net_policies.append(r_sample)
                        net_actions.extend(actions)
                        # Regard argmax_bleu calculated from decode_chain_argmax as baseline used in self-critic.
                        # Each token has same reward as 'sample_bleu - argmax_bleu'.
                        # [x] * y: stretch 'x' to [1*y] list in which each element is 'x'.
                        net_advantages.extend([sample_bleu - argmax_bleu] * len(actions))
                        bleus_sample.append(sample_bleu)
                    dial_shown = True

                # Provided all output samples have higher bleu than 0.99, this epoch has no need to optimize the RL model.
                if not net_policies:
                    continue

                # Data for decode_chain_sampling samples and the number of such samples is the same as args.samples parameter.
                # Logits of all output tokens whose size is N * output vocab size; N is the number of output tokens of decode_chain_sampling samples.
                policies_v = torch.cat(net_policies)
                policies_v = policies_v.cuda()
                # Indices of all output tokens whose size is 1 * N;
                actions_t = torch.LongTensor(net_actions).to(device)
                actions_t = actions_t.cuda()
                # All output tokens reward whose size is 1 * N;
                adv_v = torch.FloatTensor(net_advantages).to(device)
                adv_v = adv_v.cuda()
                # Compute log(softmax(logits)) of all output tokens in size of N * output vocab size;
                log_prob_v = F.log_softmax(policies_v, dim=1)
                log_prob_v = log_prob_v.cuda()
                # Q_1 = Q_2 =...= Q_n = BLEU(OUT,REF);
                # ▽J = Σ_n[Q▽logp(T)] = ▽Σ_n[Q*logp(T)] = ▽[Q_1*logp(T_1)+Q_2*logp(T_2)+...+Q_n*logp(T_n)];
                # log_prob_v[range(len(net_actions)), actions_t]: for each output, get the output token's log(softmax(logits)).
                # adv_v * log_prob_v[range(len(net_actions)), actions_t]:
                # get Q * logp(T) for all tokens of all decode_chain_sampling samples in size of 1 * N;
                log_prob_actions_v = adv_v * log_prob_v[range(len(net_actions)), actions_t]
                log_prob_actions_v = log_prob_actions_v.cuda()
                # To maximize ▽J (log_prob_actions_v) is to minimize -▽J.
                # .mean() is to calculate  Monte Carlo sampling.
                loss_policy_v = -log_prob_actions_v.mean()
                loss_policy_v = loss_policy_v.cuda()

                loss_v = loss_policy_v
                loss_v.backward()
                # To conduct a gradient ascent to minimize the loss (which is to maximize the reward).
                optimiser.step()

                tb_tracker.track("advantage", adv_v, batch_idx)
                tb_tracker.track("loss_policy", loss_policy_v, batch_idx)
                tb_tracker.track("loss_total", loss_v, batch_idx)

            # After one epoch, compute the bleus for samples in test dataset.
            bleu_test = run_test(test_data, net, end_token, device)
            # After one epoch, get the average of the decode_chain_argmax bleus for samples in training dataset.
            bleu = np.mean(bleus_argmax)
            writer.add_scalar("bleu_test", bleu_test, batch_idx)
            writer.add_scalar("bleu_argmax", bleu, batch_idx)
            # After one epoch, get the average of the decode_chain_sampling bleus for samples in training dataset.
            writer.add_scalar("bleu_sample", np.mean(bleus_sample), batch_idx)
            writer.add_scalar("skipped_samples", skipped_samples/total_samples if total_samples!=0 else 0, batch_idx)
            writer.add_scalar("epoch", batch_idx, epoch)
            log.info("Epoch %d, test BLEU: %.3f", epoch, bleu_test)
            if best_bleu is None or best_bleu < bleu_test:
                best_bleu = bleu_test
                log.info("Best bleu updated: %.4f", bleu_test)
                # Save the updated seq2seq parameters trained by RL.
                torch.save(net.state_dict(), os.path.join(saves_path, "bleu_%.3f_%02d.dat" % (bleu_test, epoch)))
            if epoch % 10 == 0:
                torch.save(net.state_dict(), os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" % (epoch, float(bleu), bleu_test)))

        time_end = time.time()
        log.info("Training time is %.3fs." % (time_end - time_start))
        print("Training time is %.3fs." % (time_end - time_start))

    writer.close()
