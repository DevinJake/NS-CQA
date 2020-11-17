#!/usr/bin/env python3
import os
import sys
import random
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
from random import randrange
from libbots import data, model, utils

import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import ptan
import json
import math

SAVES_DIR = "../data/saves/nsm_webqsp"

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
MAX_TOKENS = 40
MAX_TOKENS_INT = 43
TRAIN_RATIO = 0.985
GAMMA = 0.05
MAX_MEMORY_BUFFER_SIZE = 10
# ALPHA is the bonus scalar.
# The value of α depends on the scale of task rewards.
ALPHA = 0.1

DIC_PATH = '../data/webqsp_data/share.webqsp.question'
DIC_PATH_INT = '../data/webqsp_data/share.webqsp.question'
# DIC_PATH_INT = '../data/auto_QA_data/share_944K_INT.question'
TRAIN_QUESTION_ANSWER_PATH = '../data/webqsp_data/final_webqsp_train_RL.json'
TRAIN_QUESTION_ANSWER_PATH_INT = '../data/webqsp_data/final_webqsp_train_RL.json'
log = logging.getLogger("train")



# Calculate 0-1 sparse reward for samples in test dataset to judge the performance of the model.
def run_test(test_data, net, rev_emb_dict, end_token, device="cuda"):
    argmax_reward_sum = 0.0
    argmax_reward_count = 0.0
    # p1 is one sentence, p2 is sentence list.
    for p1, p2 in test_data:
        # Transform sentence to padded embeddings.
        input_seq = net.pack_input(p1, net.emb, device)
        # Get hidden states from encoder.
        # enc = net.encode(input_seq)
        context, enc = net.encode_context(input_seq)
        # Decode sequence by feeding predicted token to the net again. Act greedily.
        # Return N*outputvocab, N output token indices.
        _, tokens = net.decode_chain_argmax(enc, net.emb(beg_token), seq_len=data.MAX_TOKENS, context=context[0],
                                            stop_at_token=end_token)
        # Show what the output action sequence is.
        action_tokens = []
        for temp_idx in tokens:
            if temp_idx in rev_emb_dict and rev_emb_dict.get(temp_idx) != '#END':
                action_tokens.append(str(rev_emb_dict.get(temp_idx)).upper())
        # Using 0-1 reward to compute accuracy.
        reward = utils.calc_True_Reward_webqsp_novar(action_tokens, p2, False)
        # reward = random.random()
        argmax_reward_sum += float(reward)
        argmax_reward_count += 1
    if argmax_reward_count == 0:
        return 0.0
    else:
        return float(argmax_reward_sum) / float(argmax_reward_count)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    # # command line parameters
    # # -a=True means using adaptive reward to train the model. -a=False is using 0-1 reward.
    sys.argv = ['train_scst_nsm_webqsp.py', '--cuda',
                '-l=../data/saves/webqsp/crossent_even_att=0_withINT/epoch_000_0.743_0.756.dat',
                '-n=rl_TR_1%_batch8_att=0_withINT_NSM', '-s=5', '-a=0', '--att=0', '--lstm=1', '--int',
                '-w2v=300', '--beam_width=10', '--NSM']
    # sys.argv = ['train_scst_true_reward.py', '--cuda', '-l=../data/saves/crossent_even_1%/pre_bleu_0.946_55.dat', '-n=rl_even_true_1%', '-s=5']
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", required=True, help="Category to use for training. Empty string to train on full processDataset")
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-l", "--load", required=True,
                        help="Load the pre-trained model whereby continue training the RL mode")
    # Number of decoding samples.
    parser.add_argument("-s", "--samples", type=int, default=4, help="Count of samples in prob mode")
    # The size of the beam search.
    parser.add_argument("--beam_width", type=int, default=10, help="Size of beam search")
    # The dimension of the word embeddings.
    parser.add_argument("-w2v", "--word_dimension", type=int, default=50, help="The dimension of the word embeddings")
    # Choose the function to compute reward (0-1 or adaptive reward).
    # If a = true, 1 or yes, the adaptive reward is used. Otherwise 0-1 reward is used.
    parser.add_argument("-a", "--adaptive", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="0-1 or adaptive reward")
    parser.add_argument("--disable-skip", default=False, action='store_true',
                        help="Disable skipping of samples with high argmax BLEU")
    parser.add_argument("--NSM", default=False, action='store_true',
                        help="Neural Symbolic Machine")
    # Choose the function to compute reward (0-1 or adaptive reward).
    # If a = true, 1 or yes, the adaptive reward is used. Otherwise 0-1 reward is used.
    parser.add_argument("--att", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using attention mechanism in seq2seq")
    parser.add_argument("--lstm", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using LSTM mechanism in seq2seq")
    # If false, the embedding tensors in the model do not need to be trained.
    parser.add_argument('--embed-grad', action='store_false', help='optimizing word embeddings when training')
    parser.add_argument('--int', action='store_true', help='training model with INT mask information')
    parser.add_argument("--MonteCarlo", action='store_true', default=False,
                        help="using Monte Carlo algorithm for REINFORCE")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    log.info("Device info: %s", str(device))

    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    # # List of (question, {question information and answer}) pairs, the training pairs are in format of 1:1.
    if args.int:
        phrase_pairs, emb_dict = data.load_RL_data_TR_INT(TRAIN_QUESTION_ANSWER_PATH_INT, DIC_PATH_INT, MAX_TOKENS_INT,
                                                          bool(args.NSM))
        log.info("Obtained %d phrase pairs with %d uniq words from %s with INT mask information.", len(phrase_pairs),
                 len(emb_dict), TRAIN_QUESTION_ANSWER_PATH_INT)
    else:
        phrase_pairs, emb_dict = data.load_RL_data_TR(TRAIN_QUESTION_ANSWER_PATH, DIC_PATH, MAX_TOKENS, bool(args.NSM))
        log.info("Obtained %d phrase pairs with %d uniq words from %s without INT mask information.", len(phrase_pairs),
                 len(emb_dict), TRAIN_QUESTION_ANSWER_PATH)

    data.save_emb_dict(saves_path, emb_dict)
    end_token = emb_dict[data.END_TOKEN]
    train_data = data.encode_phrase_pairs_RLTR(phrase_pairs, emb_dict)
    # # list of (seq1, [seq*]) pairs，把训练对做成1：N的形式；
    # train_data = data.group_train_data(train_data)
    train_data = data.group_train_data_RLTR(train_data)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    train_data, test_data = data.split_train_test(train_data, TRAIN_RATIO)
    log.info("Training data converted, got %d samples", len(train_data))
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))
    log.info("Batch size is %d", BATCH_SIZE)
    log.info("Beam search size is %d", args.beam_width)
    if args.att:
        log.info("Using attention mechanism to train the SEQ2SEQ model...")
    else:
        log.info("Train the SEQ2SEQ model without attention mechanism...")
    if args.lstm:
        log.info("Using LSTM mechanism to train the SEQ2SEQ model...")
    else:
        log.info("Using RNN mechanism to train the SEQ2SEQ model...")
    if args.MonteCarlo:
        log.info("Using Monte Carlo algorithm for Policy Gradient...")
    if args.NSM:
        log.info("Using Neural Symbolic Machine algorithm for RL...")

    # Index -> word.
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}
    # PhraseModel.__init__() to establish a LSTM model.
    net = model.PhraseModel(emb_size=args.word_dimension, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE,
                            LSTM_FLAG=args.lstm, ATT_FLAG=args.att).to(device)
    # Using cuda.
    net.cuda()
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)
    # Load the pre-trained seq2seq model.
    net.load_state_dict(torch.load(args.load))
    log.info("Model loaded from %s, continue training in RL mode...", args.load)
    if (args.adaptive):
        log.info("Using adaptive reward to train the REINFORCE model...")
    else:
        log.info("Using 0-1 sparse reward to train the REINFORCE model...")

    # BEGIN token
    beg_token = torch.LongTensor([emb_dict[data.BEGIN_TOKEN]]).to(device)
    beg_token = beg_token.cuda()

    # TBMeanTracker (TensorBoard value tracker):
    # allows to batch fixed amount of historical values and write their mean into TB
    with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
        # optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
        optimiser = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LEARNING_RATE, eps=1e-3)
        batch_idx = 0
        batch_count = 0
        best_true_reward = None
        time_start = time.time()

        print("start epoch, time: ", time_start)

        # Loop in epochs.
        for epoch in range(MAX_EPOCHS):
            random.shuffle(train_data)
            dial_shown = False

            # total_samples = 0
            # skipped_samples = 0
            true_reward_argmax = []
            true_reward_sample = []

            for batch in data.iterate_batches(train_data, BATCH_SIZE):
                batch_idx += 1
                # Each batch conduct one gradient upweight.
                batch_count += 1
                # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
                # It’s important to call this before loss.backward(),
                # otherwise you’ll accumulate the gradients from multiple passes.
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
                net_losses = []
                probability_normalization = []
                # Transform ID to embedding.
                beg_embedding = net.emb(beg_token)
                beg_embedding = beg_embedding.cuda()

                nsm_net_losses = []

                for idx, inp_idx in enumerate(input_batch):
                    # # Test whether the input sequence is correctly transformed into indices.
                    # input_tokens = [rev_emb_dict[temp_idx] for temp_idx in inp_idx]
                    # print (input_tokens)
                    # Get IDs of reference sequences' tokens corresponding to idx-th input sequence in batch.
                    qa_info = output_batch[idx]
                    # print("%s is training..." % (qa_info['qid']))
                    # print (qa_info['qid'])
                    # # Get the (two-layer) hidden state of encoder of idx-th input sequence in batch.
                    item_enc = net.get_encoded_item(enc, idx)
                    # # 'r_argmax' is the list of out_logits list and 'actions' is the list of output tokens.
                    # # The output tokens are generated greedily by using chain_argmax (using last setp's output token as current input token).
                    r_argmax, actions = net.decode_chain_argmax(item_enc, beg_embedding, data.MAX_TOKENS, context[idx],
                                                                stop_at_token=end_token)
                    # Show what the output action sequence is.
                    action_tokens = []
                    for temp_idx in actions:
                        if temp_idx in rev_emb_dict and rev_emb_dict.get(temp_idx) != '#END':
                            action_tokens.append(str(rev_emb_dict.get(temp_idx)).upper())
                    # Get the highest BLEU score as baseline used in self-critic.
                    # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
                    # Otherwise the adaptive reward is used.
                    argmax_reward = utils.calc_True_Reward_webqsp_novar(action_tokens, qa_info, args.adaptive)
                    # argmax_reward = random.random()
                    true_reward_argmax.append(argmax_reward)

                    if args.NSM and 'pseudo_gold_program_reward' not in qa_info:
                        pseudo_program_tokens = str(qa_info['pseudo_gold_program']).strip().split()
                        pseudo_program_reward = utils.calc_True_Reward_webqsp_novar(pseudo_program_tokens, qa_info, args.adaptive)
                        qa_info['pseudo_gold_program_reward'] = pseudo_program_reward

                    # # In this case, the BLEU score is so high that it is not needed to train such case with RL.
                    # if not args.disable_skip and argmax_reward > 0.99:
                    #     skipped_samples += 1
                    #     continue

                    # In one epoch, when model is optimized for the first time, the optimized result is displayed here.
                    # After that, all samples in this epoch don't display anymore.
                    if not dial_shown:
                        # data.decode_words transform IDs to tokens.
                        log.info("Input: %s", utils.untokenize(data.decode_words(inp_idx, rev_emb_dict)))
                        orig_response = qa_info['orig_response']
                        log.info("orig_response: %s", orig_response)
                        log.info("Argmax: %s, reward=%.4f", utils.untokenize(data.decode_words(actions, rev_emb_dict)),
                                 argmax_reward)

                    sample_logits_list, action_sequence_list = net.beam_decode(hid=item_enc, seq_len=data.MAX_TOKENS,
                                                                               context=context[idx],
                                                                               start_token=beg_token,
                                                                               stop_at_token=end_token,
                                                                               beam_width=args.beam_width,
                                                                               topk=args.samples)

                    qid = qa_info['qid']

                    # The data for each task in a batch of tasks.
                    inner_net_policies = []
                    inner_net_actions = []
                    inner_net_advantages = []
                    inner_probability_normalization = []

                    nsm_prob_list = []
                    nsm_advantage_list = []
                    nsm_alpha_list = []

                    for sample_index in range(args.samples):
                        # 'r_sample' is the list of out_logits list and 'actions' is the list of output tokens.
                        # The output tokens are sampled following probability by using chain_sampling.
                        if sample_index >= len(action_sequence_list) or sample_index >= len(sample_logits_list):
                            print("sample_index out of action_sequence_list or sample_logits_list range, sample_index: ", sample_index, len(action_sequence_list), len(sample_logits_list))
                            continue
                        actions = action_sequence_list[sample_index]
                        r_sample = sample_logits_list[sample_index]

                        # Show what the output action sequence is.
                        action_tokens = []
                        for temp_idx in actions:
                            if temp_idx in rev_emb_dict and rev_emb_dict.get(temp_idx) != '#END':
                                action_tokens.append(str(rev_emb_dict.get(temp_idx)).upper())
                        # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
                        # Otherwise the adaptive reward is used.
                        sample_reward = utils.calc_True_Reward_webqsp_novar(action_tokens, qa_info, args.adaptive)
                        # sample_reward = random.random()

                        if not dial_shown:
                            log.info("Sample: %s, reward=%.4f",
                                     utils.untokenize(data.decode_words(actions, rev_emb_dict)), sample_reward)

                        if args.MonteCarlo:
                            # Record the data for each task in a batch.
                            inner_net_policies.append(r_sample)
                            inner_net_actions.extend(actions)

                        elif args.NSM:
                            # Transform each action's logits to probabilities.
                            prob_v = F.softmax(r_sample, dim=1).to(device)
                            actions_t = torch.LongTensor(actions).to(device)
                            # Get each action's probability.
                            prob = prob_v[range(len(actions)), actions_t].to(device)
                            # Get the probability of the action sequence.
                            prob_prod = prob.prod().to(device)
                            # Get the unbiased rewaed.
                            advantage = sample_reward - argmax_reward
                            nsm_prob_list.append(prob_prod)
                            nsm_advantage_list.append(advantage)

                            # Handle alpha value used in NSM.
                            pseudo_program = str(qa_info['pseudo_gold_program']).strip()
                            predicted_program = (' '.join(action_tokens)).strip()
                            if pseudo_program == predicted_program:
                                nsm_alpha_list.append(float(ALPHA))
                            else:
                                nsm_alpha_list.append(0.0)
                            # When finding a better program, using the better one to replace the worse one.
                            # If R(C_j) > R(C*) then C* <- C_j;
                            if sample_reward > qa_info['pseudo_gold_program_reward']:
                                qa_info['pseudo_gold_program'] = predicted_program
                                qa_info['pseudo_gold_program_reward'] = sample_reward

                        else:
                            # Record the data for all tasks in a batch.
                            net_policies.append(r_sample)
                            net_actions.extend(actions)

                        if not args.NSM:
                            advantages = [sample_reward - argmax_reward] * len(actions)
                            if args.MonteCarlo:
                                inner_net_advantages.extend(advantages)
                            else:
                                net_advantages.extend(advantages)
                        true_reward_sample.append(sample_reward)

                    dial_shown = True
                    # Compute the loss for each task in a batch.
                    if args.NSM:
                        # Sum all the probabilities for all the generated action sequences as Σ(p_j')
                        question_prob_sum = torch.stack(nsm_prob_list).sum().to(device)
                        # p_j = (1-α)*p_j/Σ(p_j') + α (if C_j = C*)
                        # p_j = (1-α)*p_j/Σ(p_j') + 0.0 (if C_j ≠ C*)
                        nsm_prob = torch.stack(nsm_prob_list).to(device) * (1.0 - float(ALPHA))
                        nsm_prob_div = torch.div(nsm_prob, question_prob_sum)
                        nsm_prob_add = torch.add(nsm_prob_div, torch.FloatTensor(nsm_alpha_list).to(device))
                        prob_v = nsm_prob_add.to(device)
                        # p_j = log(p_j)
                        log_prob_v = torch.log(prob_v).to(device)
                        # J = Σ_j(p_j * adv_j)
                        log_prob_actions_v = torch.FloatTensor(nsm_advantage_list).to(device) * log_prob_v
                        log_prob_actions_v = log_prob_actions_v.to(device)
                        # Monte carlo simulation: 1/j * J.
                        # Loss is: - 1/j * J.
                        loss_policy_v = -log_prob_actions_v.mean().to(device)
                        nsm_net_losses.append(loss_policy_v)

                    elif args.MonteCarlo:
                        # Logits of all the output tokens whose size is 1 * N;
                        inner_policies_v = torch.cat(inner_net_policies).to(device)
                        # Indices of all the output tokens whose size is 1 * N;
                        inner_actions_t = torch.LongTensor(inner_net_actions).to(device)
                        # All output tokens reward whose size is 1 *pack_batch N;
                        inner_adv_v = torch.FloatTensor(inner_net_advantages).to(device)
                        # Compute log(softmax(logits)) of all output tokens in size of N * output vocab size;
                        inner_log_prob_v = F.log_softmax(inner_policies_v, dim=1).to(device)
                        # Q_1 = Q_2 =...= Q_n = BLEU(OUT,REF);
                        # ▽J = Σ_n[Q▽logp(T)] = ▽Σ_n[Q*logp(T)] = ▽[Q_1*logp(T_1)+Q_2*logp(T_2)+...+Q_n*logp(T_n)];
                        # log_prob_v[range(len(net_actions)), actions_t]: for each output, get the output token's log(softmax(logits)).
                        # adv_v * log_prob_v[range(len(net_actions)), actions_t]:
                        # get Q * logp(T) for all tokens of all decode_chain_sampling samples in size of 1 * N;
                        inner_log_prob_actions_v = inner_adv_v * inner_log_prob_v[
                            range(len(inner_net_actions)), inner_actions_t].to(device)
                        # For the optimizer is Adam (Adaptive Moment Estimation) which is a optimizer used for gradient descent.
                        # Therefore, to maximize ▽J (log_prob_actions_v) is to minimize -▽J.
                        # .mean() is to calculate Monte Carlo sampling.
                        inner_loss_policy_v = -inner_log_prob_actions_v.mean().to(device)
                        # Record the loss for each task in a batch.
                        net_losses.append(inner_loss_policy_v)

                if not net_policies and not net_losses and not nsm_net_losses:
                    continue

                # Data for decode_chain_sampling samples and the number of such samples is the same as args.samples parameter.
                if args.MonteCarlo:
                    batch_net_losses = torch.stack(net_losses).to(device)
                    # .mean() is utilized to calculate Mini-Batch Gradient Descent.
                    loss_policy_v = batch_net_losses.mean().to(device)

                elif args.NSM:
                    # Transform a list of tensors to a tensor.
                    batch_nsm_net_losses = torch.stack(nsm_net_losses).to(device)
                    # .mean() is utilized to calculate Mini-Batch Gradient Descent.
                    nsm_loss_policy_v = batch_nsm_net_losses.mean().to(device)

                else:
                    # Logits of all output tokens whose size is N * output vocab size; N is the number of output tokens of decode_chain_sampling samples.
                    policies_v = torch.cat(net_policies).to(device)
                    # Indices of all output tokens whose size is 1 * N;
                    actions_t = torch.LongTensor(net_actions).to(device)
                    # All output tokens reward whose size is 1 *pack_batch N;
                    adv_v = torch.FloatTensor(net_advantages).to(device)
                    # Compute log(softmax(logits)) of all output tokens in size of N * output vocab size;
                    log_prob_v = F.log_softmax(policies_v, dim=1).to(device)
                    # Q_1 = Q_2 =...= Q_n = BLEU(OUT,REF);
                    # ▽J = Σ_n[Q▽logp(T)] = ▽Σ_n[Q*logp(T)] = ▽[Q_1*logp(T_1)+Q_2*logp(T_2)+...+Q_n*logp(T_n)];
                    # log_prob_v[range(len(net_actions)), actions_t]: for each output, get the output token's log(softmax(logits)).
                    # adv_v * log_prob_v[range(len(net_actions)), actions_t]:
                    # get Q * logp(T) for all tokens of all decode_chain_sampling samples in size of 1 * N;
                    log_prob_actions_v = adv_v * log_prob_v[range(len(net_actions)), actions_t].to(device)
                    # For the optimizer is Adam (Adaptive Moment Estimation) which is a optimizer used for gradient descent.
                    # Therefore, to maximize ▽J (log_prob_actions_v) is to minimize -▽J.
                    # .mean() is used to calculate Monte Carlo sampling.
                    loss_policy_v = -log_prob_actions_v.mean().to(device)

                if args.NSM:
                    loss_v = nsm_loss_policy_v
                else:
                    loss_v = loss_policy_v
                # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
                # These are accumulated into x.grad for every parameter x. In pseudo-code:
                # x.grad += dloss/dx
                loss_v.backward()
                # To conduct a gradient ascent to minimize the loss (which is to maximize the reward).
                # optimizer.step updates the value of x using the gradient x.grad.
                # For example, the SGD optimizer performs:
                # x += -lr * x.grad
                optimiser.step()

                if not args.MonteCarlo and not args.NSM:
                    tb_tracker.track("advantage", adv_v, batch_idx)
                tb_tracker.track("loss_total", loss_v, batch_idx)

                log.info("Epoch %d, Batch %d is trained!", epoch, batch_count)

            # After one epoch, compute the bleus for samples in test dataset.
            true_reward_test = run_test(test_data, net, rev_emb_dict, end_token, str(device))
            # After one epoch, get the average of the decode_chain_argmax bleus for samples in training dataset.
            true_reward_armax = np.mean(true_reward_argmax)
            writer.add_scalar("true_reward_test", true_reward_test, batch_idx)
            writer.add_scalar("true_reward_armax", true_reward_armax, batch_idx)
            # After one epoch, get the average of the decode_chain_sampling bleus for samples in training dataset.
            writer.add_scalar("true_reward_sample", np.mean(true_reward_sample), batch_idx)
            # writer.add_scalar("skipped_samples", skipped_samples/total_samples if total_samples!=0 else 0, batch_idx)
            # log.info("Batch %d, skipped_samples: %d, total_samples: %d", batch_idx, skipped_samples, total_samples)
            writer.add_scalar("epoch", batch_idx, epoch)
            log.info("Epoch %d, test reward: %.3f", epoch, true_reward_test)
            if best_true_reward is None or best_true_reward < true_reward_test:
                best_true_reward = true_reward_test
                log.info("Best true reward updated: %.4f", true_reward_test)
                # Save the updated seq2seq parameters trained by RL.
                torch.save(net.state_dict(),
                           os.path.join(saves_path, "truereward_%.3f_%02d.dat" % (true_reward_test, epoch)))
            # if epoch % 10 == 0:
            # # The parameters are stored after each epoch.
            torch.save(net.state_dict(), os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" % (
                epoch, float(true_reward_armax), true_reward_test)))

        time_end = time.time()
        log.info("Training time is %.3fs." % (time_end - time_start))
        print("Training time is %.3fs." % (time_end - time_start))
    writer.close()
