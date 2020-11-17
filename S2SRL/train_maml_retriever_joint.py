#!/usr/bin/env python3
import os
import sys
import random
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter

from libbots import data, model, utils, metalearner, retriever_module

import torch
import torch.optim as optim
import time
import ptan

SAVES_DIR = "../data/saves"

MAX_EPOCHES = 4
MAX_TOKENS = 40
TRAIN_RATIO = 0.985
GAMMA = 0.05

DIC_PATH = '../data/auto_QA_data/share.question'
TRAIN_QUESTION_ANSWER_PATH = '../data/auto_QA_data/mask_even_1.0%/RL_train_TR_new_5.question'
TRAIN_944K_QUESTION_ANSWER_PATH = '../data/auto_QA_data/CSQA_DENOTATIONS_full_944K.json'
DICT_944K = '../data/auto_QA_data/CSQA_result_question_type_944K.json'
DICT_944K_WEAK = '../data/auto_QA_data/CSQA_result_question_type_count944K.json'
ORDERED_QID_QUESTION_DICT = '../data/auto_QA_data/CSQA_result_question_type_count944k_orderlist.json'
QTYPE_DOC_RANGE = '../data/auto_QA_data/944k_rangeDict.json'
log = logging.getLogger("train")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    # # command line parameters
    # # -a=True means using adaptive reward to train the model. -a=False is using 0-1 reward.
    # sys.argv = ['train_maml_true_reward.py', '--cuda', '-l=../data/saves/rl_even_TR_batch8_1%/truereward_0.739_29.dat', '-n=maml_1%_batch8_att=0_test', '-s=5', '-a=0', '--att=0', '--lstm=1', '--fast-lr=0.1', '--meta-lr=1e-4', '--steps=5', '--batches=1', '--weak=1']
    sys.argv = ['train_maml_retriever_joint.py', '-l=../data/saves/rl_even_TR_batch8_1%/truereward_0.739_29.dat',
                '-n=maml_newdata2k_reptile_retriever_joint_test', '--cuda', '-s=5', '-a=0', '--att=0', '--lstm=1', '--fast-lr=1e-4',
                '--meta-lr=1e-4', '--steps=5', '--batches=1', '--weak=1', '--embed-grad', '--beta=0.1', '--supportsets=5', '--MonteCarlo', '-retrieverl=../data/saves/retriever/AdaBound_DocEmbed_QueryEmbed_epoch_140_4.306.dat']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-l", "--load", required=True, help="Load the pre-trained model whereby continue training the RL mode")
    parser.add_argument("-retrieverl", "--retrieverload", required=True,
                        help="Load the pre-trained model whereby continue training the retriever mode")
    # Number of decoding samples.
    parser.add_argument("-s", "--samples", type=int, default=4, help="Count of samples in prob mode")
    # Choose the function to compute reward (0-1 or adaptive reward).
    # If a = true, 1 or yes, the adaptive reward is used. Otherwise 0-1 reward is used.
    parser.add_argument("-a", "--adaptive", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help="0-1 or adaptive reward")
    parser.add_argument("--disable-skip", default=False, action='store_true', help="Disable skipping of samples with high argmax BLEU")
    # Choose the function to compute reward (0-1 or adaptive reward).
    # If a = true, 1 or yes, the adaptive reward is used. Otherwise 0-1 reward is used.
    parser.add_argument("--att", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using attention mechanism in seq2seq")
    parser.add_argument("--lstm", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Using LSTM mechanism in seq2seq")
    # The action='store_true' means once the parameter is appeared in the command line, such as '--first-order',
    # the action is marked as 'True';
    # If there is no value of the parameter, the value is assigned as 'False'.
    # Conversely, if action is 'store_false', if the parameter has a value, the parameter is viewed as 'False'.
    parser.add_argument('--first-order', action='store_true', help='use the first-order approximation of MAML')
    # If false, the embedding tensors in the model do not need to be trained.
    parser.add_argument('--embed-grad', action='store_false', help='fix embeddings when training')
    parser.add_argument('--docembed-grad', action='store_false', help='fix doc embeddings when training')
    # If query_embed is true, using the sum of word embedding to represent the questions.
    parser.add_argument('--query-embed', action='store_false', help='using the sum of word embedding to represent the questions')
    parser.add_argument('--fast-lr', type=float, default=0.0001,
                        help='learning rate for the 1-step gradient update of MAML')
    parser.add_argument('--meta-lr', type=float, default=0.0001,
                        help='learning rate for the meta optimization')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='learning rate for reptile')
    parser.add_argument('--steps', type=int, default=5, help='steps in inner loop of MAML')
    parser.add_argument('--batches', type=int, default=5, help='tasks of a batch in outer loop of MAML')
    # If weak is true, it means when searching for support set, the questions with same number of E/R/T nut different relation will be retrieved if the questions in this pattern is less than the number of steps.
    parser.add_argument('--supportsets', type=int, default=5, help='number of the support sets')
    parser.add_argument("--weak", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="using weak mode to search for support set")
    parser.add_argument('--retriever-random', action='store_true', help='randomly get support set for the retriever')
    parser.add_argument("--MonteCarlo", action='store_true', default=False,
                        help="using Monte Carlo algorithm for REINFORCE")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    log.info("Device info: %s", str(device))

    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    # # List of (question, {question information and answer}) pairs, the training pairs are in format of 1:1.
    phrase_pairs, emb_dict = data.load_data_MAML(QUESTION_PATH=TRAIN_QUESTION_ANSWER_PATH, DIC_PATH=DIC_PATH, max_tokens=MAX_TOKENS)
    log.info("Obtained %d phrase pairs with %d uniq words from %s.", len(phrase_pairs), len(emb_dict), TRAIN_QUESTION_ANSWER_PATH)
    phrase_pairs_944K = data.load_data_MAML(QUESTION_PATH=TRAIN_944K_QUESTION_ANSWER_PATH, max_tokens = MAX_TOKENS)
    log.info("Obtained %d phrase pairs from %s.", len(phrase_pairs_944K), TRAIN_944K_QUESTION_ANSWER_PATH)
    data.save_emb_dict(saves_path, emb_dict)
    end_token = emb_dict[data.END_TOKEN]
    # Transform token into index in dictionary.
    train_data = data.encode_phrase_pairs_RLTR(phrase_pairs, emb_dict)
    # # list of (seq1, [seq*]) pairs，把训练对做成1：N的形式；
    # train_data = data.group_train_data(train_data)
    train_data = data.group_train_data_RLTR(train_data)

    train_data_944K = data.encode_phrase_pairs_RLTR(phrase_pairs_944K, emb_dict)
    train_data_944K = data.group_train_data_RLTR_for_support(train_data_944K)

    dict944k = data.get944k(DICT_944K)
    log.info("Reading dict944k from %s is done. %d pairs in dict944k.", DICT_944K, len(dict944k))
    dict944k_weak = data.get944k(DICT_944K_WEAK)
    log.info("Reading dict944k_weak from %s is done. %d pairs in dict944k_weak", DICT_944K_WEAK, len(dict944k_weak))

    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    train_data, test_data = data.split_train_test(train_data, TRAIN_RATIO)
    log.info("Training data converted, got %d samples", len(train_data))
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))
    log.info("Batch size is %d", args.batches)
    log.info("Number of the support sets %d", args.supportsets)
    if (args.att):
        log.info("Using attention mechanism to train the SEQ2SEQ model...")
    else:
        log.info("Train the SEQ2SEQ model without attention mechanism...")
    if (args.lstm):
        log.info("Using LSTM mechanism to train the SEQ2SEQ model...")
    else:
        log.info("Using RNN mechanism to train the SEQ2SEQ model...")
    if (args.embed_grad):
        log.info("Word embedding in the model will be updated during the training...")
    else:
        log.info("Word embedding in the model will be fixed during the training...")
    if (args.docembed_grad):
        log.info("Document embedding in the retriever model will be updated during the training...")
    else:
        log.info("Document embedding in the retriever model will be fixed during the training...")
    if (args.query_embed):
        log.info("Using the sum of word embedding to represent the questions during the training...")
    else:
        log.info("Using the document_emb which is stored in the retriever model to represent the questions...")
    if args.MonteCarlo:
        log.info("Using Monte Carlo algorithm for Policy Gradient...")

    # Index -> word.
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}
    # PhraseModel.__init__() to establish a LSTM model.
    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE, LSTM_FLAG=args.lstm, ATT_FLAG=args.att, EMBED_FLAG=args.embed_grad).to(device)
    # Using cuda.
    net.cuda()
    log.info("Model: %s", net)

    # Load the pre-trained seq2seq model.
    net.load_state_dict(torch.load(args.load))
    # print("Pre-trained network params")
    # for name, param in net.named_parameters():
    #     print(name, param.shape)
    log.info("Model loaded from %s, continue training in MAML-Reptile mode...", args.load)
    if (args.adaptive):
        log.info("Using adaptive reward to train the REINFORCE model...")
    else:
        log.info("Using 0-1 sparse reward to train the REINFORCE model...")

    docID_dict, _ = data.get_docID_indices(data.get_ordered_docID_document(ORDERED_QID_QUESTION_DICT))
    # Index -> qid.
    rev_docID_dict = {id: doc for doc, id in docID_dict.items()}

    qtype_docs_range = data.load_json(QTYPE_DOC_RANGE)

    retriever_net = retriever_module.RetrieverModel(emb_size=50, dict_size=len(docID_dict), EMBED_FLAG=args.docembed_grad,
                         device=device).to(device)
    retriever_net.cuda()
    log.info("Retriever model: %s", retriever_net)
    retriever_net.load_state_dict(torch.load(args.retrieverload))
    log.info("Retriever model loaded from %s, continue training in RL mode...", args.retrieverload)

    writer = SummaryWriter(comment="-" + args.name)
    # BEGIN token
    beg_token = torch.LongTensor([emb_dict[data.BEGIN_TOKEN]]).to(device)
    beg_token = beg_token.cuda()

    metaLearner = metalearner.MetaLearner(net=net, retriever_net=retriever_net, device=device, beg_token=beg_token, end_token=end_token, adaptive=args.adaptive, samples=args.samples, train_data_support_944K=train_data_944K, rev_emb_dict=rev_emb_dict, first_order=args.first_order, fast_lr=args.fast_lr, meta_optimizer_lr=args.meta_lr, dial_shown=False, dict=dict944k, dict_weak=dict944k_weak, steps=args.steps, weak_flag=args.weak, query_embed=args.query_embed)
    log.info("Meta-learner: %d inner steps, %f inner learning rate, "
             "%d outer steps, %f outer learning rate, using weak mode:%s, retriever random model:%s"
             %(args.steps, args.fast_lr, args.batches, args.meta_lr, str(args.weak), str(args.retriever_random)))

    # TBMeanTracker (TensorBoard value tracker):
    # allows to batch fixed amount of historical values and write their mean into TB
    with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
        maml_batch_idx = 0
        maml_batch_count = 0
        retriever_batch_idx = 0
        retriever_batch_count = 0
        best_true_reward = None
        time_start = time.time()

        # Loop in epoches.
        for epoch in range(MAX_EPOCHES):
            dial_shown = False
            random.shuffle(train_data)
            total_samples = 0
            skipped_samples = 0
            true_reward_argmax = []
            true_reward_sample = []

            # Stage1
            for batch in data.iterate_batches(train_data, args.batches):
                # The dict stores the initial parameters in the modules.
                old_param_dict = metaLearner.get_net_named_parameter()
                # temp_param_dict = metaLearner.get_net_parameter()
                maml_batch_idx += 1
                # Each batch conduct one gradient upweight.
                maml_batch_count += 1

                # Batch is represented for a batch of tasks in MAML.
                # In each task, a minibatch of support set is established.
                meta_losses, running_vars, meta_total_samples, meta_skipped_samples, true_reward_argmax_batch, true_reward_sample_batch = metaLearner.reptile_sample(batch, old_param_dict=old_param_dict, dial_shown=dial_shown, epoch_count=epoch, batch_count=maml_batch_count, docID_dict=docID_dict, rev_docID_dict=rev_docID_dict, emb_dict=emb_dict, qtype_docs_range=qtype_docs_range, random=args.retriever_random, monte_carlo=args.MonteCarlo)
                total_samples += meta_total_samples
                skipped_samples += meta_skipped_samples
                true_reward_argmax.extend(true_reward_argmax_batch)
                true_reward_sample.extend(true_reward_sample_batch)
                metaLearner.reptile_meta_update(running_vars, old_param_dict, args.beta)
                metaLearner.net.zero_grad()
                temp_param_dict = metaLearner.get_net_parameter()
                dial_shown = True
                tb_tracker.track("meta_losses", (float)(meta_losses.cpu().detach().numpy()), maml_batch_idx)

            # After one epoch, get the average of the decode_chain_argmax bleus for samples in training dataset.
            maml_true_reward_armax = np.mean(true_reward_argmax)
            writer.add_scalar("maml_true_reward_armax", maml_true_reward_armax, maml_batch_idx)
            # After one epoch, get the average of the decode_chain_sampling bleus for samples in training dataset.
            writer.add_scalar("maml_true_reward_sample", np.mean(true_reward_sample), maml_batch_idx)
            writer.add_scalar("skipped_samples", skipped_samples / total_samples if total_samples != 0 else 0,
                              maml_batch_idx)
            log.info("MAML epoch %d, skipped_samples: %d, total_samples: %d", epoch, skipped_samples, total_samples)
            writer.add_scalar("epoch", maml_batch_idx, epoch)

            # The dict stores the initial parameters in the modules.
            stage1_old_param_dict = metaLearner.get_net_named_parameter()

            log.info("---------------------------")
            log.info("MAML epoch %d, Stage 1 training is over...", epoch)
            log.info("---------------------------")

            retriever_total_samples = 0
            retriever_skipped_samples = 0
            retriever_true_reward_argmax = []
            retriever_true_reward_sample = []

            # Stage 2
            for batch in data.iterate_batches(train_data, args.batches):
                # temp_param_dict = metaLearner.get_net_parameter()
                retriever_batch_idx += 1
                # Each batch conduct one gradient upweight.
                retriever_batch_count += 1

                # Batch is represented for a batch of tasks in MAML.
                # In each task, a minibatch of support set is established.
                metaLearner.retriever_optimizer.zero_grad()
                metaLearner.retriever_net.zero_grad()

                retriever_losses, retriever_true_reward_argmax_batch, retriever_true_reward_sample_batch, retriever_total_sample, retriever_skipped_sample = metaLearner.retriever_sample(batch, old_param_dict=stage1_old_param_dict, dial_shown=dial_shown, epoch_count=epoch, batch_count=retriever_batch_count, docID_dict=docID_dict, rev_docID_dict=rev_docID_dict, emb_dict=emb_dict, qtype_docs_range=qtype_docs_range, number_of_supportsets=args.supportsets, mc=args.MonteCarlo, device=device)
                retriever_total_samples += retriever_total_sample
                retriever_skipped_samples += retriever_skipped_sample
                retriever_true_reward_argmax.extend(retriever_true_reward_argmax_batch)
                retriever_true_reward_sample.extend(retriever_true_reward_sample_batch)

                # temp_param_dict = metaLearner.retriever_net.get_retriever_net_parameter()
                metaLearner.retriever_optimizer.zero_grad()
                metaLearner.retriever_net.zero_grad()
                retriever_losses.backward()
                metaLearner.retriever_optimizer.step()
                # temp_param_dict = metaLearner.retriever_net.get_retriever_net_parameter()
                dial_shown = True
                tb_tracker.track("retriever_losses", (float)(retriever_losses.cpu().detach().numpy()), retriever_batch_idx)

            # After one epoch, get the average of the decode_chain_argmax bleus for samples in training dataset.
            retriever_true_reward_argmax = np.mean(retriever_true_reward_argmax)
            retriever_true_reward_sample = np.mean(retriever_true_reward_sample)
            writer.add_scalar("retriever_true_reward_argmax", retriever_true_reward_argmax, retriever_batch_idx)
            # After one epoch, get the average of the decode_chain_sampling bleus for samples in training dataset.
            writer.add_scalar("retriever_true_reward_sample", np.mean(retriever_true_reward_sample), retriever_batch_idx)
            writer.add_scalar("skipped_samples", retriever_skipped_samples / retriever_total_samples if retriever_total_samples != 0 else 0,
                              retriever_batch_idx)
            log.info("Retriever epoch %d, retriever_skipped_samples: %d, retriever_total_samples: %d", epoch, retriever_skipped_samples, retriever_total_samples)
            writer.add_scalar("epoch", retriever_batch_idx, epoch)
            log.info("---------------------------")
            log.info("Retriever epoch %d, Stage 2 training is over...", epoch)
            log.info("---------------------------")

            argmax_reward_sum = 0.0
            argmax_reward_count = 0.0
            # test
            # p1 is one sentence, p2 is sentence list.
            for test_task in test_data:
                _, action_tokens = metaLearner.maml_retriever_sampleForTest(task=test_task, old_param_dict=stage1_old_param_dict, docID_dict=docID_dict, rev_docID_dict=rev_docID_dict, emb_dict=emb_dict, qtype_docs_range=qtype_docs_range, steps=args.steps, mc=args.MonteCarlo)
                # Using 0-1 reward to compute accuracy.
                argmax_reward_sum += float(utils.calc_True_Reward(action_tokens, test_task[1], False))
                # argmax_reward_sum += random.random()
                argmax_reward_count += 1
            true_reward_test = float(argmax_reward_sum) / float(argmax_reward_count)

            # # The parameters are stored after each epoch.
            if best_true_reward is None or best_true_reward < true_reward_test:
                best_true_reward = true_reward_test
                log.info("Best true reward updated: %.4f", true_reward_test)
                # Save the updated seq2seq parameters trained by RL.
                torch.save(net.state_dict(), os.path.join(saves_path, "net_truereward_%.3f_%02d.dat" % (true_reward_test, epoch)))
                torch.save(retriever_net.state_dict(), os.path.join(saves_path, "retriever_truereward_%.3f_%02d.dat" % (true_reward_test, epoch)))
            # # The parameters are stored after each epoch.
            torch.save(net.state_dict(), os.path.join(saves_path, "net_epoch_%03d_%.3f_%.3f.dat" % (epoch, float(maml_true_reward_armax), true_reward_test)))
            torch.save(retriever_net.state_dict(), os.path.join(saves_path, "retriever_epoch_%03d_%.3f_%.3f.dat" % (epoch, float(retriever_true_reward_argmax), float(true_reward_test))))
            log.info("---------------------------")
            log.info("Epoch %d, testing is over, test reward: %.3f", epoch, true_reward_test)
            log.info("---------------------------")

        time_end = time.time()
        log.info("Training time is %.3fs." % (time_end - time_start))
        print("Training time is %.3fs." % (time_end - time_start))
    writer.close()
