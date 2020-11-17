#!/usr/bin/env python3
import os
import random
import argparse
import logging
import numpy as np
import sys
from tensorboardX import SummaryWriter
import time
from libbots import data, model, utils, bert_model
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
import torch.optim as optim
import torch.nn.functional as F

SAVES_DIR = "../data/saves"

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MAX_EPOCHES = 200
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

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

def run_test(test_data, net, end_token, device="cuda", rev_emb_dict=None, tokenizer=None, max_tokens=None):
    bleu_sum = 0.0
    bleu_count = 0
    for p1, p2 in test_data:
        p_list = [(p1, p2)]
        input_ids, attention_masks = tokenizer_encode(tokenizer, p_list, rev_emb_dict, device, max_tokens)
        output, output_hidden_states = net.bert_encode(input_ids, attention_masks)
        context, enc = output_hidden_states, (output.unsqueeze(0), output.unsqueeze(0))
        input_seq = net.pack_input(p1, net.emb, device)
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

def tokenizer_encode(tokenizer, batch_text, rev_emb_dict, device, max_tokens):
    input_ids_list = []
    attention_mask_list = []
    for sample in batch_text:
        sample_tokens = [rev_emb_dict[x] for x in sample[0]]
        sample_utterence = ' '.join(sample_tokens).strip()
        encoding = tokenizer.encode_plus(
            sample_utterence,
            max_length=max_tokens,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
            truncation=True
        )
        input_ids_list.append(encoding['input_ids'].to(device))
        attention_mask_list.append(encoding['attention_mask'].to(device))
    input_ids_s = torch.cat(input_ids_list, dim=0)
    attention_mask_s = torch.cat(attention_mask_list, dim=0)
    return input_ids_s, attention_mask_s

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    # command line parameters
    sys.argv = ['train_crossent.py',
                '--cuda',
                '-d=csqa',
                '--n=crossent_even_1%_att=0_withINT_BERT_test',
                '--att=0',
                '--lstm=1',
                '--int',
                '-w2v=300',
                '--bert',
                '--fix_bert',
                '--trans_bert']

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
    # If false, the embedding tensors in to(device)the model do not need to be trained.
    parser.add_argument('--embed-grad', action='store_false', help='the embeddings would not be optimized when training')
    parser.add_argument('--int', action='store_true', help='training model with INT mask information')
    parser.add_argument('--bert', action='store_true', help='training model with BERT')
    parser.add_argument('--fix_bert', action='store_true', help='training model with fixed BERT')
    parser.add_argument("-w2v", "--word_dimension", type=int, default=50, help="The dimension of the word embeddings")
    parser.add_argument('--trans_bert', action='store_true',
                        help='transform the pooled hidden state into the initial input token of the LSTM decoder')
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

    if args.bert:
        log.info("Training model with BERT...")
    if args.trans_bert:
        log.info("Transform the pooled hidden state into the initial input token of the LSTM decoder...")

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

    net = bert_model.CqaBertModel(pre_trained_model_name=PRE_TRAINED_MODEL_NAME, fix_flag=args.fix_bert, emb_size=args.word_dimension, dict_size=len(emb_dict), hid_size=model.HIDDEN_STATE_SIZE, LSTM_FLAG=args.lstm,BERT_TO_EMBEDDING_FLAG=args.trans_bert).to(device)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=PRE_TRAINED_MODEL_NAME)

    # 转到cuda
    net.cuda()
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)

    # optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    optimiser = AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=2e-5, correct_bias=False, eps=1e-8)
    best_bleu = None

    time_start = time.time()

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_data) // BATCH_SIZE * MAX_EPOCHES

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimiser,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

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

            max_tokens = (MAX_TOKENS_INT + 40) if args.int else (MAX_TOKENS + 40)
            # TODO: Transform IDs to tokens might cause information loss.
            input_ids_s, attention_mask_s = tokenizer_encode(tokenizer, batch, rev_emb_dict, device, max_tokens)
            output, output_hidden_states = net.bert_encode(input_ids_s, attention_mask_s)

            # torch.unsqueeze(index) to add a dimension for the tensor.
            # torch.unsqueeze(input, dim) → Tensor:
            # Returns a new tensor with a dimension of size one inserted at the specified position.
            # The returned tensor shares the same underlying data with this tensor.
            # A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used.
            # Negative dim will correspond to unsqueeze() applied at dim = dim + input.dim() + 1.
            # >>> x = torch.tensor([1, 2, 3, 4])
            # >>> torch.unsqueeze(x, 0)
            # tensor([[ 1,  2,  3,  4]])
            # >>> torch.unsqueeze(x, 1)
            # tensor([[ 1],
            #         [ 2],
            #         [ 3],
            #         [ 4]])

            if args.trans_bert:
                initial_input_for_decoder = net.bert_to_embedding(output)
                context, enc = output_hidden_states, net.bert_decode_one(initial_input_for_decoder)
            else:
                context, enc = output_hidden_states, (output.unsqueeze(0), output.unsqueeze(0))

            # context, enc = net.encode_context(input_seq)

            net_results = []
            net_targets = []
            for idx, out_seq in enumerate(out_seq_list):
                ref_indices = out_idx[idx][1:]
                # TODO: how to use the output of the BERT as the input of the LSTM decoder?
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

            # # Clip the norm of the gradients to 1.0.
            # # This is to help prevent the "exploding gradients" problem.
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimiser.step()

            # Update the learning rate.
            scheduler.step()

            losses.append(loss_v.item())
        bleu = bleu_sum / bleu_count
        bleu_test = run_test(test_data, net, end_token, device, rev_emb_dict, tokenizer, max_tokens)
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
