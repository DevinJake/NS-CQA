import numpy as np
import operator
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from collections import OrderedDict
from . import utils
from . import attention
from . import beam_search_node
from queue import PriorityQueue

HIDDEN_STATE_SIZE = 128
EMBEDDING_DIM = 50


# nn.Module: Base class for all neural network modules.
# Your models should also subclass this class.
class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size, LSTM_FLAG, ATT_FLAG, EMBED_FLAG=True):
        # Call __init__ function of PhraseModel's parent class (nn.Module).
        super(PhraseModel, self).__init__()

        # self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeding_dim)
        # num_embeddings = vocab_size
        # embedding_dim = embeding_dim
        # If no pre-trained embeddings is designated, the random vectors will be initialized.
        self.emb = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        if not EMBED_FLAG:
            for p in self.parameters():
                p.requires_grad = False

        # # BiLSTM
        # self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
        #                        num_layers=1, batch_first=True, bidirectional=True)
        # self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
        #                                               num_layers=2, batch_first=True)

        # LSTM
        # Inputs of LSTM are: input, (h_0, c_0).
        # In which input is (seq_len, batch, input_size): tensor containing the features of the input sequence.
        # (h_0, c_0) is the initial hidden state and cell state for each element in the batch.
        # Outputs of LSTM are: output, (h_n, c_n).
        # In which output is (seq_len, batch, num_directions * hidden_size):
        # tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
        # (h_n, c_n) is tensor containing the hidden state and cell state for t = seq_len in the batch.
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hid_size, dict_size)
        )
        self.lstm_flag = LSTM_FLAG
        self.attention_flag = ATT_FLAG
        if self.attention_flag:
            self.attention = attention.Attention(hid_size)
            print('Build attention layer.')

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            # print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            # print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


    # Using the parameters to insert into network and compute output.
    def insert_new_parameter(self, state_dict, strict):
            self.load_state_dict(state_dict, strict)

    # Using the parameters to insert into network and compute output.
    def insert_new_parameter_to_layers(self, old_param_dict):
        for (name, param) in self.named_parameters():
            if param.requires_grad:
                param.data = old_param_dict[name].clone().detach()

    # Return all the parameters that have grads.
    def grad_parameters(self, recurse=True):
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            # >>> for param in model.parameters():
            # >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad:
                yield param

    # Return all the named_parameters that have grads.
    def grad_named_parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad:
                yield (name, param)

    # hidden state;
    # return hid: (h_n, c_n) is tensor containing the hidden state and cell state for t = seq_len.
    def encode(self, x):
        _, hid = self.encoder(x)
        return hid

    # Get each time step's hidden state for encoder;
    # return outputs: output, (h_n, c_n) for LSTM;
    # context is (seq_len, batch, num_directions * hidden_size):
    # tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
    # hid is (h_n, c_n), which is tensor containing the hidden state and cell state for t = seq_len.
    def encode_context(self, x):
        # Inputs of LSTM: input, (h_0, c_0)
        #         - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
        #           of the input sequence.
        #           The input can also be a packed variable length sequence.
        #           See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
        #           :func:`torch.nn.utils.rnn.pack_sequence` for details.
        #         - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #           containing the initial hidden state for each element in the batch.
        #           If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        #         - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #           containing the initial cell state for each element in the batch.
        #           If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.
        packed_context, hid = self.encoder(x)
        # It is an inverse operation to :func:`pack_padded_sequence`.
        # Unpack your output if required.
        unpack_context, input_sizes = rnn_utils.pad_packed_sequence(packed_context, batch_first=True)
        return unpack_context, hid

    def get_encoded_item(self, encoded, index):
        # For RNN
        if not self.lstm_flag:
           return encoded[:, index:index+1]
        # For LSTM
        if self.lstm_flag:
            return encoded[0][:, index:index+1].contiguous(), \
                   encoded[1][:, index:index+1].contiguous()

    def decode_teacher(self, hid, input_seq, context):
        # Method assumes batch of size=1
        out, _ = self.decoder(input_seq, hid)
        if self.attention_flag:
            out, attn = self.attention(out, context)
        out = self.output(out.data)
        return out

    def decode_one(self, hid, input_x, context):
        # Example for unsqueeze:
        #             >>> x = torch.tensor([1, 2, 3, 4])
        #             >>> torch.unsqueeze(x, 0)
        #             tensor([[ 1,  2,  3,  4]])
        #             >>> torch.unsqueeze(x, 1)
        #             tensor([[ 1],
        #                     [ 2],
        #                     [ 3],
        #                     [ 4]])
        out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
        if (self.attention_flag):
            out, attn = self.attention(out, context)
        # Self.output(out) using nn.Linear(hid_size, dict_size) to transform hidden states into logits over output vocab.
        out = self.output(out)
        # squeeze: Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.
        return out.squeeze(dim=0), new_hid

    def decode_chain_argmax(self, hid, begin_emb, seq_len, context, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again. Act greedily
        """
        res_logits = []
        res_tokens = []
        # First cur_emb is the embedding of '#BEG'.
        cur_emb = begin_emb

        # At first using the '#BEG' as first input token and hidden states from encoder as initial hidden state to predict the first output token and first decoder hidden state.
        # Then predict the output token by using last step's output token as current step's input and last step's decoder hidden state.
        for _ in range(seq_len):
            # The out_logits is the distribution over whole output vocabulary.
            # The hid is new hidden state generated from current time step.
            out_logits, hid = self.decode_one(hid, cur_emb, context)
            # After torch.max operation, the result is a list.
            # First element is the largest logit value in dimension-1 (each row), the second value is the index of the largest logit value.
            # >>> a = torch.randn(4, 4)
            # >>> a
            # tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
            #         [ 1.1949, -1.1127, -2.2379, -0.6702],
            #         [ 1.5717, -0.9207,  0.1297, -1.8768],
            #         [-0.6172,  1.0036, -0.6060, -0.2432]])
            # >>> torch.max(a, 1)
            # (tensor([ 0.8475,  1.1949,  1.5717,  1.0036]), tensor([ 3,  0,  0,  1]))
            out_token_v = torch.max(out_logits, dim=1)[1]
            # Transform tensorflow to array and return array[0];
            out_token = out_token_v.data.cpu().numpy()[0]
            # Using current output token's embedding.
            cur_emb = self.emb(out_token_v)

            # The list of out_logits list.
            res_logits.append(out_logits)
            # The list of output tokens.
            res_tokens.append(out_token)
            # When the EOS is predicted the prediction is ended.
            if stop_at_token is not None and out_token == stop_at_token:
                break
        # torch.cat(tensors, dim=0, out=None) → Tensor
        # Concatenates the given sequence of seq tensors in the given dimension.
        # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        # >>> x = torch.randn(2, 3)
        # >>> x
        # tensor([[ 0.6580, -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497]])
        # >>> torch.cat((x, x, x), 0)
        # tensor([[ 0.6580, -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497],
        #         [ 0.6580, -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497],
        #         [ 0.6580, -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497]])
        # >>> torch.cat((x, x, x), 1)
        # tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
        #          -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
        #          -0.5790,  0.1497]])
        # Concatenate follow rows.
        return torch.cat(res_logits), res_tokens

    def decode_chain_sampling(self, hid, begin_emb, seq_len, context, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again.
        Act according to probabilities
        """
        res_logits = []
        res_actions = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb, context)
            # Using softmax to transform logits to probabilities.
            out_probs_v = F.softmax(out_logits, dim=1)
            out_probs = out_probs_v.data.cpu().numpy()[0]
            # np.random.choice(out_probs.shape[0], p=out_probs):
            # choose one index from out_probs.shape[0] by the probabilities associated with each entry as out_probs.
            action = int(np.random.choice(out_probs.shape[0], p=out_probs))
            # Transform action to tensor and cast it to the device where begin_emb is in.
            action_v = torch.LongTensor([action]).to(begin_emb.device)
            action_v = action_v.cuda()
            # Get the embedding of the sampled output token.
            cur_emb = self.emb(action_v)

            res_logits.append(out_logits)
            res_actions.append(action)
            if stop_at_token is not None and action == stop_at_token:
                break
        return torch.cat(res_logits), res_actions

    def beam_decode(self, hid, seq_len, context, start_token, stop_at_token = None, beam_width = 10, topk = 5):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))
        # Start with the start of the sentence token: torch.LongTensor([[SOS_token]], device=device)
        # starting node -  hidden vector, previous node, word id, logp, length, logits
        node = beam_search_node.BeamSearchNode(hid, None, start_token, 0, 1, None)
        nodes = PriorityQueue()

        # start the queue
        # The smaller the value is, the higher the priority of the node is.
        # The unique ID of a node is used to avoid conflict between elements in the heap.
        nodes.put((-node.eval(), id(node), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000:
                break

            # fetch the best node
            score, _, n = nodes.get()

            action_v = n.wordid
            # Get the embedding of the sampled output token.
            decoder_input = self.emb(action_v)
            decoder_hidden = n.h

            # tensor.item(): if only one element is in the tensor, tensor.item() will return the value of the element.
            if n.wordid.item() == stop_at_token or n.leng == seq_len and n.prevNode != None:
                endnodes.append((score, id(n), n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            # decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            out_logits, decoder_hidden = self.decode_one(decoder_hidden, decoder_input, context)
            decoder_output = F.log_softmax(out_logits, dim=1)
            # out_probs = out_probs_v.data.cpu().numpy()[0]

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(-1)
                log_p = log_prob[0][new_k].item()

                # hidden vector, previous node, word id, logp, length
                node = beam_search_node.BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1, out_logits)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, id(nn), nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        all_res_logits = []
        # The sorted() function sorts the elements of a given iterable
        # in a specific order (either ascending or descending).
        # The syntax of sorted() is: sorted(iterable, key=None, reverse=False)
        # This is a great way to sort a list of tuples on the second key:
        # a = [ (1,'z'), (2, 'x'), (3, 'y') ]
        # a.sort(key=operator.itemgetter(1))
        # Or using lambda: a.sort(key=lambda x: x[1])
        for score, _, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            res_logits = []

            # back trace
            while n.prevNode != None:
                utterance.append(n.wordid.item())
                res_logits.append(n.logits)
                n = n.prevNode

            # [::-1]: Reverse.
            utterance = utterance[::-1]
            res_logits = res_logits[::-1]
            utterances.append(utterance)
            all_res_logits.append(torch.cat(res_logits))

        return all_res_logits, utterances

    def pack_batch_no_out(self, batch, embeddings, device="cpu"):
        # Asserting statements is a convenient way to insert debugging assertions into a program.
        # To guarantee that the batch is a list.
        assert isinstance(batch, list)
        # The format of batch is a list of tuple: ((tuple),[[list of token ID list]])
        # A lambda function is a small anonymous function, the example is as following.
        # x = lambda a, b: a * b
        # print(x(5, 6))
        # Sort descending (CuDNN requirements) batch中第一个元素为最长的句子；
        batch.sort(key=lambda s: len(s[0]), reverse=True)
        # input_idx：一个batch的输入句子的tokens对应的ID矩阵；Each row is corresponding to one input sentence.
        # output_idx：一个batch的输出句子的tokens对应的ID矩阵；Each row is corresponding to a list of several output sentences.
        # zip wants a bunch of arguments to zip together, but what you have is a single argument (a list, whose elements are also lists).
        # The * in a function call "unpacks" a list (or other iterable), making each of its elements a separate argument.
        # For list p = [[1,2,3],[4,5,6]];
        # So without the *, you're doing zip( [[1,2,3],[4,5,6]] ). With the *, you're doing zip([1,2,3], [4,5,6]) = [(1, 4), (2, 5), (3, 6)].
        input_idx, output_idx = zip(*batch)
        # create padded matrix of inputs
        # map() function returns a list of the results after applying the given function to each item of a given iterable (list, tuple etc.)
        # For example:
        # numbers = (1, 2, 3, 4)
        # result = map(lambda x: x + x, numbers)
        # print(list(result))
        # Output: {2, 4, 6, 8}
        # 建立长度词典，为batch中每一个元素的长度；
        lens = list(map(len, input_idx))
        # 以最长的句子来建立batch*最长句子长度的全0矩阵；
        input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
        # 将batch中每个句子的tokens对应的ID向量填入全0矩阵完成padding；
        # idx：index，x：token ID 组成的向量；
        for idx, x in enumerate(input_idx):
            input_mat[idx, :len(x)] = x
        # 将padding后的矩阵转换为tensor matrix；
        input_v = torch.tensor(input_mat).to(device)
        input_v = input_v.cuda()
        # 封装成PackedSequence类型的对象；
        # The padded sequence is the transposed matrix which is ``B x T x *``,
        # where `T` is the length of the longest sequence and `B` is the batch size.
        # Following the matrix is the list of lengths of each sequence in the batch (also in transposed format).
        # For instance:
        # [ a b c c d d d ]
        # [ a b c d ]
        # [ a b c ]
        # [ a b ]
        # could be transformed into [a,a,a,a,b,b,b,b,c,c,c,c,d,d,d,d] with batch size [4,4,3,2,1,1,1].
        input_seq = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)
        input_seq = input_seq.cuda()
        r = embeddings(input_seq.data)
        # lookup embeddings；embeddings为模型已经建立的词向量矩阵；
        # r: the [B x T x dimension] matrix of the embeddings of the occurred words in input sequence.
        # The order is followed by the order in input_seq.
        # Which is transforming [a,a,a,a,b,b,b,b,c,c,c,c,d,d,d,d] into [embedding(a), embedding(a), ..., embedding(d), embedding(d)]
        r = r.cuda()
        # 加入了词嵌入的input_seq；
        # For instance, given data  ``abc`` and `x`
        #         the :class:`PackedSequence` would contain data ``axbc`` with ``batch_sizes=[2,1,1]``.
        # emb_input_seq is [B x T x dimension] matrix of the embeddings of the occurred words in input sequence with the batch size.
        # For instance, emb_input_seq is the padded data: [embedding(a), embedding(a), ..., embedding(d), embedding(d)] with batch size [4,4,3,2,1,1,1].
        emb_input_seq = rnn_utils.PackedSequence(r, input_seq.batch_sizes)
        emb_input_seq = emb_input_seq.cuda()
        return emb_input_seq, input_idx, output_idx

    def pack_input(self, input_data, embeddings, device="cpu"):
        input_v = torch.LongTensor([input_data]).to(device)
        input_v = input_v.cuda()
        r = embeddings(input_v)
        return rnn_utils.pack_padded_sequence(r, [len(input_data)], batch_first=True)


    def pack_batch(self, batch, embeddings, device="cpu"):
        emb_input_seq, input_idx, output_idx = self.pack_batch_no_out(batch, embeddings, device)

        # prepare output sequences, with end token stripped
        output_seq_list = []
        for out in output_idx:
            output_seq_list.append(self.pack_input(out[:-1], embeddings, device))
        return emb_input_seq, output_seq_list, input_idx, output_idx


    def seq_bleu(self, model_out, ref_seq):
        model_seq = torch.max(model_out.data, dim=1)[1]
        model_seq = model_seq.cpu().numpy()
        return utils.calc_bleu(model_seq, ref_seq)
