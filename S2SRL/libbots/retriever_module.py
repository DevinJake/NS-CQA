import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class RetrieverModel(nn.Module):
    def __init__(self, emb_size, dict_size, EMBED_FLAG=False, hid1_size = 300, hid2_size = 200, output_size = 128, device='cpu'):
        # Call __init__ function of PhraseModel's parent class (nn.Module).
        super(RetrieverModel, self).__init__()
        # With :attr:`padding_idx` set, the embedding vector at
        #         :attr:`padding_idx` is initialized to all zeros. However, notice that this
        #         vector can be modified afterwards, e.g., using a customized
        #         initialization method, and thus changing the vector used to pad the
        #         output. The gradient for this vector from :class:`~torch.nn.Embedding`
        #         is always zero.
        self.document_emb = nn.Embedding(num_embeddings=dict_size+1, embedding_dim=emb_size, padding_idx=dict_size)
        if not EMBED_FLAG:
            for p in self.parameters():
                p.requires_grad = False
        # torch.nn.init.uniform_(tensor, a=0.0, b=1.0)[SOURCE]
        # Fills the input Tensor with values drawn from the uniform distribution \mathcal{U}(a, b)U(a,b) .
        # Parameters
        # tensor – an n-dimensional torch.Tensor
        # a – the lower bound of the uniform distribution
        # b – the upper bound of the uniform distribution
        # Examples
        # >>> w = torch.empty(3, 5)
        # >>> nn.init.uniform_(w)
        # Using uniform distribution of [-sqrt(6.0 / (fanin + fanout))] to initialize the model.
        self.hid_layer1 = nn.Linear(emb_size, hid1_size)
        limit = math.sqrt(6.0 / (emb_size + hid1_size))
        nn.init.uniform_(self.hid_layer1.weight, -limit, limit)
        nn.init.uniform_(self.hid_layer1.bias, -limit, limit)
        # Or using the xavier_uniform to initialize the weight.
        # Also, xavier_uniform will fail on bias, as it has less than 2 dimensions,
        # so that fan_in and fan_out cannot be computed.
        # nn.init.xavier_uniform_(self.hid_layer1.weight)
        self.hid_layer2 = nn.Linear(hid1_size, hid2_size)
        limit = math.sqrt(6.0 / (hid1_size + hid2_size))
        nn.init.uniform_(self.hid_layer2.weight, -limit, limit)
        nn.init.uniform_(self.hid_layer2.bias, -limit, limit)
        # nn.init.xavier_uniform_(self.hid_layer2.weight)

        self.output_layer = nn.Linear(hid2_size, output_size)
        limit = math.sqrt(6.0 / (hid2_size + output_size))
        nn.init.uniform_(self.output_layer.weight, -limit, limit)
        nn.init.uniform_(self.output_layer.bias, -limit, limit)
        # nn.init.xavier_uniform_(self.output_layer.weight)
        self.device = device

    # todo activation function? negative sampling?
    def forward(self, query_tensor, range):
        documents = self.pack_input(range)
        query_tensor = torch.tanh(self.hid_layer1(query_tensor))
        query_tensor = torch.tanh(self.hid_layer2(query_tensor))
        query_tensor = torch.tanh(self.output_layer(query_tensor))
        documents = torch.tanh(self.hid_layer1(documents))
        documents = torch.tanh(self.hid_layer2(documents))
        documents = torch.tanh(self.output_layer(documents))
        cosine_output = torch.cosine_similarity(query_tensor, documents, dim=1)
        logsoftmax_output = F.log_softmax(cosine_output, dim=0)
        softmax_output = F.softmax(cosine_output, dim=0)
        return logsoftmax_output, softmax_output, cosine_output

    def get_retriever_net_parameter(self):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        params = self.named_parameters()
        param_dict = dict()
        for name, param in params:
            param_dict[name] = param.to(self.device).clone().detach()
        return param_dict

    def pack_input(self, indices):
        dict_size = self.document_emb.weight.shape[0]-1
        # Process one document.
        if not isinstance(indices, tuple):
            index = indices
            if index >= dict_size or index < 0:
                index = dict_size
            input_v = torch.LongTensor([index]).to(self.device)
            input_v = input_v.cuda()
            r = self.document_emb(input_v)
            return r
        # Process a range of documents.
        else:
            list = [dict_size if (i >= dict_size or i < 0) else i for i in range(indices[0], indices[1])]
            input_v = torch.LongTensor(list).to(self.device)
            input_v = input_v.cuda()
            r = self.document_emb(input_v)
            return r

    @classmethod
    def calculate_rank(self, vector):
        rank = 1
        order_list = {}
        if isinstance(vector, list):
            for value in sorted(vector, reverse=True):
                if value not in order_list:
                    order_list[value] = rank
                rank += 1
            order = [order_list[i] for i in vector]
            return order
        else:
            return []