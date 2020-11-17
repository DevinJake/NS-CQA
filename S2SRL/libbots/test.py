import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import nltk
import math
from random import randrange
KEY_ATTN_SCORE = 'attention_score'

# (5, 3, 256) & (5, 5, 256);
def attention(context, output):
    batch_size = output.size(0)
    hidden_size = output.size(2)
    input_size = context.size(1)
    linear_out = nn.Linear(hidden_size * 2, hidden_size)
    # (5, 256, 3)
    context_trans = context.transpose(1, 2)
    # (5, 5, 3)
    attn = torch.bmm(output, context.transpose(1, 2))
    # Returns a new tensor with the same data as the self tensor but of a different shape.
    # [4, 4] -> view(-1, 8) -> [2, 8]
    # Here .view operation is used to squeeze outputs in each batches into one batch.
    # (25, 3)
    attn_view = attn.view(-1, input_size)
    # (25, 3)
    attn1 = F.softmax(attn_view, dim=1)
    # Here .view operation is used to un-squeeze outputs into different batches.
    # (5, 5, 3)
    attn1 = attn1.view(batch_size, -1, input_size)
    attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
    # (5, 5, 256)
    mix = torch.bmm(attn, context)
    print(mix.size())
    # (5, 5, 512)
    combined = torch.cat((mix, output), dim=2)
    # (25, 512)
    combined1 = combined.view(-1, 2 * hidden_size)
    # (25, 256)
    combined1 = linear_out(combined1)
    # (25, 256)
    combined1 = torch.tanh(combined1)
    # (5, 5, 256)
    combined1 = combined1.view(batch_size, -1, hidden_size)
    output = torch.tanh(linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
    return output, attn

# (5, 5, 256);
def forward_step(output):
    output_size = 10000
    batch_size = output.size(0)
    hidden_size = output.size(2)
    out = nn.Linear(hidden_size, output_size)
    function = F.log_softmax
    # Returns a contiguous tensor containing the same data as self tensor.
    # If self tensor is contiguous, this function returns the self tensor.
    # (5, 5, 256)
    output1 = output.contiguous()
    # (25, 256)
    output1 = output1.view(-1, hidden_size)
    # (25, 10000)
    output1 = out(output1)
    output1 = function(output1, dim=1)
    # # (5, 10000, 5) WHY?
    # output1 = output1.view(batch_size, output_size, -1)
    # predicted_softmax  = function(out(output.contiguous().view(-1, hidden_size)), dim=1).view(batch_size, output_size, -1)
    # (5, 5, 10000)
    output1 = output1.view(batch_size, -1, output_size)
    predicted_softmax = function(out(output.contiguous().view(-1, hidden_size)), dim=1).view(batch_size, -1, output_size)
    print(predicted_softmax.size())
    return predicted_softmax

# To compute the Jaccard score between action sequences s1 and s2.
def jaccard_similarity(s1, s2):
    if s1 is None or len(s1) == 0:
        return 0.0
    elif s2 is None or len(s2) == 0:
        return 0.0
    else:
        jd = nltk.jaccard_distance(set(s1), set(s2))
        return 1.0 - jd

def levenshtein_similarity(source, target):
    """
    To compute the edit-distance between source and target.
    If source is list, regard each element in the list as a character.
    :param list1
    :param list2
    :return:
    """
    if source is None or len(source) == 0:
        return 0.0
    elif target is None or len(target) == 0:
        return 0.0
    elif type(source) != type(target):
        return 0.0
    matrix = [[i + j for j in range(len(target) + 1)] for i in range(len(source) + 1)]
    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if (source[i - 1] == target[j - 1]):
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    distance = float(matrix[len(source)][len(target)])
    length = float(len(source) if len(source) >= len(target) else len(target))
    return 1.0 - distance / length

# To judge whether s1 and s2 are same lists or not.
def duplicate(s1,s2):
    compare = lambda a,b: len(a)==len(b) and len(a)==sum([1 for i,j in zip(a,b) if i==j])
    return compare(s1, s2)

if __name__ == "__main__":
    # s1 = ['A2', '(', 'TYPE1', '-', 'RELATION1', 'TYPE2', ')', 'A6', '(', 'ENTITY1', ')']
    # s2 = ['A2', '(', 'TYPE2', '-', 'RELATION1', 'TYPE1', ')', 'A6', '(', 'ENTITY1', ')']
    # print(jaccard_similarity(s1, s2))
    # print(levenshtein_similarity(s2, s1))

    # print(s1)
    # random_index = randrange(0, len(s1))
    # print(random_index)
    # s1.pop(random_index)
    # print(s1)
    #
    # sum_list = [1, 2, 3]
    # print(sum(sum_list))

    # print(duplicate(s1, s2))

    # ETA = 0.08
    # LAMBDA_0 = 0.1
    # epoch = 0
    # print(math.pow(1.0 + ETA, float(epoch) + 1.0) * LAMBDA_0)
    # print(min(1.0, math.pow(1.0 + ETA, float(epoch) + 1.0) * LAMBDA_0))

    rnn = nn.LSTM(300, 128, num_layers=1, batch_first=True)
    input = torch.randn(32, 1, 300)
    h0 = torch.randn(1, 3, 20)
    c0 = torch.randn(1, 3, 20)
    output, (hn, cn) = rnn(input)
    print((hn, cn))

    '''max_length = 50
    batch_size = 5
    eos_id = 2
    ret_dict = dict()
    ret_dict[KEY_ATTN_SCORE] = list()
    context = torch.randn(5, 3, 256)
    print(context.size())
    input = torch.randn(5, 5, 256)
    print(input.size())
    # (5, 5, 256) & (5, 5, 3)
    output, attn = attention(context, input)
    # (5, 5, 10000)
    decoder_output = forward_step(output)
    decoder_outputs = []
    sequence_symbols = []
    # 1 * batch_size list in which the value of each element is max_length.
    lengths = np.array([max_length] * batch_size)
    print(lengths)
    # step_output: (5, 1, 10000), step_attn: (5, 1, 3)
    def decode(step, step_output, step_attn):
        decoder_outputs.append(step_output)
        ret_dict[KEY_ATTN_SCORE].append(step_attn)
        symbols1 = decoder_outputs[-1]
        # Returns the k largest elements of the given input tensor along a given dimension.
        # If dim is not given, the last dimension of the input is chosen.
        # The indices of topk elements will also be returned.
        # k largest elements is stored in returned result as the first tensor and the second tensor is about the indices.
        symbols1 = symbols1.topk(1)
        symbols1 = symbols1[1]
        symbols = decoder_outputs[-1].topk(1)[1]
        sequence_symbols.append(symbols)
        # torch.eq(input, other, out=None) → Tensor:
        # Computes element-wise equality, the second argument can be a number
        # or a tensor whose shape is broadcastable with the first argument.
        # Returns: A torch.BoolTensor containing a True at each location where comparison is true.
        # To tell whether an eos symbol is predicted.
        eos_batches = symbols.data.eq(eos_id)
        if eos_batches.dim() > 0:
            # From tensor to list.
            eos_batches = eos_batches.cpu().view(-1).numpy()
            # Compute element-wise boolean value in a list.
            update_idx = ((lengths > step) & eos_batches) != 0
            # In a boolean list update_idx, if one element is true,
            # then assign the corresponding element in lengths as len(sequence_symbols).
            # Or the element in lengths remains unchanged.
            # Then in next iteration, for such element, since the length of it is updated and equals to step,
            # the length would not be updated anymore, which means the output of this sample is ended.
            lengths[update_idx] = len(sequence_symbols)
        return symbols
    for di in range(decoder_output.size(1)):
        # Get the di-th output of all samples in a batch.
        step_output = decoder_output[:, di, :]
        # attn: (5, 5, 3)
        if attn is not None:
            step_attn = attn[:, di, :]
        else:
            step_attn = None
        decode(di, step_output, step_attn)

    # context1 = torch.randn(25, 3)
    # context1 = F.softmax(context1, dim=1)
    # context2 = context1.view(5, -1, 3)
    # print(context2.size())
    # for di in range(context2.size(1)):
    #     step_output = context2[:, di, :]
    #     print(step_output)

    dict_temp = {'name': 1}
    print(list(dict_temp.keys())[0])
    print(len(dict_temp))

    def MyFn(s,N):
        return abs(s-N)
    strs = [1.1, 2.0, 2.4, 4]
    N=2
    print(sorted(strs, key=lambda x:MyFn(x,N)))

    strs = [1.1, 2.0, 2.4, 4]
    strs1 = [5,6]
    strs.extend(strs1)
    print(strs)

    a = np.array([[1, 2], [3, 4]])
    print(float(np.mean(a)))

    lin = nn.Linear(1, 1)
    w = lin.weight

    lin(torch.randn(1, 1)).backward()

    print('lin.weight.grad: ' + str(lin.weight.grad))
    print('w.grad: ' + str(w.grad))
    print(id(w) == id(lin.weight))


    def basic_fun(x):
        return 3 * (x * x)

    def get_grad(inp, grad_var):
        A = basic_fun(inp)
        A.backward()
        return grad_var.grad

    x = torch.tensor([1.0], requires_grad=True)
    xx = x.clone()

    # Grad wrt x will work
    # print(x.creator is None)  # is it a leaf? Yes
    # grad = 6;
    print(get_grad(x, x))
    # grad = 12;
    # Since xx is cloned by x, so x is the leaf node.
    # When grad of xx is computed by backward(), the gradients propagating to the cloned tensor will propagate to the original tensor, so the grad of x is accumlated by the grad of xx, i.e., 6.
    print(get_grad(xx, x))

    # Grad wrt xx won't work
    # print(xx.creator is None)  # is it a leaf? No
    # When using backward() to calculate the gradients automatically, the un-leaf node will not hold the gradients.
    # grad = None;
    print(get_grad(xx, xx))
    # grad = None;
    print(get_grad(x, xx))'''
