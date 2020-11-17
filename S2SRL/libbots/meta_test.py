import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.optim as optim
from torch.autograd import Variable
from numpy.random import choice
from torch.utils.data.sampler import WeightedRandomSampler

"""
This solution seems to work so far. 
The idea is to first remove all Parameters in the Module, and replace them by tensors under object attributes with the same name. 
Here is what I used:
"""
def flip_parameters_to_tensors(module):
    attr = []
    while bool(module._parameters):
        attr.append( module._parameters.popitem() )
    setattr(module, 'registered_parameters_name', [])

    for i in attr:
        setattr(module, i[0], torch.zeros(i[1].shape,requires_grad=True))
        module.registered_parameters_name.append(i[0])

    module_name = [k for k,v in module._modules.items()]

    for name in module_name:
        flip_parameters_to_tensors(module._modules[name])

"""
Then, we can used the saved list of previously active attributes to assign the tensors.
This way, the flattened vector is assigned to all tensors of the NN for evaluation. 
The backward() gets back to the parameter vector outside the NN.
"""
def set_all_parameters(module, theta):
    count = 0

    for name in module.registered_parameters_name:
        a = count
        b = a + getattr(module, name).numel()
        t = torch.reshape(theta[0,a:b], getattr(module, name).shape)
        setattr(module, name, t)

        count += getattr(module, name).numel()

    module_name = [k for k,v in module._modules.items()]
    for name in module_name:
        count += set_all_parameters(module._modules[name], theta)
    return count

def gradient_test():
    # 第一个卷积层，我们可以看到它的权值是随机初始化的；
    w = nn.Linear(1, 1)
    x = torch.tensor([1.0], requires_grad=False)
    y = torch.tensor(2.0, requires_grad=False)
    print('w.weight: ')
    print(w.weight)
    print('w.bias: ')
    print(w.bias)
    weight0 = w.weight
    loss = (y - w(x)) * (y - w(x))
    print('loss: ')
    print(loss)
    w.zero_grad()
    grads = torch.autograd.grad(loss, w.weight, create_graph=True)
    print('grads:')
    print(grads)
    new_weight = w.weight - grads[0]
    print('w.weight:')
    print(w.weight)
    print('new_weight:')
    print(new_weight)
    print('w.new_bias: ')
    print(w.bias)
    print('###########################################')

    # note: can assign parameters to nn.Module and track the original grad_fn of the parameters
    del w.weight
    w.weight = new_weight
    weight1 = new_weight
    print('w.weight: ')
    print(w.weight)
    x = torch.tensor([2.0], requires_grad=False)
    y = torch.tensor(3.0, requires_grad=False)
    loss = (y - w(x)) * (y - w(x))
    print('loss: ')
    print(loss)
    w.zero_grad()
    grads = torch.autograd.grad(loss, w.weight, create_graph=True)
    new_weight = w.weight - grads[0]
    print('grads:')
    print(grads)
    maml_grads = torch.autograd.grad(loss, weight0, create_graph=True)
    print('maml_grads:')
    print(maml_grads)
    print('w.weight:')
    print(w.weight)
    print('new_weight:')
    print(new_weight)
    print('w.new_bias: ')
    print(w.bias)
    print('###########################################')

    del w.weight
    w.weight = new_weight
    print('w.weight: ')
    print(w.weight)
    x = torch.tensor([3.0], requires_grad=False)
    y = torch.tensor(4.0, requires_grad=False)
    loss = (y - w(x)) * (y - w(x))
    print('loss: ')
    print(loss)
    w.zero_grad()
    new_weight = w.weight - grads[0]
    print('grads:')
    print(grads)
    w1_grads = torch.autograd.grad(loss, weight1, create_graph=True)
    print('weight1_grads:')
    print(w1_grads)
    maml_grads = torch.autograd.grad(loss, weight0, create_graph=True)
    print('maml_grads:')
    print(maml_grads)
    print('w.weight:')
    print(w.weight)
    print('new_weight:')
    print(new_weight)
    print('w.new_bias: ')
    print(w.bias)
    print('###########################################')

def detach_test():
    x = torch.tensor(([2.0]), requires_grad=True)
    xx = torch.tensor(([2.0]), requires_grad=True)
    yy = torch.tensor(([1.0]), requires_grad=True)
    x.data = xx.clone().detach() - yy.clone().detach()
    y = x ** 2
    z = 2 * y
    w = z ** 3

    # This is the subpath
    # Do not use detach()
    p = z
    # p = z.detach()
    q = torch.tensor(([2.0]), requires_grad=True)
    pq = p * q
    pq.backward(retain_graph=True)

    w.backward()
    print('x.grad:')
    print(x.grad)
    print('xx.grad:')
    print(xx.grad)
    print('yy.grad:')
    print(yy.grad)
    # x.data = x.data-x.grad.data
    x.data -= x.grad.data
    print('x:')
    print(x)
    print('xx:')
    print(xx)

def grad_test():
    x = torch.tensor(([2.0]), requires_grad=True)
    xx = torch.tensor(([2.0]), requires_grad=True)
    yy = torch.tensor(([1.0]), requires_grad=True)

    x_power = x * x
    x_power = x * x + 1
    # x_power.backward()
    # print('x.grad:')
    # print(x.grad)

    y = x * xx
    z = x * yy
    z.backward()

    print('x.grad:')
    print(x.grad)
    print('xx.grad:')
    print(xx.grad)
    print('yy.grad:')
    print(yy.grad)

def qid_test():
    qid_lists = list()
    qids = ['Quantitative Reasoning (All)_27412', 'Quantitative Reasoning (All)_33555', 'Quantitative Reasoning (All)_7283', 'Quantitative Reasoning (All)_74598', 'Quantitative Reasoning (All)_77421']
    ids = [['Quantitative Reasoning (All)_2741', 'Quantitative Reasoning (All)_3355', 'Quantitative Reasoning (All)_728', 'Quantitative Reasoning (All)_7498', 'Quantitative Reasoning (All)_7421'], ['Quantitative Reasoning (All)_2712', 'Quantitative Reasoning (All)_33555', 'Quantitative Reasoning (All)_7283', 'Quantitative Reasoning (All)_7598', 'Quantitative Reasoning (All)_77421'], ['Quantitative Reasoning (All)_33555', 'Quantitative Reasoning (All)_27412', 'Quantitative Reasoning (All)_7283', 'Quantitative Reasoning (All)_77421', 'Quantitative Reasoning (All)_74598']]

    qids1 = ['Quantitative Reasoning (All)_7498', 'Quantitative Reasoning (All)_3355', 'Quantitative Reasoning (All)_728',
     'Quantitative Reasoning (All)_2741', 'Quantitative Reasoning (All)_7421']
    qids1.sort()
    qid_lists.append(qids1)
    identical_flag = False
    for qids_temp in ids:
        qids_temp.sort()
        if qids_temp == qids:
            identical_flag = True
            break
    if not identical_flag:
        print(qids)
    else:
        print('has same!')

def stack_cat_test():
    retriever_net_policies = list()
    temp = torch.tensor(0.0, requires_grad=True)
    logprob_lists = list()
    logprob = [temp + 1.0, temp + 1.1, temp + 1.2]
    logprob1 = torch.stack(logprob)
    logprob_lists.append(logprob1)
    logprob = [temp + 2.0, temp + 2.1, temp + 2.2]
    logprob1 = torch.stack(logprob)
    logprob_lists.append(logprob1)
    logprob = [torch.tensor(3.0), torch.tensor(3.1), torch.tensor(3.2)]
    logprob_lists.append(torch.stack(logprob))
    a = torch.cat(logprob_lists)
    print(a)
    retriever_net_policies.append(a)

    temp = torch.tensor(1.0, requires_grad=True)
    logprob_lists = list()
    logprob = [temp + 1.0, temp + 1.1, temp + 1.2]
    logprob1 = torch.stack(logprob)
    logprob_lists.append(logprob1)
    logprob = [temp + 2.0, temp + 2.1, temp + 2.2]
    logprob1 = torch.stack(logprob)
    logprob_lists.append(logprob1)
    logprob = [torch.tensor(13.0), torch.tensor(13.1), torch.tensor(13.2)]
    logprob_lists.append(torch.stack(logprob))
    b = torch.cat(logprob_lists)
    print(b)
    retriever_net_policies.append(b)

    c = torch.cat(retriever_net_policies)
    print(c)

if __name__ == "__main__":
    # qid_test()
    # print('----------------')
    # grad_test()
    # print('----------------')
    stack_cat_test()
    print('----------------')
    # gradient_test()
    input_shapes = ((1, 3, 3, 4),)
    for s in input_shapes:
        print(s)
    a = tuple(torch.randn(s) for s in input_shapes)
    print(a)
    print(*a)

    a = list()
    temp = torch.tensor(2.0, requires_grad=True)
    temp = temp.clone().detach()
    a.append(temp)
    print(np.mean(a))

    tensor_list = []
    t1 = torch.tensor([[1.0,2.0],[3.0,4.0]], requires_grad=True)
    tensor_list.append(t1)
    t2 = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], requires_grad=True)
    tensor_list.append(t2)
    t3 = torch.tensor([[-1.0, -1], [-1, -1]], requires_grad=True)
    tensor_list.append(t3)
    bb = torch.stack(tensor_list).mean(0)
    aa = bb.detach()
    print(tensor_list)
    print(aa)
    cc = t1.detach() - aa
    print(cc)

    detach_test()

    x = torch.tensor([3,6,100,5.5,3.4,2.2,-1.0,101])
    orders = torch.topk(x, 3)
    print(orders)

    log_prob_v = [[1,2,3],[4,5,6],[7,8,9]]
    log_prob_v = torch.LongTensor(log_prob_v).to('cuda')
    actions_t = [0,1,2]
    actions_t = torch.LongTensor(actions_t).to('cuda')
    log_prob_actions_v = log_prob_v[[0,1,2], [0,1,2]]
    print(log_prob_actions_v)

    # document_range_list = [i for i in range(10, 20)]
    # softmax_output = [0.1,0.2,0.01,0.02,0.11,0.22,0.003,0,0.4,0.33]
    # softmax_output_list = torch.tensor(softmax_output).to('cuda')
    # softmax_output_prob = F.softmax(softmax_output_list, dim=0)
    # softmax_output_prob_list = softmax_output_prob.tolist()
    # draw_list = []
    # for i in range(5):
    #     draw = choice(document_range_list, size=5, replace=False, p=softmax_output_prob_list)
    #     draw_list.append(draw)
    # print(draw_list)

    document_range_list = [i for i in range(10, 20)]
    cosine_output = [-0.01,0.02,0.003,0,-0.03,0.1,0.2,0.3,0.33,0.13]
    softmax_output_list = torch.tensor(cosine_output).to('cuda')
    softmax_output_prob = F.softmax(softmax_output_list, dim=0)
    orders = torch.topk(softmax_output_prob, 5)
    order_list = orders[1].tolist()
    order_softmax_output_prob = [softmax_output_prob[x] for x in order_list]
    draw_list = []
    for i in range(10):
        draws = []
        draw = list(WeightedRandomSampler(order_softmax_output_prob, 3, replacement=False))
        for j in draw:
            draws.append(order_list[j])
        draw_list.append(draws)
    print(draw_list)

    for cosine, softmax_prob in zip(cosine_output,softmax_output_prob):
        print('%s,%s'%(str(cosine),str(softmax_prob)))

    retriever_net_policies = list()

    temp = torch.tensor(0.0, requires_grad=True)
    logprob_lists = list()
    logprob = [temp+1.0, temp+1.1, temp+1.2]
    logprob_lists.append(torch.tensor(logprob))
    logprob = [temp+2.0, temp+2.1, temp+2.2]
    logprob_lists.append(torch.tensor(logprob))
    logprob = [torch.tensor(3.0), torch.tensor(3.1), torch.tensor(3.2)]
    logprob_lists.append(torch.tensor(logprob))
    a = torch.cat(logprob_lists)
    print(a)

    # retriever_net_policies.append(logprob_lists)
    #
    # logprob_lists = list()
    # logprob = [temp+11.0, temp+11.1, temp+11.2]
    # logprob_lists.append(logprob)
    # logprob = [torch.tensor(12.0), torch.tensor(12.1), torch.tensor(12.2)]
    # logprob_lists.append(logprob)
    # logprob = [torch.tensor(13.0), torch.tensor(13.1), torch.tensor(13.2)]
    # logprob_lists.append(logprob)
    # retriever_net_policies.append(logprob_lists)

    retriever_net_policies_tensor = torch.tensor(retriever_net_policies).view(-1)
    retriever_net_policies_tensor = retriever_net_policies_tensor.cuda()
    print(retriever_net_policies_tensor)





