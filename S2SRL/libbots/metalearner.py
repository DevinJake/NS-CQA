import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from . import data, model, utils, retriever, reparam_module, adabound
import torch.optim as optim
import torch.nn.functional as F
import random
import logging
from torch.utils.data.sampler import WeightedRandomSampler
log = logging.getLogger("MetaLearner")

class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self, net=None, retriever_net = None, device='cpu', beg_token=None, end_token = None, adaptive=False, samples=5, train_data_support_944K=None, rev_emb_dict=None, first_order=False, fast_lr=0.001, meta_optimizer_lr=0.0001, dial_shown = False, dict=None, dict_weak=None, steps=5, weak_flag=False, query_embed=True):
        self.net = net
        self.retriever_net = retriever_net
        self.device = device
        self.beg_token = beg_token
        self.end_token = end_token
        # The training data from which the top-N samples (support set) are found.
        self.train_data_support_944K = train_data_support_944K
        self.rev_emb_dict = rev_emb_dict
        self.adaptive = adaptive
        self.samples = samples
        self.first_order = first_order
        self.fast_lr = fast_lr
        self.meta_optimizer_lr = meta_optimizer_lr
        # self.meta_optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        # self.meta_optimizer = optim.Adam(net.parameters(), lr=meta_optimizer_lr, eps=1e-3)
        self.meta_optimizer = None if net is None else optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=meta_optimizer_lr, eps=1e-3)
        # self.inner_optimizer = optim.Adam(net.parameters(), lr=fast_lr, eps=1e-3)
        self.inner_optimizer = None if net is None else optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=fast_lr, eps=1e-3)
        self.dial_shown = dial_shown
        self.retriever = None if (dict is None or dict_weak is None) else retriever.Retriever(dict, dict_weak)
        self.retriever_optimizer = None if retriever_net is None else adabound.AdaBound(filter(lambda p: p.requires_grad, retriever_net.parameters()), lr=1e-3, final_lr=0.1)
        self.steps = steps
        self.weak_flag = weak_flag
        self.query_embed = query_embed
        '''# note: Reparametrize it!
        self.reparam_net = reparam_module.ReparamModule(self.net)
        print(f"reparam_net has {self.reparam_net.param_numel} parameters")
        # module.parameters() is traversing the parameters by elements in the module._parameters.items().
        # Tensors like weights or biases are stored in the module._parameters, and the items in _parameters and weights/biases are the same items.
        # In class ReparamModule(net), the weights/biases of net have been deleted from each module's _parameters.
        # b = reparam_net.parameters()
        # for param in b:
        #     print (param)
        # (reparam_net.flat_param,): Creating a tuple with one element
        # parameters(): get items from module._parameters.items()
        assert tuple(self.reparam_net.parameters()) == (self.reparam_net.flat_param,)
        print(f"reparam_net now has **only one** vector parameter of shape {self.reparam_net.flat_param.shape}")'''

    def reptile_meta_update(self, running_vars, old_param_dict, beta):
        for (name, param) in self.net.named_parameters():
            if param.requires_grad:
                mean = torch.stack(running_vars[name]).mean(0).clone().detach()
                old = old_param_dict[name].clone().detach()
                param.data = old + beta * (mean - old)

    def first_order_meta_update(self, grads_list, old_param_dict):
        # self.named_parameters() is consisted of an iterator over module parameters,
        # yielding both the name of the parameter as well as the parameter itself.
        # grads is the gradient of self.parameters().
        # Each module parameter is computed as parameter = parameter - step_size * grad.
        # When being saved in the OrderedDict of self.named_parameters(), it likes:
        # OrderedDict([('sigma', Parameter containing:
        # tensor([0.6931, 0.6931], requires_grad=True)), ('0.weight', Parameter containing:
        # tensor([[1., 1.],
        #         [1., 1.]], requires_grad=True)), ('0.bias', Parameter containing:
        # tensor([0., 0.], requires_grad=True)), ('1.weight', Parameter containing:
        # tensor([[1., 1.],
        #         [1., 1.]], requires_grad=True)), ('1.bias', Parameter containing:
        # tensor([0., 0.], requires_grad=True))])

        # Theta <- Theta - beta * (1/k) * ∑_(i=1:k)[grad(loss_theta_in/theta_in)]
        # Update the value of the current parameters of the model.
        for (name, param) in self.net.named_parameters():
            if param.requires_grad:
                average_grad = torch.stack(grads_list[name]).mean(0).clone().detach()
                # old_param = old_param_dict[name].clone().detach()
                param.data = old_param_dict[name].clone().detach() - self.meta_optimizer_lr * average_grad

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        # It’s important to call this before loss.backward(),
        # otherwise you’ll accumulate the gradients from multiple passes.
        self.meta_optimizer.zero_grad()
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
        # These are accumulated into x.grad for every parameter x. In pseudo-code:
        # x.grad += dloss/dx
        loss.backward()
        # To conduct a gradient ascent to minimize the loss (which is to maximize the reward).
        # optimizer.step updates the value of x using the gradient x.grad.
        # For example, the SGD optimizer performs:
        # x += -lr * x.grad
        self.meta_optimizer.step()

    def reparam_meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        # It’s important to call this before loss.backward(),
        # otherwise you’ll accumulate the gradients from multiple passes.
        # self.reparam_net.zero_grad()

        theta_0 = self.reparam_net.flat_param
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
        # These are accumulated into x.grad for every parameter x. In pseudo-code:
        # x.grad += dloss/dx
        loss.backward()
        # To conduct a gradient ascent to minimize the loss (which is to maximize the reward).
        # optimizer.step updates the value of x using the gradient x.grad.
        # For example, the SGD optimizer performs:
        # x += -lr * x.grad
        theta_0.data = theta_0.data - self.meta_optimizer_lr * theta_0.grad.data

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                # When you get the parameters of your net, it does not clone the tensors.
                # So in your case, before and after contain the same tensors.
                # So when the optimizer update the weights in place, it updates both your lists.
                # You could call .clone() on each parameter so that a deep copy will be used to solve the problem.
                # clone(): Returns a copy of the self tensor. The copy has the same size and data type as self.
                # Unlike copy_(), this function is recorded in the computation graph.
                # Gradients propagating to the cloned tensor will propagate to the original tensor.
                # But when using optimizer.step(), the cloned new parameters will not update.
                # param_dict[name] = param.to(device=self.device)
                # star: shouldn't it be param.to(device=self.device).clone().detach()?
                # Since 'gradients propagating to the cloned tensor will propagate to the original tensor',
                # when using the cloned parameters to compute gradients, the gradients will be accumlated in the original tensor
                # but not the cloned ones.
                param_dict[name] = param.to(device=self.device).clone()
        return param_dict

    def get_net_named_parameter(self):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        params = self.net.named_parameters()
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                # When you get the parameters of your net, it does not clone the tensors.
                # So in your case, before and after contain the same tensors.
                # So when the optimizer update the weights in place, it updates both your lists.
                # You could call .clone() on each parameter so that a deep copy will be used to solve the problem.
                # clone(): Returns a copy of the self tensor. The copy has the same size and data type as self.
                # Unlike copy_(), this function is recorded in the computation graph.
                # Gradients propagating to the cloned tensor will propagate to the original tensor.
                # But when using optimizer.step(), the cloned new parameters will not update.
                # param_dict[name] = param.to(device=self.device)
                # Since 'gradients propagating to the cloned tensor will propagate to the original tensor',
                # when using the cloned parameters to compute gradients, the gradients will be accumlated in the original tensor
                # but not the cloned ones.
                param_dict[name] = param.to(device=self.device).clone().detach()
        return param_dict

    def get_net_parameter(self):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        params = self.net.named_parameters()
        param_dict = dict()
        for name, param in params:
            param_dict[name] = param.to(device=self.device).clone().detach()
        return param_dict

    def establish_support_set(self, task, N=5, weak=False, train_data_support_944K=None):
        # Find top-N in train_data_support;
        # get_top_N(train_data, train_data_support, N)
        # Support set is none. Use the training date per se as support set.
        batch = list()
        if N==0:
            batch.append(task)
        else:
            key_name, key_weak, question, qid = self.retriever.AnalyzeQuestion(task[1])
            topNList = self.retriever.RetrieveWithMaxTokens(N, key_name, key_weak, question, train_data_support_944K, weak, qid)
            for name in topNList:
                qid = list(name.keys())[0] if len(name) > 0 else 'NONE'
                if qid in train_data_support_944K:
                    batch.append(train_data_support_944K[qid])
                if len(batch) ==N:
                    break
        return batch

    # Randomly select training samples as support set.
    def establish_random_support_set(self, task=None, N=5, train_data_support_944K=None):
        batch = list()
        if N==0:
            batch.append(task)
        else:
            key_name, key_weak, question, qid = self.retriever.AnalyzeQuestion(task[1])
            topNList = self.retriever.RetrieveRandomSamplesWithMaxTokens(N=N, key_weak=key_weak, train_data_944k=train_data_support_944K, qid=qid)
            for name in topNList:
                qid = list(name.keys())[0] if len(name) > 0 else 'NONE'
                if qid in train_data_support_944K:
                    batch.append(train_data_support_944K[qid])
                if len(batch) ==N:
                    break
        return batch

    # Return support samples strictly following the ranking.
    def establish_support_set_by_retriever_argmax(self, task, N=5, train_data_support_944K=None, docID_dict=None, rev_docID_dict=None, emb_dict=None, qtype_docs_range=None):
        # Find top-N in train_data_support;
        # get_top_N(train_data, train_data_support, N)
        # Support set is none. Use the training date per se as support set.
        # temp_dict = self.retriever_net.get_retriever_net_parameter()
        batch = list()
        if N==0:
            batch.append(task)
        else:
            key_name, key_weak, question, qid = self.retriever.AnalyzeQuestion(task[1])
            if not self.query_embed:
                query_tensor = torch.tensor(self.retriever_net.pack_input(docID_dict['qid']).tolist(), requires_grad=False).cuda()
            else:
                query_tensor = data.get_question_embedding(question, emb_dict, self.net)
            if key_weak in qtype_docs_range:
                document_range = (qtype_docs_range[key_weak]['start'], qtype_docs_range[key_weak]['end'])
            else:
                document_range = (0, len(docID_dict))
            logsoftmax_output, _, _ = self.retriever_net(query_tensor, document_range)
            orders = torch.topk(logsoftmax_output, N+10)
            # order_temp = self.retriever_net.calculate_rank(logsoftmax_output.tolist())
            # first_index = 0
            # for i, temp in enumerate(order_temp):
            #     if temp==1:
            #         first_index = i
            #         break
            order_list = orders[1].tolist()
            topNIndices = [k+document_range[0] for k in order_list]
            topNList = [rev_docID_dict[k] for k in topNIndices]
            for qid in topNList:
                if qid in train_data_support_944K:
                    batch.append(train_data_support_944K[qid])
                if len(batch)==N:
                    break
        return batch

    # Return support samples strictly following the probability distribution.
    def establish_support_set_by_retriever_sampling(self, task, N=5, train_data_support_944K=None, docID_dict=None, rev_docID_dict=None, emb_dict=None, qtype_docs_range=None, number_of_supportsets=5):
        retriever_total_samples = 0
        retriever_skip_samples = 0
        batch_list = list()
        logprob_list = list()
        key_name, key_weak, question, qid = self.retriever.AnalyzeQuestion(task[1])
        if not self.query_embed:
            query_tensor = torch.tensor(self.retriever_net.pack_input(docID_dict['qid']).tolist(),
                                        requires_grad=False).cuda()
        else:
            query_tensor = data.get_question_embedding(question, emb_dict, self.net)
        if key_weak in qtype_docs_range:
            document_range = (qtype_docs_range[key_weak]['start'], qtype_docs_range[key_weak]['end'])
        else:
            document_range = (0, len(docID_dict))
        logsoftmax_output, softmax_output, cos_output = self.retriever_net(query_tensor, document_range)
        # Get top N+45 samples.
        # A namedtuple of (values, indices) is returned,
        # where the indices are the indices of the elements in the original input tensor.
        orders = torch.topk(cos_output, N+45)
        order_list = orders[1].tolist()
        # To confine the search space in the top N+45 samples with the highest probabilities.
        order_softmax_output_prob = [softmax_output[x] for x in order_list]
        qid_lists = list()
        for i in range(number_of_supportsets):
            batch = list()
            logprob_for_samples = list()
            # Samples elements from [0,..,len(weights)-1] with probability of top N+45 samples.
            # You can also use the keyword replace=False to change the behavior so that sampling is without replacement.
            # WeightedRandomSampler: Samples elements from [0,..,len(weights)-1] with given probabilities (weights).
            draw = list(WeightedRandomSampler(order_softmax_output_prob, N, replacement=False))
            draw_list = [order_list[j] for j in draw]

            # draw_without_filtering = list(WeightedRandomSampler(softmax_output, N+10, replacement=False))
            # topNIndices_1 = [k + document_range[0] for k in draw_without_filtering]
            # logprobs_1 = [logsoftmax_output[k] for k in draw_without_filtering]
            # topNList_1 = [rev_docID_dict[k] for k in topNIndices_1]

            topNIndices = [k + document_range[0] for k in draw_list]
            logprobs = [logsoftmax_output[k] for k in draw_list]
            topNList = [rev_docID_dict[k] for k in topNIndices]
            qids = list()
            for qid, logprob in zip(topNList, logprobs):
                if qid in train_data_support_944K:
                    batch.append(train_data_support_944K[qid])
                    logprob_for_samples.append(logprob)
                    qids.append(qid)
                if len(batch) == N:
                    break
            qids.sort()
            retriever_total_samples += 1
            if len(qid_lists) == 0:
                batch_list.append(batch)
                logprob_list.append(torch.stack(logprob_for_samples))
                qid_lists.append(qids)
            else:
                identical_flag = False
                for qids_temp in qid_lists:
                    if qids_temp == qids:
                        retriever_skip_samples += 1
                        identical_flag = True
                        break
                if not identical_flag:
                    qid_lists.append(qids)
                    batch_list.append(batch)
                    logprob_list.append(torch.stack(logprob_for_samples))
        return batch_list, logprob_list, retriever_total_samples, retriever_skip_samples

    def reparam_update_params(self, inner_loss, theta, lr=0.1, first_order=False):
        # NOTE: what is meaning of one step here? What is one step?
        # One step here means the loss of a batch of samples (a task) are computed.
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        # nn.Module.zero_grad() Sets gradients of all model parameters to zero.
        # It’s important to call this before loss.backward(),
        # otherwise you’ll accumulate the gradients from multiple passes.
        # self.net.zero_grad(names_weights_copy)
        # self.reparam_net.zero_grad()

        # create_graph (bool, optional) – If True, graph of the derivative will be constructed,
        # allowing to compute higher order derivative products. Defaults to False.
        # first_order is set as false and not first_order is true, so create_graph is set as true,
        # which means allowing to compute higher order derivative products.
        # self.parameters(): Returns an iterator over module parameters.
        # torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False):
        # outputs (sequence of Tensor) – outputs of the differentiated function.
        # inputs (sequence of Tensor) – Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).
        # In this case, the gradient of self.parameters() is inputs.
        # The autograd.grad function returns an object that match the inputs argument,
        # so we could get the the gradient of self.parameters() as returned value.
        # Here if we do not use the first_order configuration, we set it as true.
        # grads = torch.autograd.grad(inner_loss, names_weights_copy.values(),
        #     create_graph=not first_order)
        gtheta, = torch.autograd.grad(inner_loss, theta,
                                    create_graph=not first_order)
        # update
        theta = theta - lr * gtheta
        return theta

    def update_params(self, inner_loss, names_weights_copy, step_size=0.1, first_order=False):
        # NOTE: what is meaning of one step here? What is one step?
        # One step here means the loss of a batch of samples (a task) are computed.
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        # nn.Module.zero_grad() Sets gradients of all model parameters to zero.
        # It’s important to call this before loss.backward(),
        # otherwise you’ll accumulate the gradients from multiple passes.
        # self.net.zero_grad(names_weights_copy)
        self.net.zero_grad()

        # create_graph (bool, optional) – If True, graph of the derivative will be constructed,
        # allowing to compute higher order derivative products. Defaults to False.
        # first_order is set as false and not first_order is true, so create_graph is set as true,
        # which means allowing to compute higher order derivative products.
        # self.parameters(): Returns an iterator over module parameters.
        # torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False):
        # outputs (sequence of Tensor) – outputs of the differentiated function.
        # inputs (sequence of Tensor) – Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).
        # In this case, the gradient of self.parameters() is inputs.
        # The autograd.grad function returns an object that match the inputs argument,
        # so we could get the the gradient of self.parameters() as returned value.
        # Here if we do not use the first_order configuration, we set it as true.
        # grads = torch.autograd.grad(inner_loss, names_weights_copy.values(),
        #     create_graph=not first_order)
        grads = torch.autograd.grad(inner_loss, self.net.parameters(),
                                    create_graph=not first_order)
        # self.named_parameters() is consisted of an iterator over module parameters,
        # yielding both the name of the parameter as well as the parameter itself.
        # grads is the gradient of self.parameters().
        # Each module parameter is computed as parameter = parameter - step_size * grad.
        # When being saved in the OrderedDict of self.named_parameters(), it likes:
        # OrderedDict([('sigma', Parameter containing:
        # tensor([0.6931, 0.6931], requires_grad=True)), ('0.weight', Parameter containing:
        # tensor([[1., 1.],
        #         [1., 1.]], requires_grad=True)), ('0.bias', Parameter containing:
        # tensor([0., 0.], requires_grad=True)), ('1.weight', Parameter containing:
        # tensor([[1., 1.],
        #         [1., 1.]], requires_grad=True)), ('1.bias', Parameter containing:
        # tensor([0., 0.], requires_grad=True))])
        updated_names_weights_dict = dict()
        # names_grads_wrt_params = dict(zip(names_weights_copy.keys(), grads))
        # for key in names_grads_wrt_params.keys():
        #     print(str(key)+': ')
        #     print('names_weights_copy:')
        #     print(names_weights_copy[key])
        #     print('names_grads_wrt_params:')
        #     print(names_grads_wrt_params[key])
        #     updated_names_weights_dict[key] = names_weights_copy[key] - step_size * names_grads_wrt_params[key]
        #     print('updated_names_weights_dict:')
        #     print(updated_names_weights_dict[key])

        for (name, param), grad in zip(self.net.named_parameters(), grads):
            updated_names_weights_dict[name] = param - step_size * grad
            # print(str(name) + ': ')
            # print('self.net.named_parameters:')
            # print(param)
            # print('grad:')
            # print(grad)
            # print('updated_names_weights_dict:')
            # print(updated_names_weights_dict[name])

        return updated_names_weights_dict

    # The loss used to calculate theta' for each pseudo-task.
    # Compute the inner loss for the one-step gradient update.
    # The inner loss is REINFORCE with baseline [2].
    def inner_loss(self, task, weights=None, dial_shown=True):
        total_samples = 0
        skipped_samples = 0

        batch = list()
        batch.append(task)
        true_reward_argmax_step = []
        true_reward_sample_step = []

        # TODO
        if weights is not None:
            self.net.insert_new_parameter(weights, True)
        # input_seq: the padded and embedded batch-sized input sequence.
        # input_batch: the token ID matrix of batch-sized input sequence. Each row is corresponding to one input sentence.
        # output_batch: the token ID matrix of batch-sized output sequences. Each row is corresponding to a list of several output sentences.
        input_seq, input_batch, output_batch = self.net.pack_batch_no_out(batch, self.net.emb, self.device)
        input_seq = input_seq.cuda()
        # Get (two-layer) hidden state of encoder of samples in batch.
        # enc = net.encode(input_seq)
        context, enc = self.net.encode_context(input_seq)

        net_policies = []
        net_actions = []
        net_advantages = []
        # Transform ID to embedding.
        beg_embedding = self.net.emb(self.beg_token)
        beg_embedding = beg_embedding.cuda()

        for idx, inp_idx in enumerate(input_batch):
            # # Test whether the input sequence is correctly transformed into indices.
            # input_tokens = [rev_emb_dict[temp_idx] for temp_idx in inp_idx]
            # print (input_tokens)
            # Get IDs of reference sequences' tokens corresponding to idx-th input sequence in batch.
            qa_info = output_batch[idx]
            # print("            Support sample %s is training..." % (qa_info['qid']))
            # print (qa_info['qid'])
            # # Get the (two-layer) hidden state of encoder of idx-th input sequence in batch.
            item_enc = self.net.get_encoded_item(enc, idx)
            # # 'r_argmax' is the list of out_logits list and 'actions' is the list of output tokens.
            # # The output tokens are generated greedily by using chain_argmax (using last setp's output token as current input token).
            r_argmax, actions = self.net.decode_chain_argmax(item_enc, beg_embedding, data.MAX_TOKENS, context[idx],
                                                             stop_at_token=self.end_token)
            # Show what the output action sequence is.
            action_tokens = []
            for temp_idx in actions:
                if temp_idx in self.rev_emb_dict and self.rev_emb_dict.get(temp_idx) != '#END':
                    action_tokens.append(str(self.rev_emb_dict.get(temp_idx)).upper())
            # Get the highest BLEU score as baseline used in self-critic.
            # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
            # Otherwise the adaptive reward is used.
            # argmax_reward = utils.calc_True_Reward(action_tokens, qa_info, self.adaptive)
            argmax_reward = random.random()
            true_reward_argmax_step.append(argmax_reward)

            # # In this case, the BLEU score is so high that it is not needed to train such case with RL.
            # if not args.disable_skip and argmax_reward > 0.99:
            #     skipped_samples += 1
            #     continue

            # In one epoch, when model is optimized for the first time, the optimized result is displayed here.
            # After that, all samples in this epoch don't display anymore.
            if not dial_shown:
                # data.decode_words transform IDs to tokens.
                log.info("Input: %s", utils.untokenize(data.decode_words(inp_idx, self.rev_emb_dict)))
                log.info("Argmax: %s, reward=%.4f", utils.untokenize(data.decode_words(actions, self.rev_emb_dict)),
                         argmax_reward)

            action_memory = list()
            for _ in range(self.samples):
                # 'r_sample' is the list of out_logits list and 'actions' is the list of output tokens.
                # The output tokens are sampled following probabilitis by using chain_sampling.
                r_sample, actions = self.net.decode_chain_sampling(item_enc, beg_embedding, data.MAX_TOKENS,
                                                                   context[idx], stop_at_token=self.end_token)
                total_samples += 1

                # Omit duplicate action sequence to decrease the computing time and to avoid the case that
                # the probability of such kind of duplicate action sequences would be increased redundantly and abnormally.
                duplicate_flag = False
                if len(action_memory) > 0:
                    for temp_list in action_memory:
                        if utils.duplicate(temp_list, actions):
                            duplicate_flag = True
                            break
                if not duplicate_flag:
                    action_memory.append(actions)
                else:
                    skipped_samples += 1
                    continue
                # Show what the output action sequence is.
                action_tokens = []
                for temp_idx in actions:
                    if temp_idx in self.rev_emb_dict and self.rev_emb_dict.get(temp_idx) != '#END':
                        action_tokens.append(str(self.rev_emb_dict.get(temp_idx)).upper())

                # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
                # Otherwise the adaptive reward is used.
                # sample_reward = utils.calc_True_Reward(action_tokens, qa_info, self.adaptive)
                sample_reward = random.random()
                true_reward_sample_step.append(sample_reward)

                if not dial_shown:
                    log.info("Sample: %s, reward=%.4f", utils.untokenize(data.decode_words(actions, self.rev_emb_dict)),
                             sample_reward)

                net_policies.append(r_sample)
                net_actions.extend(actions)
                # Regard argmax_bleu calculated from decode_chain_argmax as baseline used in self-critic.
                # Each token has same reward as 'sample_bleu - argmax_bleu'.
                # [x] * y: stretch 'x' to [1*y] list in which each element is 'x'.

                # # If the argmax_reward is 1.0, then whatever the sample_reward is,
                # # the probability of actions that get reward = 1.0 could not be further updated.
                # # The GAMMA is used to adjust this scenario.
                # if argmax_reward == 1.0:
                #     net_advantages.extend([sample_reward - argmax_reward + GAMMA] * len(actions))
                # else:
                #     net_advantages.extend([sample_reward - argmax_reward] * len(actions))

                net_advantages.extend([sample_reward - argmax_reward] * len(actions))

        if not net_policies:
            log.info("The net_policies is empty!")
            # TODO the format of 0.0 should be the same as loss_v.
            return 0.0, total_samples, skipped_samples

        # Data for decode_chain_sampling samples and the number of such samples is the same as args.samples parameter.
        # Logits of all output tokens whose size is N * output vocab size; N is the number of output tokens of decode_chain_sampling samples.
        policies_v = torch.cat(net_policies)
        policies_v = policies_v.cuda()
        # Indices of all output tokens whose size is 1 * N;
        actions_t = torch.LongTensor(net_actions).to(self.device)
        actions_t = actions_t.cuda()
        # All output tokens reward whose size is 1 * N;
        adv_v = torch.FloatTensor(net_advantages).to(self.device)
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
        # For the optimizer is Adam (Adaptive Moment Estimation) which is a optimizer used for gradient descent.
        # Therefore, to maximize ▽J (log_prob_actions_v) is to minimize -▽J.
        # .mean() is to calculate Monte Carlo sampling.
        loss_policy_v = -log_prob_actions_v.mean()
        loss_policy_v = loss_policy_v.cuda()

        loss_v = loss_policy_v
        return loss_v, total_samples, skipped_samples, true_reward_argmax_step, true_reward_sample_step

    def first_order_inner_loss(self, task, dial_shown=True, mc=False):
        total_samples = 0
        skipped_samples = 0

        batch = list()
        batch.append(task)
        true_reward_argmax_step = []
        true_reward_sample_step = []

        # input_seq: the padded and embedded batch-sized input sequence.
        # input_batch: the token ID matrix of batch-sized input sequence. Each row is corresponding to one input sentence.
        # output_batch: the token ID matrix of batch-sized output sequences. Each row is corresponding to a list of several output sentences.
        input_seq, input_batch, output_batch = self.net.pack_batch_no_out(batch, self.net.emb, self.device)
        input_seq = input_seq.cuda()
        # Get (two-layer) hidden state of encoder of samples in batch.
        # enc = net.encode(input_seq)
        context, enc = self.net.encode_context(input_seq)

        net_policies = []
        net_actions = []
        net_advantages = []
        net_losses = []
        # Transform ID to embedding.
        beg_embedding = self.net.emb(self.beg_token)
        beg_embedding = beg_embedding.cuda()

        for idx, inp_idx in enumerate(input_batch):
            # # Test whether the input sequence is correctly transformed into indices.
            # input_tokens = [rev_emb_dict[temp_idx] for temp_idx in inp_idx]
            # print (input_tokens)
            # Get IDs of reference sequences' tokens corresponding to idx-th input sequence in batch.
            qa_info = output_batch[idx]
            # print("            Support sample %s is training..." % (qa_info['qid']))
            # print (qa_info['qid'])
            # # Get the (two-layer) hidden state of encoder of idx-th input sequence in batch.
            item_enc = self.net.get_encoded_item(enc, idx)
            # # 'r_argmax' is the list of out_logits list and 'actions' is the list of output tokens.
            # # The output tokens are generated greedily by using chain_argmax (using last setp's output token as current input token).
            r_argmax, actions = self.net.decode_chain_argmax(item_enc, beg_embedding, data.MAX_TOKENS, context[idx],
                                                             stop_at_token=self.end_token)
            # Show what the output action sequence is.
            action_tokens = []
            for temp_idx in actions:
                if temp_idx in self.rev_emb_dict and self.rev_emb_dict.get(temp_idx) != '#END':
                    action_tokens.append(str(self.rev_emb_dict.get(temp_idx)).upper())
            # Get the highest BLEU score as baseline used in self-critic.
            # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
            # Otherwise the adaptive reward is used.
            argmax_reward = utils.calc_True_Reward(action_tokens, qa_info, self.adaptive)
            # argmax_reward = random.random()
            true_reward_argmax_step.append(argmax_reward)

            # # In this case, the BLEU score is so high that it is not needed to train such case with RL.
            # if not args.disable_skip and argmax_reward > 0.99:
            #     skipped_samples += 1
            #     continue

            # In one epoch, when model is optimized for the first time, the optimized result is displayed here.
            # After that, all samples in this epoch don't display anymore.
            if not dial_shown:
                # data.decode_words transform IDs to tokens.
                log.info("Input: %s", utils.untokenize(data.decode_words(inp_idx, self.rev_emb_dict)))
                log.info("Argmax: %s, reward=%.4f", utils.untokenize(data.decode_words(actions, self.rev_emb_dict)),
                         argmax_reward)

            action_memory = list()

            sample_losses = []
            for _ in range(self.samples):
                # Monte-carlo: the data for each task in a batch of tasks.
                inner_net_policies = []
                inner_net_actions = []
                inner_net_advantages = []

                # 'r_sample' is the list of out_logits list and 'actions' is the list of output tokens.
                # The output tokens are sampled following probabilities by using chain_sampling.
                r_sample, actions = self.net.decode_chain_sampling(item_enc, beg_embedding, data.MAX_TOKENS,
                                                                   context[idx], stop_at_token=self.end_token)
                total_samples += 1

                # Omit duplicate action sequence to decrease the computing time and to avoid the case that
                # the probability of such kind of duplicate action sequences would be increased redundantly and abnormally.
                duplicate_flag = False
                if len(action_memory) > 0:
                    for temp_list in action_memory:
                        if utils.duplicate(temp_list, actions):
                            duplicate_flag = True
                            break
                if not duplicate_flag:
                    action_memory.append(actions)
                else:
                    skipped_samples += 1
                    continue
                # Show what the output action sequence is.
                action_tokens = []
                for temp_idx in actions:
                    if temp_idx in self.rev_emb_dict and self.rev_emb_dict.get(temp_idx) != '#END':
                        action_tokens.append(str(self.rev_emb_dict.get(temp_idx)).upper())

                # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
                # Otherwise the adaptive reward is used.
                sample_reward = utils.calc_True_Reward(action_tokens, qa_info, self.adaptive)
                # sample_reward = random.random()
                true_reward_sample_step.append(sample_reward)
                advantages = [sample_reward - argmax_reward] * len(actions)

                if not dial_shown:
                    log.info("Sample: %s, reward=%.4f", utils.untokenize(data.decode_words(actions, self.rev_emb_dict)),
                             sample_reward)

                if not mc:
                    net_policies.append(r_sample)
                    net_actions.extend(actions)
                    # Regard argmax_bleu calculated from decode_chain_argmax as baseline used in self-critic.
                    # Each token has same reward as 'sample_bleu - argmax_bleu'.
                    # [x] * y: stretch 'x' to [1*y] list in which each element is 'x'.

                    # # If the argmax_reward is 1.0, then whatever the sample_reward is,
                    # # the probability of actions that get reward = 1.0 could not be further updated.
                    # # The GAMMA is used to adjust this scenario.
                    # if argmax_reward == 1.0:
                    #     net_advantages.extend([sample_reward - argmax_reward + GAMMA] * len(actions))
                    # else:
                    #     net_advantages.extend([sample_reward - argmax_reward] * len(actions))
                    net_advantages.extend(advantages)

                else:
                    inner_net_policies.append(r_sample)
                    inner_net_actions.extend(actions)
                    inner_net_advantages.extend(advantages)

                if mc:
                    inner_policies_v = torch.cat(inner_net_policies).to(self.device)
                    # Indices of all output tokens whose size is 1 * N;
                    inner_actions_t = torch.LongTensor(inner_net_actions).to(self.device)
                    # All output tokens reward whose size is 1 *pack_batch N;
                    inner_adv_v = torch.FloatTensor(inner_net_advantages).to(self.device)
                    # Compute log(softmax(logits)) of all output tokens in size of N * output vocab size;
                    inner_log_prob_v = F.log_softmax(inner_policies_v, dim=1).to(self.device)
                    # Q_1 = Q_2 =...= Q_n = BLEU(OUT,REF);
                    # ▽J = Σ_n[Q▽logp(T)] = ▽Σ_n[Q*logp(T)] = ▽[Q_1*logp(T_1)+Q_2*logp(T_2)+...+Q_n*logp(T_n)];
                    # log_prob_v[range(len(net_actions)), actions_t]: for each output, get the output token's log(softmax(logits)).
                    # adv_v * log_prob_v[range(len(net_actions)), actions_t]:
                    # get Q * logp(T) for all tokens of all decode_chain_sampling samples in size of 1 * N;
                    inner_log_prob_actions_v = inner_adv_v * inner_log_prob_v[
                        range(len(inner_net_actions)), inner_actions_t].to(self.device)
                    # For the optimizer is Adam (Adaptive Moment Estimation) which is a optimizer used for gradient descent.
                    # Therefore, to maximize ▽J (log_prob_actions_v) is to minimize -▽J.
                    # .sum() is calculate the loss for a sample.
                    inner_sample_loss_policy_v = -inner_log_prob_actions_v.sum().to(self.device)
                    sample_losses.append(inner_sample_loss_policy_v)

            if mc:
                task_loss = torch.stack(sample_losses).to(self.device)
                inner_task_loss_policy_v = task_loss.mean().to(self.device)
                # Record the loss for each task in a batch.
                net_losses.append(inner_task_loss_policy_v)

        if not net_losses and not net_policies:
            log.info("The net_policies is empty!")
            # TODO the format of 0.0 should be the same as loss_v.
            return 0.0, total_samples, skipped_samples

        if not mc:
            # Data for decode_chain_sampling samples and the number of such samples is the same as args.samples parameter.
            # Logits of all output tokens whose size is N * output vocab size; N is the number of output tokens of decode_chain_sampling samples.
            policies_v = torch.cat(net_policies)
            policies_v = policies_v.cuda()
            # Indices of all output tokens whose size is 1 * N;
            actions_t = torch.LongTensor(net_actions).to(self.device)
            actions_t = actions_t.cuda()
            # All output tokens reward whose size is 1 * N;
            adv_v = torch.FloatTensor(net_advantages).to(self.device)
            adv_v = adv_v.cuda()
            # Compute log(softmax(logits)) of all output tokens in size of N * output vocab size;
            log_prob_v = F.log_softmax(policies_v, dim=1)
            log_prob_v = log_prob_v.cuda()
            # Q_1 = Q_2 =...= Q_n = BLEU(OUT,REF);
            # ▽J = Σ_n[Q▽logp(T)] = ▽Σ_n[Q*logp(T)] = ▽[Q_1*logp(T_1)+Q_2*logp(T_2)+...+Q_n*logp(T_n)];
            # log_prob_v[range(len(net_actions)), actions_t]: for each output, get the output token's log(softmax(logits)).
            # adv_v * log_prob_v[range(len(net_actions)), actions_t]:
            # get Q * logp(T) for all tokens of all decode_chain_sampling samples in size of 1 * N;
            # Suppose log_prob_v is a two-dimensional tensor, value of which is [[1,2,3],[4,5,6],[7,8,9]];
            # log_prob_actions_v = log_prob_v[[0,1,2], [0,1,2]]
            # log_prob_actions_v is: tensor([1, 5, 9], device='cuda:0').
            log_prob_actions_v = adv_v * log_prob_v[range(len(net_actions)), actions_t]
            log_prob_actions_v = log_prob_actions_v.cuda()
            # For the optimizer is Adam (Adaptive Moment Estimation) which is a optimizer used for gradient descent.
            # Therefore, to maximize ▽J (log_prob_actions_v) is to minimize -▽J.
            # .mean() is to calculate Monte Carlo sampling.
            loss_policy_v = -log_prob_actions_v.mean()
            loss_policy_v = loss_policy_v.cuda()

        else:
            batch_net_losses = torch.stack(net_losses).to(self.device)
            # .mean() is utilized to calculate Mini-Batch Gradient Descent.
            loss_policy_v = batch_net_losses.mean().to(self.device)

        loss_v = loss_policy_v
        return loss_v, total_samples, skipped_samples, true_reward_argmax_step, true_reward_sample_step

    def reparam_inner_loss(self, task, weights=None, dial_shown=True):
        total_samples = 0
        skipped_samples = 0

        batch = list()
        batch.append(task)
        true_reward_argmax_step = []
        true_reward_sample_step = []

        # Using input parameters to insert into the model.
        self.reparam_net.set_parameter_buffer(weights)

        # input_seq: the padded and embedded batch-sized input sequence.
        # input_batch: the token ID matrix of batch-sized input sequence. Each row is corresponding to one input sentence.
        # output_batch: the token ID matrix of batch-sized output sequences. Each row is corresponding to a list of several output sentences.
        input_seq, input_batch, output_batch = self.net.pack_batch_no_out(batch, self.reparam_net.module.emb, self.device)
        input_seq = input_seq.cuda()
        # Get (two-layer) hidden state of encoder of samples in batch.
        # enc = net.encode(input_seq)
        context, enc = self.reparam_net.module.encode_context(input_seq)

        net_policies = []
        net_actions = []
        net_advantages = []
        # Transform ID to embedding.
        beg_embedding = self.reparam_net.module.emb(self.beg_token)
        beg_embedding = beg_embedding.cuda()

        for idx, inp_idx in enumerate(input_batch):
            # # Test whether the input sequence is correctly transformed into indices.
            # input_tokens = [rev_emb_dict[temp_idx] for temp_idx in inp_idx]
            # print (input_tokens)
            # Get IDs of reference sequences' tokens corresponding to idx-th input sequence in batch.
            qa_info = output_batch[idx]
            # print("            Support sample %s is training..." % (qa_info['qid']))
            # print (qa_info['qid'])
            # # Get the (two-layer) hidden state of encoder of idx-th input sequence in batch.
            item_enc = self.reparam_net.module.get_encoded_item(enc, idx)
            # # 'r_argmax' is the list of out_logits list and 'actions' is the list of output tokens.
            # # The output tokens are generated greedily by using chain_argmax (using last setp's output token as current input token).
            r_argmax, actions = self.reparam_net.module.decode_chain_argmax(item_enc, beg_embedding, data.MAX_TOKENS, context[idx],
                                                             stop_at_token=self.end_token)
            # Show what the output action sequence is.
            action_tokens = []
            for temp_idx in actions:
                if temp_idx in self.rev_emb_dict and self.rev_emb_dict.get(temp_idx) != '#END':
                    action_tokens.append(str(self.rev_emb_dict.get(temp_idx)).upper())
            # Get the highest BLEU score as baseline used in self-critic.
            # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
            # Otherwise the adaptive reward is used.
            # argmax_reward = utils.calc_True_Reward(action_tokens, qa_info, self.adaptive)
            argmax_reward = random.random()
            true_reward_argmax_step.append(argmax_reward)

            # # In this case, the BLEU score is so high that it is not needed to train such case with RL.
            # if not args.disable_skip and argmax_reward > 0.99:
            #     skipped_samples += 1
            #     continue

            # In one epoch, when model is optimized for the first time, the optimized result is displayed here.
            # After that, all samples in this epoch don't display anymore.
            if not dial_shown:
                # data.decode_words transform IDs to tokens.
                log.info("Input: %s", utils.untokenize(data.decode_words(inp_idx, self.rev_emb_dict)))
                log.info("Argmax: %s, reward=%.4f", utils.untokenize(data.decode_words(actions, self.rev_emb_dict)),
                         argmax_reward)

            action_memory = list()
            for _ in range(self.samples):
                # 'r_sample' is the list of out_logits list and 'actions' is the list of output tokens.
                # The output tokens are sampled following probabilitis by using chain_sampling.
                r_sample, actions = self.reparam_net.module.decode_chain_sampling(item_enc, beg_embedding, data.MAX_TOKENS,
                                                                   context[idx], stop_at_token=self.end_token)
                total_samples += 1

                # Omit duplicate action sequence to decrease the computing time and to avoid the case that
                # the probability of such kind of duplicate action sequences would be increased redundantly and abnormally.
                duplicate_flag = False
                if len(action_memory) > 0:
                    for temp_list in action_memory:
                        if utils.duplicate(temp_list, actions):
                            duplicate_flag = True
                            break
                if not duplicate_flag:
                    action_memory.append(actions)
                else:
                    skipped_samples += 1
                    continue
                # Show what the output action sequence is.
                action_tokens = []
                for temp_idx in actions:
                    if temp_idx in self.rev_emb_dict and self.rev_emb_dict.get(temp_idx) != '#END':
                        action_tokens.append(str(self.rev_emb_dict.get(temp_idx)).upper())

                # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
                # Otherwise the adaptive reward is used.
                # sample_reward = utils.calc_True_Reward(action_tokens, qa_info, self.adaptive)
                sample_reward = random.random()
                true_reward_sample_step.append(sample_reward)

                if not dial_shown:
                    log.info("Sample: %s, reward=%.4f", utils.untokenize(data.decode_words(actions, self.rev_emb_dict)),
                             sample_reward)

                net_policies.append(r_sample)
                net_actions.extend(actions)
                # Regard argmax_bleu calculated from decode_chain_argmax as baseline used in self-critic.
                # Each token has same reward as 'sample_bleu - argmax_bleu'.
                # [x] * y: stretch 'x' to [1*y] list in which each element is 'x'.

                # # If the argmax_reward is 1.0, then whatever the sample_reward is,
                # # the probability of actions that get reward = 1.0 could not be further updated.
                # # The GAMMA is used to adjust this scenario.
                # if argmax_reward == 1.0:
                #     net_advantages.extend([sample_reward - argmax_reward + GAMMA] * len(actions))
                # else:
                #     net_advantages.extend([sample_reward - argmax_reward] * len(actions))

                net_advantages.extend([sample_reward - argmax_reward] * len(actions))

        if not net_policies:
            log.info("The net_policies is empty!")
            # TODO the format of 0.0 should be the same as loss_v.
            return 0.0, total_samples, skipped_samples

        # Data for decode_chain_sampling samples and the number of such samples is the same as args.samples parameter.
        # Logits of all output tokens whose size is N * output vocab size; N is the number of output tokens of decode_chain_sampling samples.
        policies_v = torch.cat(net_policies)
        policies_v = policies_v.cuda()
        # Indices of all output tokens whose size is 1 * N;
        actions_t = torch.LongTensor(net_actions).to(self.device)
        actions_t = actions_t.cuda()
        # All output tokens reward whose size is 1 * N;
        adv_v = torch.FloatTensor(net_advantages).to(self.device)
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
        # For the optimizer is Adam (Adaptive Moment Estimation) which is a optimizer used for gradient descent.
        # Therefore, to maximize ▽J (log_prob_actions_v) is to minimize -▽J.
        # .mean() is to calculate Monte Carlo sampling.
        loss_policy_v = -log_prob_actions_v.mean()
        loss_policy_v = loss_policy_v.cuda()

        loss_v = loss_policy_v

        self.reparam_net.reset_initial_parameter_buffer()

        return loss_v, total_samples, skipped_samples, true_reward_argmax_step, true_reward_sample_step

    def sample(self, tasks, first_order=False, dial_shown=True, epoch_count=0, batch_count=0):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        Here number of tasks is 8.
        """
        task_losses = []
        true_reward_argmax_batch = []
        true_reward_sample_batch = []
        total_samples = 0
        skipped_samples = 0
        self.net.zero_grad()
        # To get copied weights of the model for inner training.
        initial_names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
        for task in tasks:
            # names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())
            names_weights_copy = initial_names_weights_copy

            log.info("Task %s is training..." % (str(task[1]['qid'])))

            # Establish support set.
            support_set = self.establish_support_set(task, self.steps, self.weak_flag, self.train_data_support_944K)

            for step_sample in support_set:
                inner_loss, inner_total_samples, inner_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.inner_loss(step_sample, weights=names_weights_copy, dial_shown=True)
                total_samples += inner_total_samples
                skipped_samples += inner_skipped_samples
                true_reward_argmax_batch.extend(true_reward_argmax_step)
                true_reward_sample_batch.extend(true_reward_sample_step)
                log.info("        Epoch %d, Batch %d, support sample %s is trained!" % (epoch_count, batch_count, str(step_sample[1]['qid'])))

                # Get the new parameters after a one-step gradient update
                # Each module parameter is computed as parameter = parameter - step_size * grad.
                # When being saved in the OrderedDict of self.named_parameters(), it likes:
                # OrderedDict([('sigma', Parameter containing:
                # tensor([0.6931, 0.6931], requires_grad=True)), ('0.weight', Parameter containing:
                # tensor([[1., 1.],
                #         [1., 1.]], requires_grad=True)), ('0.bias', Parameter containing:
                # tensor([0., 0.], requires_grad=True)), ('1.weight', Parameter containing:
                # tensor([[1., 1.],
                #         [1., 1.]], requires_grad=True)), ('1.bias', Parameter containing:
                # tensor([0., 0.], requires_grad=True))])
                names_weights_copy = self.update_params(inner_loss, names_weights_copy=names_weights_copy, step_size=self.fast_lr, first_order=first_order)

            meta_loss, outer_total_samples, outer_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.inner_loss(task, weights=names_weights_copy, dial_shown=dial_shown)
            task_losses.append(meta_loss)
            total_samples += outer_total_samples
            skipped_samples += outer_skipped_samples
            true_reward_argmax_batch.extend(true_reward_argmax_step)
            true_reward_sample_batch.extend(true_reward_sample_step)
            log.info("Epoch %d, Batch %d, task %s is trained!" % (epoch_count, batch_count, str(task[1]['qid'])))
        meta_losses = torch.mean(torch.stack(task_losses))
        return meta_losses, total_samples, skipped_samples, true_reward_argmax_batch, true_reward_sample_batch

    # Using first-order to approximate the result of 2nd order MAML.
    def first_order_sample(self, tasks, old_param_dict = None, first_order=False, dial_shown=True, epoch_count=0, batch_count=0, mc=False):
        """
        Sample trajectories (before and after the update of the parameters) for all the tasks `tasks`.
        Here number of tasks is 1.
        """
        task_losses = []
        grads_list = {}
        true_reward_argmax_batch = []
        true_reward_sample_batch = []
        total_samples = 0
        skipped_samples = 0

        for task in tasks:
            # For each task, the initial parameters are the same, i.e., the value stored in old_param_dict.
            # temp_param_dict = self.get_net_parameter()
            if old_param_dict is not None:
                self.net.insert_new_parameter_to_layers(old_param_dict)
            # temp_param_dict = self.get_net_parameter()
            # Try to solve the bug: "UserWarning: RNN module weights are not part of single contiguous chunk of memory".
            self.net.encoder.flatten_parameters()
            self.net.decoder.flatten_parameters()
            self.net.zero_grad()
            # log.info("Task %s is training..." % (str(task[1]['qid'])))
            # Establish support set.
            support_set = self.establish_support_set(task, self.steps, self.weak_flag, self.train_data_support_944K)

            for step_sample in support_set:
                self.inner_optimizer.zero_grad()
                inner_loss, inner_total_samples, inner_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.first_order_inner_loss(step_sample, dial_shown=True, mc=mc)
                total_samples += inner_total_samples
                skipped_samples += inner_skipped_samples
                true_reward_argmax_batch.extend(true_reward_argmax_step)
                true_reward_sample_batch.extend(true_reward_sample_step)
                # log.info("        Epoch %d, Batch %d, support sample %s is trained!" % (epoch_count, batch_count, str(step_sample[1]['qid'])))
                # Inner update.
                inner_loss.backward()
                # To conduct a gradient ascent to minimize the loss (which is to maximize the reward).
                # optimizer.step updates the value of x using the gradient x.grad.
                # For example, the SGD optimizer performs:
                # x += -lr * x.grad
                self.inner_optimizer.step()
                # temp_param_dict = self.get_net_parameter()

            # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
            # It’s important to call this before loss.backward(),
            # otherwise you’ll accumulate the gradients from multiple passes.
            self.inner_optimizer.zero_grad()
            meta_loss, outer_total_samples, outer_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.first_order_inner_loss(task, dial_shown=dial_shown, mc=mc)
            task_losses.append(meta_loss)
            self.net.zero_grad()
            # Theta <- Theta - beta * (1/k) * ∑_(i=1:k)[grad(loss_theta_in/theta_in)]
            grads = torch.autograd.grad(meta_loss, self.net.grad_parameters())
            if not isinstance(grads_list, dict):
                grads_list = {}
            if isinstance(grads_list, dict) and len(grads_list) == 0:
                for (name, _), grad in zip(self.net.grad_named_parameters(), grads):
                    grads_list[name] = []
                    grads_list[name].append(grad.clone().detach())
            else:
                for (name, _), grad in zip(self.net.grad_named_parameters(), grads):
                    grads_list[name].append(grad.clone().detach())
            total_samples += outer_total_samples
            skipped_samples += outer_skipped_samples
            true_reward_argmax_batch.extend(true_reward_argmax_step)
            true_reward_sample_batch.extend(true_reward_sample_step)
            log.info("Epoch %d, Batch %d, task %s is trained!" % (epoch_count, batch_count, str(task[1]['qid'])))
        meta_losses = torch.sum(torch.stack(task_losses))
        return meta_losses, grads_list, total_samples, skipped_samples, true_reward_argmax_batch, true_reward_sample_batch

    # Using reptile to implement MAML.
    def reptile_sample(self, tasks, old_param_dict=None, dial_shown=True, epoch_count=0,
                           batch_count=0, docID_dict=None, rev_docID_dict=None, emb_dict=None, qtype_docs_range=None, random=False, monte_carlo=False):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        Here number of tasks is 1.
        """
        task_losses = []
        true_reward_argmax_batch = []
        true_reward_sample_batch = []
        total_samples = 0
        skipped_samples = 0
        running_vars = {}

        for task in tasks:
            # For each task, the initial parameters are the same, i.e., the value stored in old_param_dict.
            # temp_param_dict = self.get_net_parameter()
            if old_param_dict is not None:
                self.net.insert_new_parameter_to_layers(old_param_dict)
            # temp_param_dict = self.get_net_parameter()
            # Try to solve the bug: "UserWarning: RNN module weights are not part of single contiguous chunk of memory".
            self.net.encoder.flatten_parameters()
            self.net.decoder.flatten_parameters()
            self.net.zero_grad()
            # log.info("Task %s for reptile is training..." % (str(task[1]['qid'])))
            # Establish support set.
            # If random_flag==True, randomly select support samples in the same question category.
            if random:
                support_set = self.establish_random_support_set(task=task, N=self.steps, train_data_support_944K=self.train_data_support_944K)
            else:
                # support_set1 = self.establish_support_set(task=task, N=self.steps, weak=self.weak_flag, train_data_support_944K=self.train_data_support_944K)
                # support_set_sample, logprob_list = self.establish_support_set_by_retriever_sampling(task=task, N=self.steps, train_data_support_944K=self.train_data_support_944K,docID_dict=docID_dict, rev_docID_dict=rev_docID_dict, emb_dict=emb_dict, qtype_docs_range=qtype_docs_range)
                support_set = self.establish_support_set_by_retriever_argmax(task=task, N=self.steps, train_data_support_944K=self.train_data_support_944K, docID_dict=docID_dict, rev_docID_dict=rev_docID_dict, emb_dict=emb_dict, qtype_docs_range=qtype_docs_range)

            for step_sample in support_set:
                self.inner_optimizer.zero_grad()
                inner_loss, inner_total_samples, inner_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.first_order_inner_loss(
                    step_sample, dial_shown=True, mc=monte_carlo)
                total_samples += inner_total_samples
                skipped_samples += inner_skipped_samples
                true_reward_argmax_batch.extend(true_reward_argmax_step)
                true_reward_sample_batch.extend(true_reward_sample_step)
                # log.info("        Epoch %d, Batch %d, support sample %s is trained!" % (epoch_count, batch_count, str(step_sample[1]['qid'])))
                # Inner update.
                inner_loss.backward()
                # To conduct a gradient ascent to minimize the loss (which is to maximize the reward).
                # optimizer.step updates the value of x using the gradient x.grad.
                # For example, the SGD optimizer performs:
                # x += -lr * x.grad
                self.inner_optimizer.step()
                # temp_param_dict = self.get_net_parameter()

            # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
            # It’s important to call this before loss.backward(),
            # otherwise you’ll accumulate the gradients from multiple passes.
            self.inner_optimizer.zero_grad()
            self.net.zero_grad()
            meta_loss, outer_total_samples, outer_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.first_order_inner_loss(task, dial_shown=dial_shown, mc=monte_carlo)
            task_losses.append(meta_loss)

            meta_loss.backward()
            # To conduct a gradient ascent to minimize the loss (which is to maximize the reward).
            # optimizer.step updates the value of x using the gradient x.grad.
            # For example, the SGD optimizer performs:
            # x += -lr * x.grad
            self.inner_optimizer.step()
            # temp_param_dict = self.get_net_parameter()
            # Store the parameters of model for each meta gradient update step (each task).
            if running_vars == {}:
                for name, param in self.get_net_named_parameter().items():
                    running_vars[name] = []
                    running_vars[name].append(param.data)
            else:
                for name, param in self.get_net_named_parameter().items():
                    # Add up the value of each parameter of the model in each meta gradient update step.
                    running_vars[name].append(param.data)

            total_samples += outer_total_samples
            skipped_samples += outer_skipped_samples
            true_reward_argmax_batch.extend(true_reward_argmax_step)
            true_reward_sample_batch.extend(true_reward_sample_step)
            log.info("Epoch %d, Batch %d, task %s is trained!" % (epoch_count, batch_count, str(task[1]['qid'])))
        meta_losses = torch.sum(torch.stack(task_losses))
        return meta_losses, running_vars, total_samples, skipped_samples, true_reward_argmax_batch, true_reward_sample_batch

    # Train the retriever.
    def retriever_sample(self, tasks, old_param_dict=None, dial_shown=True, epoch_count=0,
                       batch_count=0, docID_dict=None, rev_docID_dict=None, emb_dict=None, qtype_docs_range=None, number_of_supportsets=5, mc=False, device='cpu'):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        Here number of tasks is 1.
        """
        retriever_true_reward_argmax_batch = []
        retriever_true_reward_sample_batch = []
        retriever_net_policies = []
        retriever_net_advantages = []
        retriever_total_samples = 0
        retriever_skipped_samples = 0

        for task in tasks:
            # log.info("Task %s for retriever is training..." % (str(task[1]['qid'])))

            # Argmax as baseline.
            # For each task, the initial parameters are the same, i.e., the value stored in old_param_dict.
            # temp_param_dict = self.get_net_parameter()
            if old_param_dict is not None:
                self.net.insert_new_parameter_to_layers(old_param_dict)
            # temp_param_dict = self.get_net_parameter()
            # Try to solve the bug: "UserWarning: RNN module weights are not part of single contiguous chunk of memory".
            self.net.encoder.flatten_parameters()
            self.net.decoder.flatten_parameters()
            self.net.zero_grad()

            support_set = self.establish_support_set_by_retriever_argmax(task=task, N=self.steps, train_data_support_944K=self.train_data_support_944K, docID_dict=docID_dict, rev_docID_dict=rev_docID_dict, emb_dict=emb_dict, qtype_docs_range=qtype_docs_range)
            for step_sample in support_set:
                self.inner_optimizer.zero_grad()
                inner_loss, _, _, _, _ = self.first_order_inner_loss(
                    step_sample, dial_shown=True, mc=mc)
                # log.info("        Epoch %d, Batch %d, support sample for argmax_reward %s is trained!" % (epoch_count, batch_count, str(step_sample[1]['qid'])))
                # Inner update.
                inner_loss.backward()
                self.inner_optimizer.step()
                # temp_param_dict = self.get_net_parameter()

            input_seq = self.net.pack_input(task[0], self.net.emb)
            # enc = net.encode(input_seq)
            context, enc = self.net.encode_context(input_seq)
            # # Always use the first token in input sequence, which is '#BEG' as the initial input of decoder.
            _, actions = self.net.decode_chain_argmax(enc, input_seq.data[0:1],
                                                     seq_len=data.MAX_TOKENS, context=context[0],
                                                     stop_at_token=self.end_token)
            # Show what the output action sequence is.
            action_tokens = []
            for temp_idx in actions:
                if temp_idx in self.rev_emb_dict and self.rev_emb_dict.get(temp_idx) != '#END':
                    action_tokens.append(str(self.rev_emb_dict.get(temp_idx)).upper())
            # Get the highest BLEU score as baseline used in self-critic.
            # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
            # Otherwise the adaptive reward is used.
            retriever_argmax_reward = utils.calc_True_Reward(action_tokens, task[1], self.adaptive)
            # retriever_argmax_reward = random.random()
            retriever_true_reward_argmax_batch.append(retriever_argmax_reward)

            # Reward for each sampling support set.
            # Establish support set.
            support_sets, logprob_lists, total_samples, skip_samples = self.establish_support_set_by_retriever_sampling(task=task, N=self.steps,
                                                                         train_data_support_944K=self.train_data_support_944K,
                                                                         docID_dict=docID_dict,
                                                                         rev_docID_dict=rev_docID_dict,
                                                                         emb_dict=emb_dict,
                                                                         qtype_docs_range=qtype_docs_range,number_of_supportsets=number_of_supportsets)

            if not mc:
                retriever_net_policies.append(torch.cat(logprob_lists))
            retriever_total_samples += total_samples
            retriever_skipped_samples += skip_samples
            support_set_count = 0
            net_losses = []
            for j, support_set in enumerate(support_sets):
                # For each task, the initial parameters are the same, i.e., the value stored in old_param_dict.
                # temp_param_dict = self.get_net_parameter()
                if old_param_dict is not None:
                    self.net.insert_new_parameter_to_layers(old_param_dict)
                # temp_param_dict = self.get_net_parameter()
                # Try to solve the bug: "UserWarning: RNN module weights are not part of single contiguous chunk of memory".
                self.net.encoder.flatten_parameters()
                self.net.decoder.flatten_parameters()
                self.net.zero_grad()
                for step_sample in support_set:
                    self.inner_optimizer.zero_grad()
                    inner_loss, _, _, _, _ = self.first_order_inner_loss(
                        step_sample, dial_shown=True, mc=mc)
                    # log.info("        Epoch %d, Batch %d, support sets %d, sample for sample_reward %s is trained!" % (epoch_count, batch_count, support_set_count, str(step_sample[1]['qid'])))
                    # Inner update.
                    inner_loss.backward()
                    # To conduct a gradient ascent to minimize the loss (which is to maximize the reward).
                    # optimizer.step updates the value of x using the gradient x.grad.
                    # For example, the SGD optimizer performs:
                    # x += -lr * x.grad
                    self.inner_optimizer.step()
                    # temp_param_dict = self.get_net_parameter()
                support_set_count += 1

                input_seq = self.net.pack_input(task[0], self.net.emb)
                # enc = net.encode(input_seq)
                context, enc = self.net.encode_context(input_seq)
                # # Always use the first token in input sequence, which is '#BEG' as the initial input of decoder.
                _, actions = self.net.decode_chain_argmax(enc, input_seq.data[0:1],
                                                         seq_len=data.MAX_TOKENS, context=context[0],
                                                         stop_at_token=self.end_token)
                # Show what the output action sequence is.
                action_tokens = []
                for temp_idx in actions:
                    if temp_idx in self.rev_emb_dict and self.rev_emb_dict.get(temp_idx) != '#END':
                        action_tokens.append(str(self.rev_emb_dict.get(temp_idx)).upper())
                # Get the highest BLEU score as baseline used in self-critic.
                # If the last parameter is false, it means that the 0-1 reward is used to calculate the accuracy.
                # Otherwise the adaptive reward is used.
                retriever_sample_reward = utils.calc_True_Reward(action_tokens, task[1], self.adaptive)
                # retriever_sample_reward = random.random()
                retriever_true_reward_sample_batch.append(retriever_sample_reward)
                advantages = [retriever_sample_reward - retriever_argmax_reward] * len(support_set)
                if not mc:
                    retriever_net_advantages.extend(advantages)
                else:
                    inner_adv_v = torch.FloatTensor(advantages).to(device)
                    inner_log_prob_v = logprob_lists[j].to(device)
                    inner_log_prob_adv_v = inner_log_prob_v * inner_adv_v
                    inner_log_prob_adv_v = inner_log_prob_adv_v.to(device)
                    inner_loss_policy_v = -inner_log_prob_adv_v.sum()
                    inner_loss_policy_v = inner_loss_policy_v.to(device)
                    net_losses.append(inner_loss_policy_v)

            log.info("Epoch %d, Batch %d, task %s for retriever is trained!" % (epoch_count, batch_count, str(task[1]['qid'])))

        if not mc:
            log_prob_v = torch.cat(retriever_net_policies)
            log_prob_v = log_prob_v.cuda()

            adv_v = torch.FloatTensor(retriever_net_advantages)
            adv_v = adv_v.cuda()

            log_prob_actions_v = log_prob_v * adv_v
            log_prob_actions_v = log_prob_actions_v.cuda()

            loss_policy_v = -log_prob_actions_v.mean()
            loss_policy_v = loss_policy_v.cuda()

        else:
            batch_net_losses = torch.stack(net_losses).to(device)
            loss_policy_v = batch_net_losses.mean().to(device)

        loss_v = loss_policy_v
        return loss_v, retriever_true_reward_argmax_batch, retriever_true_reward_sample_batch, retriever_total_samples, retriever_skipped_samples

    # Using reparam class to accomplish 2nd derivative for MAML.
    def reparam_sample(self, tasks, first_order=False, dial_shown=True, epoch_count=0, batch_count=0):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        """
        task_losses = []
        true_reward_argmax_batch = []
        true_reward_sample_batch = []
        total_samples = 0
        skipped_samples = 0
        self.net.zero_grad()

        for task in tasks:
            log.info("Task %s is training..." % (str(task[1]['qid'])))

            # Establish support set.
            support_set = self.establish_support_set(task, self.steps, self.weak_flag, self.train_data_support_944K)

            theta_0 = self.reparam_net.flat_param
            theta = theta_0

            for step_sample in support_set:
                inner_loss, inner_total_samples, inner_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.reparam_inner_loss(step_sample, weights=theta, dial_shown=True)
                total_samples += inner_total_samples
                skipped_samples += inner_skipped_samples
                true_reward_argmax_batch.extend(true_reward_argmax_step)
                true_reward_sample_batch.extend(true_reward_sample_step)
                log.info("        Epoch %d, Batch %d, support sample %s is trained!" % (
                epoch_count, batch_count, str(step_sample[1]['qid'])))

                # Get the new parameters after a one-step gradient update
                # Each module parameter is computed as parameter = parameter - step_size * grad.
                # When being saved in the OrderedDict of self.named_parameters(), it likes:
                # OrderedDict([('sigma', Parameter containing:
                # tensor([0.6931, 0.6931], requires_grad=True)), ('0.weight', Parameter containing:
                # tensor([[1., 1.],
                #         [1., 1.]], requires_grad=True)), ('0.bias', Parameter containing:
                # tensor([0., 0.], requires_grad=True)), ('1.weight', Parameter containing:
                # tensor([[1., 1.],
                #         [1., 1.]], requires_grad=True)), ('1.bias', Parameter containing:
                # tensor([0., 0.], requires_grad=True))])
                theta = self.reparam_update_params(inner_loss, theta=theta,
                                                        lr=self.fast_lr, first_order=first_order)

            meta_loss, outer_total_samples, outer_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.reparam_inner_loss(task, weights=theta, dial_shown=dial_shown)
            task_losses.append(meta_loss)
            total_samples += outer_total_samples
            skipped_samples += outer_skipped_samples
            true_reward_argmax_batch.extend(true_reward_argmax_step)
            true_reward_sample_batch.extend(true_reward_sample_step)
            log.info("Epoch %d, Batch %d, task %s is trained!" % (epoch_count, batch_count, str(task[1]['qid'])))
        meta_losses = torch.sum(torch.stack(task_losses))
        return meta_losses, total_samples, skipped_samples, true_reward_argmax_batch, true_reward_sample_batch

    def sampleForTest(self, task, first_order=False, dial_shown=True, epoch_count=0, batch_count=0):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        Here number of tasks is 1.
        """
        task_losses = []
        true_reward_argmax_batch = []
        true_reward_sample_batch = []
        total_samples = 0
        skipped_samples = 0
        self.net.zero_grad()
        # To get copied weights of the model for inner training.
        names_weights_copy = self.get_inner_loop_parameter_dict(self.net.named_parameters())

        log.info("Task %s is training..." % (str(task[1]['qid'])))

        # Establish support set.
        support_set = self.establish_support_set(task, self.steps, self.weak_flag, self.train_data_support_944K)

        for step_sample in support_set:
            # todo: use the similarity between the sample in support set and the task to scale the reward or loss
            #  when meta optimization.
            inner_loss, inner_total_samples, inner_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.inner_loss(step_sample, weights=names_weights_copy, dial_shown=True)
            total_samples += inner_total_samples
            skipped_samples += inner_skipped_samples
            true_reward_argmax_batch.extend(true_reward_argmax_step)
            true_reward_sample_batch.extend(true_reward_sample_step)
            # log.info("        Epoch %d, Batch %d, support sample %s is trained!" % (epoch_count, batch_count, str(step_sample[1]['qid'])))

            # Get the new parameters after a one-step gradient update
            # Each module parameter is computed as parameter = parameter - step_size * grad.
            # When being saved in the OrderedDict of self.named_parameters(), it likes:
            # OrderedDict([('sigma', Parameter containing:
            # tensor([0.6931, 0.6931], requires_grad=True)), ('0.weight', Parameter containing:
            # tensor([[1., 1.],
            #         [1., 1.]], requires_grad=True)), ('0.bias', Parameter containing:
            # tensor([0., 0.], requires_grad=True)), ('1.weight', Parameter containing:
            # tensor([[1., 1.],
            #         [1., 1.]], requires_grad=True)), ('1.bias', Parameter containing:
            # tensor([0., 0.], requires_grad=True))])
            names_weights_copy = self.update_params(inner_loss, names_weights_copy=names_weights_copy, step_size=self.fast_lr, first_order=first_order)

        if names_weights_copy is not None:
            self.net.insert_new_parameter(names_weights_copy, True)

        input_seq = self.net.pack_input(task[0], self.net.emb)
        # enc = net.encode(input_seq)
        context, enc = self.net.encode_context(input_seq)
        # # Always use the first token in input sequence, which is '#BEG' as the initial input of decoder.
        _, tokens = self.net.decode_chain_argmax(enc, input_seq.data[0:1],
                                            seq_len=data.MAX_TOKENS, context=context[0], stop_at_token=self.end_token)
        token_string = ''
        for token in tokens:
            if token in self.rev_emb_dict and self.rev_emb_dict.get(token) != '#END':
                token_string += str(self.rev_emb_dict.get(token)).upper() + ' '
        token_string = token_string.strip()

        return token_string

    def first_order_sampleForTest(self, task, old_param_dict = None, first_order=False, dial_shown=True, epoch_count=0, batch_count=0,random=False, mc=False):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        Here number of tasks is 1.
        """
        # For each task, the initial parameters are the same, i.e., the value stored in old_param_dict.
        # temp_param_dict = self.get_net_parameter()
        if old_param_dict is not None:
            self.net.insert_new_parameter_to_layers(old_param_dict)
        # temp_param_dict = self.get_net_parameter()
        # Try to solve the bug: "UserWarning: RNN module weights are not part of single contiguous chunk of memory".
        self.net.encoder.flatten_parameters()
        self.net.decoder.flatten_parameters()
        self.net.zero_grad()
        log.info("Task %s is testing..." % (str(task[1]['qid'])))

        true_reward_argmax_batch = []
        true_reward_sample_batch = []
        total_samples = 0
        skipped_samples = 0

        # Establish support set.
        if not random:
            support_set = self.establish_support_set(task, self.steps, self.weak_flag, self.train_data_support_944K)
        else:
            # log.info("Using random support set...")
            support_set = self.establish_random_support_set(task=task, N=self.steps, train_data_support_944K=self.train_data_support_944K)

        for step_sample in support_set:
            self.inner_optimizer.zero_grad()
            inner_loss, inner_total_samples, inner_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.first_order_inner_loss(step_sample, dial_shown=True, mc=mc)
            total_samples += inner_total_samples
            skipped_samples += inner_skipped_samples
            true_reward_argmax_batch.extend(true_reward_argmax_step)
            true_reward_sample_batch.extend(true_reward_sample_step)
            # log.info("        Support sample %s is trained!" % (str(step_sample[1]['qid'])))
            # Inner update.
            inner_loss.backward()
            self.inner_optimizer.step()
            # temp_param_dict = self.get_net_parameter()

        input_seq = self.net.pack_input(task[0], self.net.emb)
        # enc = net.encode(input_seq)
        context, enc = self.net.encode_context(input_seq)
        # # Always use the first token in input sequence, which is '#BEG' as the initial input of decoder.
        _, tokens = self.net.decode_chain_argmax(enc, input_seq.data[0:1],
                                            seq_len=data.MAX_TOKENS, context=context[0], stop_at_token=self.end_token)
        token_string = ''
        for token in tokens:
            if token in self.rev_emb_dict and self.rev_emb_dict.get(token) != '#END':
                token_string += str(self.rev_emb_dict.get(token)).upper() + ' '
        token_string = token_string.strip()
        return token_string

    def maml_retriever_sampleForTest(self, task, old_param_dict = None, docID_dict=None, rev_docID_dict=None, emb_dict=None, qtype_docs_range=None, steps=5, mc=False):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        Here number of tasks is 1.
        """
        # For each task, the initial parameters are the same, i.e., the value stored in old_param_dict.
        # temp_param_dict = self.get_net_parameter()
        if old_param_dict is not None:
            self.net.insert_new_parameter_to_layers(old_param_dict)
        # temp_param_dict = self.get_net_parameter()
        # Try to solve the bug: "UserWarning: RNN module weights are not part of single contiguous chunk of memory".
        self.net.encoder.flatten_parameters()
        self.net.decoder.flatten_parameters()
        self.net.zero_grad()
        log.info("Task %s is testing..." % (str(task[1]['qid'])))

        true_reward_argmax_batch = []
        true_reward_sample_batch = []
        total_samples = 0
        skipped_samples = 0

        # Establish support set.
        support_set = self.establish_support_set_by_retriever_argmax(task=task, N=steps, train_data_support_944K=self.train_data_support_944K, docID_dict=docID_dict, rev_docID_dict=rev_docID_dict, emb_dict=emb_dict, qtype_docs_range=qtype_docs_range)

        for step_sample in support_set:
            self.inner_optimizer.zero_grad()
            inner_loss, inner_total_samples, inner_skipped_samples, true_reward_argmax_step, true_reward_sample_step = self.first_order_inner_loss(step_sample, dial_shown=True, mc=mc)
            total_samples += inner_total_samples
            skipped_samples += inner_skipped_samples
            true_reward_argmax_batch.extend(true_reward_argmax_step)
            true_reward_sample_batch.extend(true_reward_sample_step)
            # Inner update.
            inner_loss.backward()
            self.inner_optimizer.step()
            # temp_param_dict = self.get_net_parameter()
            # log.info("        Support sample %s is trained!" % (str(step_sample[1]['qid'])))

        input_seq = self.net.pack_input(task[0], self.net.emb)
        # enc = net.encode(input_seq)
        context, enc = self.net.encode_context(input_seq)
        # # Always use the first token in input sequence, which is '#BEG' as the initial input of decoder.
        _, tokens = self.net.decode_chain_argmax(enc, input_seq.data[0:1],
                                            seq_len=data.MAX_TOKENS, context=context[0], stop_at_token=self.end_token)
        token_string = ''
        for token in tokens:
            if token in self.rev_emb_dict and self.rev_emb_dict.get(token) != '#END':
                token_string += str(self.rev_emb_dict.get(token)).upper() + ' '
        token_string = token_string.strip()

        # Show what the output action sequence is.
        action_tokens = []
        for temp_idx in tokens:
            if temp_idx in self.rev_emb_dict and self.rev_emb_dict.get(temp_idx) != '#END':
                action_tokens.append(str(self.rev_emb_dict.get(temp_idx)).upper())

        return token_string, action_tokens
