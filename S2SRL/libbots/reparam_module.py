import torch
import torch.nn as nn
import warnings
import types
from collections import namedtuple
from contextlib import contextmanager

# A module is a container from which layers, model subparts (e.g. BasicBlock in resnet in torchvision) and models should inherit.
# Why should they? Because the inheritance from nn.Module allows you to call methods like to("cuda:0"), .eval(), .parameters() or register hooks easily.
class ReparamModule(nn.Module):
    def __init__(self, module):
        super(ReparamModule, self).__init__()
        self.module = module
        self.saved_views = []

        param_infos = []
        params = []
        param_numels = []
        param_shapes = []
        # self.modules() uses the depth-first traversal method to returns an iterator over all modules in the network,
        # including net itself, net's children, children of net's children, etc.
        # 'm' itself is the module where all information of the module is included.
        # Note
        # self.modules() is traversing the items in self._modules.items().
        # When a module is assigned to self.module, the self._modules will automatically add the module accordingly.
        for m in self.modules():
            print('#############################')
            print(type(m))
            # print(m)
            print(m.named_parameters(recurse=False))
            # named_parameters(prefix='', recurse=True):
            # Returns an iterator over module parameters,
            # yielding both the name of the parameter as well as the parameter itself.
            # recurse (bool) – if True, then yields parameters of this module and all submodules.
            # Otherwise, yields only parameters that are direct members of this module.
            for n, p in m.named_parameters(recurse=False):
                # param_infos: all the module and its weight/bias in the format of (module name, weight/bias name);
                # params: the parameters (weights/biases themselves) in each module, stored in the 'Tensor' data structure.
                # param_numels: the number of elements in each module;
                # param_shapes: the shape of the elements in each module;
                if p is not None:
                    param_infos.append((m, n))
                    # tensor.detach() creates a tensor that shares storage with tensor that does not require grad.
                    # tensor.clone()creates a copy of tensor that imitates the original tensor's requires_grad field.
                    # You should use detach() when attempting to remove a tensor from a computation graph,
                    # and clone as a way to copy the tensor while still keeping the copy as a part of the computation graph it came from.
                    # So, as you said, x.clone() maintains the connection with the computation graph.
                    # That means, if you use the new cloned tensor, and derive the loss from the new one,
                    # the gradients of that loss can be computed all the way back even beyond the point where the new tensor was created.
                    # However, if you detach the new tensor, as it is done in the case of .new_tensor(),
                    # then the gradients will only be computed from loss backward up to that new tensor but not further than that.
                    params.append(p.detach())
                    # numel(self, input): Returns the total number of elements in the `input` tensor.
                    param_numels.append(p.numel())
                    # p.size(): get the shape of p.
                    param_shapes.append(p.size())

        # dtype: int, float, or other types;
        assert len(set(p.dtype for p in params)) <= 1, \
            "expects all parameters in module to have same dtype"

        # store the info for unflatten
        # List to tuple;
        self._param_infos = tuple(param_infos)
        self._param_numels = tuple(param_numels)
        self._param_shapes = tuple(param_shapes)

        # flatten
        # p.reshape(-1): from shape(2,3) to shape(6,) a series of elements, [[1 2 3], [4 5 6]] -> [1 2 3 4 5 6]
        # torch.cat(tensors, dim=0, out=None) → Tensor: torch.cat(tensors, dim=0, out=None) → Tensor;
        # Concatenates the given sequence of seq tensors in the given dimension.
        # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        # All parameters are stored in the flat_param in a series of elements.
        flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in params], 0))
        # register_parameter(name, param): Adds a parameter to the module. The parameter can be accessed as an attribute using given name.
        self.register_parameter('flat_param', flat_param)
        self.param_numel = flat_param.numel()
        # Note: why delete detached (copied with no gradients) tensors in the net?
        del params

        # deregister the names as parameters;
        # Note: delete all weights and bias in self._param_infos, module._parameters and module.weights/biases as well;
        for m, n in self._param_infos:
            #  Delete weight/bias from each module's _parameters.
            delattr(m, n)

        # register the views as plain attributes: add weights/biases into module and of course also self._param_infos, but NOT module._parameters.
        self._unflatten_param(self.flat_param)

        # now buffers
        # they are not reparametrized. just store info as (module, name, buffer)
        # Buffers are named tensors that do not update gradients at every step like parameters.
        # The good thing is when you save the model, all params and buffers are saved,
        # and when you move the model to or off the CUDA params and buffers will go as well.
        # For example, BatchNorm’s running_mean is not a parameter, but is part of the persistent state.
        # An example for a buffer can be found in _BatchNorm module where the running_mean,
        # running_var and num_batches_tracked are registered as buffers and updated by accumulating statistics of data forwarded through the layer.
        # This is in contrast to weight and bias parameters that learns an affine transformation of the data using regular SGD optimization.
        buffer_infos = []
        for m in self.modules():
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    buffer_infos.append((m, n, b))

        self._buffer_infos = tuple(buffer_infos)
        self._traced_self = None

    def trace(self, example_input, **trace_kwargs):
        assert self._traced_self is None, 'This ReparamModule is already traced'

        if isinstance(example_input, torch.Tensor):
            example_input = (example_input,)
        example_input = tuple(example_input)
        example_param = (self.flat_param.detach().clone(),)
        example_buffers = (tuple(b.detach().clone() for _, _, b in self._buffer_infos),)

        self._traced_self = torch.jit.trace_module(
            self,
            inputs=dict(
                _forward_with_param=example_param + example_input,
                _forward_with_param_and_buffers=example_param + example_buffers + example_input,
            ),
            **trace_kwargs,
        )

        # replace forwards with traced versions
        self._forward_with_param = self._traced_self._forward_with_param
        self._forward_with_param_and_buffers = self._traced_self._forward_with_param_and_buffers
        return self

    def clear_views(self):
        for m, n in self._param_infos:
            setattr(m, n, None)  # This will set as plain attr

    def _apply(self, *args, **kwargs):
        if self._traced_self is not None:
            self._traced_self._apply(*args, **kwargs)
            return self
        return super(ReparamModule, self)._apply(*args, **kwargs)

    def _unflatten_param(self, flat_param):
        # According to the value recorded in the self._param_numels and self._param_shapes
        # to split the flat_param into parameter matrices tuple.
        # torch.split(tensor, split_size_or_sections, dim=0): Splits the tensor into chunk along certain dimension.
        # t.view(s): range shape of elements in t based on s from _param_shapes.
        ps = (t.view(s) for (t, s) in zip(flat_param.split(self._param_numels), self._param_shapes))
        # Insert the weights and biases back into the self._param_infos.
        # note: what's the difference between deleted attributes and added attributes?
        # setarr: add weights/biases into module and of course also self._param_infos, but NOT module._parameters.
        for (m, n), p in zip(self._param_infos, ps):
            setattr(m, n, p)  # This will set as plain attr

    # By using '@contextmanager', when using 'with' to call 'unflattened_param', the codes before 'yield' will be executed,
    # which means the original parameters (theta) will be stored in the saved_views.
    # Then the input flat_param will insert into parameters of modules in the net.
    # When 'unflattened_param' is closed, the codes after 'yield' will be executed to again insert the original parameters 'theta' into the net.
    @contextmanager
    def unflattened_param(self, flat_param):
        # getattr(object, name[, default]) -> value
        # Get a named attribute from an object; getattr(x, 'y') is equivalent to x.y.
        # Get a list of tensors (weights/biases).
        saved_views = [getattr(m, n) for m, n in self._param_infos]
        self._unflatten_param(flat_param)
        yield
        # Why not just `self._unflatten_param(self.flat_param)`?
        # 1. because of https://github.com/pytorch/pytorch/issues/17583
        # 2. slightly faster since it does not require reconstruct the split+view
        #    graph
        for (m, n), p in zip(self._param_infos, saved_views):
            setattr(m, n, p)

    @contextmanager
    def replaced_buffers(self, buffers):
        for (m, n, _), new_b in zip(self._buffer_infos, buffers):
            setattr(m, n, new_b)
        yield
        for m, n, old_b in self._buffer_infos:
            setattr(m, n, old_b)

    def _forward_with_param_and_buffers(self, flat_param, buffers, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            with self.replaced_buffers(buffers):
                return self.module(*inputs, **kwinputs)

    def _forward_with_param(self, flat_param, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            return self.module(*inputs, **kwinputs)

    def forward(self, *inputs, flat_param=None, buffers=None, **kwinputs):
        if flat_param is None:
            flat_param = self.flat_param
        if buffers is None:
            return self._forward_with_param(flat_param, *inputs, **kwinputs)
        else:
            return self._forward_with_param_and_buffers(flat_param, tuple(buffers), *inputs, **kwinputs)

    def _set_param_and_buffers(self, flat_param, buffers, **kwinputs):
        # getattr(object, name[, default]) -> value
        # Get a named attribute from an object; getattr(x, 'y') is equivalent to x.y.
        # Get a list of tensors (weights/biases).
        self.saved_views = [getattr(m, n) for m, n in self._param_infos]
        self._unflatten_param(flat_param)
        for (m, n, _), new_b in zip(self._buffer_infos, buffers):
            setattr(m, n, new_b)

    def _set_param(self, flat_param, **kwinputs):
        # getattr(object, name[, default]) -> value
        # Get a named attribute from an object; getattr(x, 'y') is equivalent to x.y.
        # Get a list of tensors (weights/biases).
        self.saved_views = [getattr(m, n) for m, n in self._param_infos]
        self._unflatten_param(flat_param)

    def set_parameter_buffer(self, flat_param=None, buffers=None, **kwinputs):
        if flat_param is None:
            flat_param = self.flat_param
        if buffers is None:
            self._set_param(flat_param, **kwinputs)
        else:
            self._set_param_and_buffers(flat_param, tuple(buffers), **kwinputs)

    # Reset the parameters and buffers to initial value of the model.
    def reset_initial_parameter_buffer(self):
        for (m, n), p in zip(self._param_infos, self.saved_views):
            setattr(m, n, p)
        for m, n, old_b in self._buffer_infos:
            setattr(m, n, old_b)


