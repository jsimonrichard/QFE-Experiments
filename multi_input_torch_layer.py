# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the classes and functions for integrating QNodes with the Torch Module
API."""

import contextlib
import functools
import inspect
import math
from collections.abc import Iterable
from typing import Callable, Dict, Union, Any

from pennylane import QNode

try:
    import torch
    from torch.nn import Module

    TORCH_IMPORTED = True
except ImportError:
    # The following allows this module to be imported even if PyTorch is not installed. Users
    # will instead see an ImportError when instantiating the MultiInputTorchLayer.
    from unittest.mock import Mock

    Module = Mock
    TORCH_IMPORTED = False

def format_shapes(shapes):
    return {
        weight: (tuple(size) if isinstance(size, Iterable) else () if size == 1 else (size,))
        for weight, size in shapes.items()
    }


class MultiInputTorchLayer(Module):
    

    def __init__(
        self,
        qnode: QNode,
        input_shapes: dict,
        weight_shapes: dict,
        init_method: Union[Callable, Dict[str, Union[Callable, Any]]] = None,
        # FIXME: Cannot change type `Any` to `torch.Tensor` in init_method because it crashes the
        # tests that don't use torch module.
    ):
        if not TORCH_IMPORTED:
            raise ImportError(
                "MultiInputTorchLayer requires PyTorch. PyTorch can be installed using:\n"
                "pip install torch\nAlternatively, "
                "visit https://pytorch.org/get-started/locally/ for detailed "
                "instructions."
            )
        super().__init__()

        # Format and validate shapes
        self.input_shapes = format_shapes(input_shapes)
        assert len(input_shapes.keys()) > 0
        weight_shapes = format_shapes(weight_shapes)

        # validate the QNode signature, and convert to a Torch QNode.
        # TODO: update the docstring regarding changes to restrictions when tape mode is default.
        self._signature_validation(qnode, self.input_shapes, weight_shapes)
        self.qnode = qnode
        self.qnode.interface = "torch"

        self.qnode_weights: Dict[str, torch.nn.Parameter] = {}

        self._init_weights(init_method=init_method, weight_shapes=weight_shapes)
        self._initialized = True

    def _signature_validation(self, qnode: QNode, input_shapes: dict, weight_shapes: dict):
        sig = inspect.signature(qnode.func).parameters

        # if self.input_arg not in sig:
        #     raise TypeError(
        #         f"QNode must include an argument with name {self.input_arg} for inputting data"
        #     )

        # if self.input_arg in set(weight_shapes.keys()):
        #     raise ValueError(
        #         f"{self.input_arg} argument should not have its dimension specified in "
        #         f"weight_shapes"
        #     )

        param_kinds = [p.kind for p in sig.values()]

        if inspect.Parameter.VAR_POSITIONAL in param_kinds:
            raise TypeError("Cannot have a variable number of positional arguments")

        if inspect.Parameter.VAR_KEYWORD not in param_kinds and set(input_shapes.keys()) | set(weight_shapes.keys()) != set(sig.keys()):
            raise ValueError("Must specify a shape for every non-input parameter in the QNode")

    def forward(self, **input_kwargs):  # pylint: disable=arguments-differ
        """Evaluates a forward pass through the QNode based upon input data and the initialized
        weights.

        Args:
            **kwargs: inputs according to input_shapes

        Returns:
            tensor: output data
        """

        # Validate the input shapes
        has_batch_dim = None
        batch_dim = 1
        for key, value in input_kwargs.items():
            if has_batch_dim is None:
                if value.shape == torch.Size(self.input_shapes[key]): 
                    has_batch_dim = False
                elif value.shape[:-1] == torch.Size(self.input_shapes[key]):
                    has_batch_dim = True
                    batch_dim = value.shape[-1]
            else:
                if value.shape == torch.Size(self.input_shapes[key]) and not has_batch_dim:
                    continue
                elif value.shape[:-1] == torch.Size(self.input_shapes[key]) and has_batch_dim and value.shape[-1] == batch_dim:
                    continue
                
                raise ValueError("Inconsistent input data dimensions")

        output_dtype = input_kwargs.values()[0].dtype

        # Here we assume that only one of the dimentions if any (the last one) is a batch dimension

        # # in case the input has more than one batch dimension
        # if has_batch_dim:
        #     batch_dims = inputs.shape[:-1]
        #     inputs = torch.reshape(inputs, (-1, inputs.shape[-1]))

        # calculate the forward pass as usual
        kwargs = {
            **input_kwargs,
            **{arg: weight.to(output_dtype) for arg, weight in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)

        if isinstance(res, torch.Tensor):
            return res.type(output_dtype)

        # if len(x.shape) > 1:
        #     res = [torch.reshape(r, (x.shape[0], -1)) for r in res]

        # return torch.hstack(res).type(self.output_dtype)

        # # reshape to the correct number of batch dims
        # if has_batch_dim:
        #     results = torch.reshape(results, (*batch_dims, *results.shape[1:]))

        return res

    def construct(self, args, kwargs):
        """Constructs the wrapped QNode on input data using the initialized weights.

        This method was added to match the QNode interface. The provided args
        must contain a single item, which is the input to the layer. The provided
        kwargs is unused.

        Args:
            args (tuple): A tuple containing one entry that is the input to this layer
            kwargs (dict): Unused
        """
        x = args[0]
        kwargs = {
            self.input_arg: x,
            **{arg: weight.data.to(x) for arg, weight in self.qnode_weights.items()},
        }
        self.qnode.construct((), kwargs)


    def __getattr__(self, item):
        """If the qnode is initialized, first check to see if the attribute is on the qnode."""
        if self._initialized:
            with contextlib.suppress(AttributeError):
                return getattr(self.qnode, item)

        return super().__getattr__(item)

    def __setattr__(self, item, val):
        """If the qnode is initialized and item is already a qnode property, update it on the qnode, else
        just update the torch layer itself."""
        if self._initialized and item in self.qnode.__dict__:
            setattr(self.qnode, item, val)
        else:
            super().__setattr__(item, val)

    def _init_weights(
        self,
        weight_shapes: Dict[str, tuple],
        init_method: Union[Callable, Dict[str, Union[Callable, Any]], None],
    ):
        r"""Initialize and register the weights with the given init_method. If init_method is not
        specified, weights are randomly initialized from the uniform distribution on the interval
        [0, 2π].

        Args:
            weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
                their corresponding shapes
            init_method (Union[Callable, Dict[str, Union[Callable, torch.Tensor]], None]): Either a
                `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`__ function for
                initializing the QNode weights or a dictionary specifying the callable/value used for
                each weight. If not specified, weights are randomly initialized using the uniform
                distribution over :math:`[0, 2 \pi]`.
        """

        def init_weight(weight_name: str, weight_size: tuple) -> torch.Tensor:
            """Initialize weights.

            Args:
                weight_name (str): weight name
                weight_size (tuple): size of the weight

            Returns:
                torch.Tensor: tensor containing the weights
            """
            if init_method is None:
                init = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
            elif callable(init_method):
                init = init_method
            elif isinstance(init_method, dict):
                init = init_method[weight_name]
                if isinstance(init, torch.Tensor):
                    if tuple(init.shape) != weight_size:
                        raise ValueError(
                            f"The Tensor specified for weight '{weight_name}' doesn't have the "
                            + "appropiate shape."
                        )
                    return init
            return init(torch.Tensor(*weight_size)) if weight_size else init(torch.Tensor(1))[0]

        for name, size in weight_shapes.items():
            self.qnode_weights[name] = torch.nn.Parameter(
                init_weight(weight_name=name, weight_size=size)
            )

            self.register_parameter(name, self.qnode_weights[name])

    def __str__(self):
        detail = "<Quantum Torch Layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__

    _initialized = False
