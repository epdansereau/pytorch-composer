from pytorch_composer.Layer import Layer
import numpy as np
from pytorch_composer.CodeSection import Vars


class permute(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(dimension_arg, other_args, variables)
        self.layer_type = "permute"
        self.reshape_dim = None
        
        self.dimension_key = "dims"
        self.required_args = ["dims"]
        self.kw_args = []
        self.spaces = {
            "dims":"list"
        }


    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    @classmethod
    def create(cls, permutation, other_args = None, variables = None):
        layer = cls(permutation, other_args, variables)
        layer.permutation = layer.valid_args["dims"]
        if len(layer.input_dim) > len(layer.permutation):
            layer.reshape_dim = layer.input_dim[:-2] + \
                [np.prod(layer.input_dim[len(layer.permutation) - 1:])]
        else:
            layer.reshape_dim = layer.input_dim + \
                [1] * (len(layer.permutation) - len(layer.input_dim))
        return layer
    
    @staticmethod
    def sloppy_permute_arg(arg, len_):
        return list(np.array(arg).argsort()[:len_]) + list(np.arange(len(arg), len_))

    @property
    def valid_args(self):
        args = self.active_args
        args["dims"] = self.sloppy_permute_arg(args["dims"],len(self.input_dim))
        return args

    def update_variables(self):
        out = self.reshape_dim.copy()
        for i, v in zip(self.permutation, self.reshape_dim):
            out[i] = v
        self.variables.update_x(out, self.permutation[self.batch_rank])

    # Updating the block object:

    def update_block(self, block):
        if self.input_dim != self.output_dim:
            if len(self.input_dim) != len(self.output_dim):
                block.add_forward(
                    ["reshape", "x = x.view{}".format(tuple(self.reshape_dim))])
            block.add_forward(
                ["permute", "x = x.permute{}".format(tuple(self.permutation))])
