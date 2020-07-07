from pytorch_composer.Layer import Layer
import numpy as np
from pytorch_composer.CodeSection import Vars


class permute(Layer):

    def __init__(self, variables):
        self.layer_type = "permute"
        self.input_dim = variables.output_dim.copy()
        self.reshape_dim = None
        self.permutation = None
        self.variables = variables


    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    @classmethod
    def create(cls, permutation, other_args = None, variables = None):
        if variables is None:
            variables = Vars({})
            variables.add_variable("x",cls.default_dim(),cls.default_batch_rank())
        layer = cls(variables)
        layer.dimension_arg = permutation
        layer.other_args = other_args
        layer.permutation = permutation
        if len(layer.input_dim) > len(permutation):
            layer.reshape_dim = layer.input_dim[:-2] + \
                [np.prod(layer.input_dim[len(permutation) - 1:])]
        else:
            layer.reshape_dim = layer.input_dim + \
                [1] * (len(permutation) - len(layer.input_dim))
        layer.update_variables(permutation)
        return layer
    

    def update_variables(self, permutation):
        out = self.reshape_dim.copy()
        for i, v in zip(permutation, self.reshape_dim):
            out[i] = v
        self.variables.update_x(out, permutation[self.batch_rank])

    # Updating the block object:

    def update_block(self, block):
        if self.input_dim != self.output_dim:
            if len(self.input_dim) != len(self.output_dim):
                block.add_forward(
                    ["reshape", "x = x.view{}".format(tuple(self.reshape_dim))])
            block.add_forward(
                ["permute", "x = x.permute{}".format(tuple(self.permutation))])
