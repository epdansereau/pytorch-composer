from pytorch_composer.Layer import Layer
import numpy as np

class permute(Layer):

    def __init__(self, input_dim):
        self.layer_type = "permute"
        self.input_dim = input_dim
        self.reshape_dim = None
        self.output_dim = None
        self.permutation = None
        
    @classmethod
    def create(cls, input_dim, permutation, other_args=None):
        if other_args is None:
            other_args = {}
        layer = cls(input_dim)
        layer.permutation = permutation
        # Finding
        if len(input_dim) > len(permutation):
            layer.reshape_dim = input_dim[:-2] + [np.prod(input_dim[len(permutation) - 1:])]
        else:
            layer.reshape_dim = input_dim + [1]*(len(permutation)-len(input_dim))
        layer.output_dim = layer.reshape_dim.copy()
        for i,v in zip(permutation,layer.reshape_dim):
            layer.output_dim[i] = v
        return layer

    def update_block(self, block):
        if self.input_dim !=  self.output_dim:
            if len(self.input_dim) != len(self.output_dim):                
                block.add_forward(
                    ["reshape", "x = x.view{}".format(tuple(self.reshape_dim))])
            block.add_forward(
                ["permute", "x = x.permute{}".format(tuple(self.permutation))])
        return block
