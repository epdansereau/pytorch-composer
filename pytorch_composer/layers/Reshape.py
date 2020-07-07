from pytorch_composer.Layer import Layer
from pytorch_composer.adaptive_resizing import resizing_args
from pytorch_composer.layers.AdaptiveAvgPool1d import AdaptiveAvgPool1d
from pytorch_composer.layers.AdaptiveAvgPool2d import AdaptiveAvgPool2d
from pytorch_composer.layers.AdaptiveAvgPool3d import AdaptiveAvgPool3d

from pytorch_composer.CodeSection import Vars

class Reshape(Layer):

    def __init__(self, variables):
        self.layer_type = "reshape"
        self.input_dim = variables.output_dim.copy()
        self.reshape_dim = None
        self.pool = None
        self.variables = variables

    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    @classmethod
    def create(cls, dimension_arg, other_args = None, variables = None):
        if variables is None:
            variables = Vars({})
            variables.add_variable("x",cls.default_dim(),cls.default_batch_rank())        
        layer = cls(variables)
        layer.dimension_arg = dimension_arg
        layer.other_args = other_args
        res_dims = resizing_args(layer.input_dim, list(dimension_arg))
        if len(res_dims) == 3:
            # Pooling layer needed
            layer.reshape_dim, pool_args, out = res_dims
            layer.variables.update_x(layer.reshape_dim)
            if len(layer.reshape_dim) == 3:
                pool = AdaptiveAvgPool1d
            if len(layer.reshape_dim) == 4:
                pool = AdaptiveAvgPool2d
            if len(layer.reshape_dim) == 5:
                pool = AdaptiveAvgPool3d
            layer.pool = pool.create(tuple(pool_args), None, layer.variables.copy())
            layer.variables.update_x(out)
        else:
            layer.variables.update_x(res_dims[-1])
        return layer

    # Updating the block object:

    def update_block(self, block):
        if self.pool is not None:
            block.add_forward(
                ["reshape", "x = x.view{}".format(tuple(self.reshape_dim))])
            self.pool.update_block(block)
        block.add_forward(
            ["reshape", "x = x.view{}".format(tuple(self.output_dim))])
