from pytorch_composer.Layer import Layer
from pytorch_composer.adaptive_resizing import resizing_args
from pytorch_composer.layers.AdaptiveAvgPool1d import AdaptiveAvgPool1d
from pytorch_composer.layers.AdaptiveAvgPool2d import AdaptiveAvgPool2d
from pytorch_composer.layers.AdaptiveAvgPool3d import AdaptiveAvgPool3d

class Reshape(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(
                 dimension_arg,
                 other_args,
                 variables,
                 layer_type = "reshape",
                 dimension_key = "output_size",
                 required_args = ['output_size'],
                 spaces = {
                    'output_size':"list",
                 }
        )
        self.reshape_dim = None
        self.pool = None
        
    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    @classmethod
    def create(cls, dimension_arg, other_args = None, variables = None):       
        layer = cls(dimension_arg, other_args, variables)
        output_size = layer.valid_args["output_size"]
        res_dims = resizing_args(layer.input_dim, list(output_size))
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
