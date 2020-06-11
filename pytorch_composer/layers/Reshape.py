from pytorch_composer.Layer import Layer
from pytorch_composer.adaptive_resizing import resizing_args
from pytorch_composer.layers.AdaptiveAvgPool1d import AdaptiveAvgPool1d
from pytorch_composer.layers.AdaptiveAvgPool2d import AdaptiveAvgPool2d
from pytorch_composer.layers.AdaptiveAvgPool3d import AdaptiveAvgPool3d


class Reshape(Layer):

    def __init__(self, input_dim, batch_rank):
        self.layer_type = "reshape"
        self.input_dim = input_dim
        self.output_dim = None
        self.reshape_dim = None
        self.pool = None
        self.batch_rank = batch_rank

    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    @classmethod
    def create(cls, input_dim, output_dim, other_args, batch_rank):
        if other_args is None:
            other_args = {}
        layer = cls(input_dim, batch_rank)
        args = resizing_args(input_dim, list(output_dim))
        if len(args) == 3:
            layer.reshape_dim, pool_args, layer.output_dim = args
            if len(layer.reshape_dim) == 3:
                pool = AdaptiveAvgPool1d
            if len(layer.reshape_dim) == 4:
                pool = AdaptiveAvgPool2d
            if len(layer.reshape_dim) == 5:
                pool = AdaptiveAvgPool3d

            layer.pool = pool.create(
                layer.reshape_dim, tuple(pool_args), None, 0)
        else:
            layer.output_dim = args[-1]
        return layer

    # Updating the block object:

    def update_block(self, block):
        if self.pool is not None:
            block.add_forward(
                ["reshape", "x = x.view{}".format(tuple(self.reshape_dim))])
            block = self.pool.update_block(block)
        block.add_forward(
            ["reshape", "x = x.view{}".format(tuple(self.output_dim))])
        return block
