from pytorch_composer.Layer import Layer
from pytorch_composer.adaptive_resizing import resizing_args
from pytorch_composer.layers.AdaptiveAvgPool1d import AdaptiveAvgPool1d
from pytorch_composer.layers.AdaptiveAvgPool2d import AdaptiveAvgPool2d
from pytorch_composer.layers.AdaptiveAvgPool3d import AdaptiveAvgPool3d

from copy import deepcopy

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
        
        
    def update_model(self, model):
        output_size = self.valid_args["output_size"]
        input_dim = model.block.output_dim.copy()
        res_dims = resizing_args(input_dim, list(output_size))
        if len(res_dims) == 3:
            # Pooling layer needed
            reshape_dim, pool_args, out = res_dims
            model.block.variables.update_x(reshape_dim)
            if len(reshape_dim) == 3:
                pool = AdaptiveAvgPool1d
            if len(reshape_dim) == 4:
                pool = AdaptiveAvgPool2d
            if len(reshape_dim) == 5:
                pool = AdaptiveAvgPool3d
            pool = pool(tuple(pool_args), None, model)
            model.block.add_forward(
                ["reshape", "x = x.view{}".format(tuple(reshape_dim))])
            pool._update(model)
            model.block.variables.update_x(out)
        else:
            model.block.variables.update_x(res_dims[-1])        

        model.block.add_forward(
            ["reshape", "x = x.view{}".format(tuple(model.block.output_dim))])

