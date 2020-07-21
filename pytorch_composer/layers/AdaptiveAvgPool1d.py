from pytorch_composer.Layer import Layer
import numpy as np


class AdaptiveAvgPool1d(Layer):
    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(
                 dimension_arg,
                 other_args,
                 variables,
                 layer_type = "adaptiveavgpool1d",
                 nn = "nn.AdaptiveAvgPool1d",
                 description = "Resizing with adaptive average pooling",
                 default_args = None,
                 dimension_key = 'output_size',
                 required_args = ['output_size'],
                 kw_args = [],
                 spaces = {
                    "output_size":"n",
                }
        )

    # Main loop:

    # Valid permutation:

    @staticmethod
    def required_batch_rank(data_dim, data_rank, args):
        return 0

    # Valid input dimensions:

    # Creating the layer:

    @property
    def valid_args(self):
        args = self.active_args
        if not 'output_size' in args:
            args['output_size'] = self.input_dim[-1]
        if isinstance(args["output_size"], tuple):
            args["output_size"] = args["output_size"][0]
        return args        

    # Updating the block object:
    
    @property
    def reshape_dims(self):
        input_dims = self.input_dim.copy()
        if len(input_dims) <= 3:
            return [1]*(3 - len(input_dims)) + input_dims
        else:
            return [input_dims[0]] + [int(np.prod(input_dims[1:-1]))] + [input_dims[-1]]
        
    def update_model(self, model):
        out = self.input_dim.copy()
        out[-1] = self.valid_args["output_size"]
        model.block.variables.update_x(out)
        self.add_reusable_layer(model.block)
        if len(self.input_dim) != 3:
            model.block.forward_function[-1][-1] = "(x.view{})".format(tuple(self.reshape_dims))
            model.block.forward_function[-1].append(".view{}".format(tuple(self.output_dim)))
