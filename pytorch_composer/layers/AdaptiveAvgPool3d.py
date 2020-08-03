from pytorch_composer.Layer import Layer
import numpy as np

class AdaptiveAvgPool3d(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(
                 dimension_arg,
                 other_args,
                 variables,
                 layer_type = "adaptiveavgpool3d",
                 nn = "nn.AdaptiveAvgPool3d",
                 description = "Resizing with adaptive average pooling",
                 default_args = None,
                 dimension_key = 'output_size',
                 required_args = ['output_size'],
                 kw_args = [],
                 spaces = {
                    'output_size':('list',3),
                },
        )

    # Main loop:

    # Valid permutation:

    @staticmethod
    def required_batch_rank(data_dim, data_batch_rank, args):
        # If the batch rank is in the last two position, it is
        # permuted to 0
        if len(data_dim) <= 3:
            return 0
        elif len(data_dim) >= 4:
            if len(data_dim) - 1 >= data_batch_rank and len(data_dim) - 3 <= data_batch_rank:
                return 0
        return None

    # Valid input dimensions:

    # Creating the layer:

    @property
    def valid_args(self):
        args = self.active_args
        args["output_size"] = self.int_to_tuple(args["output_size"])
        return args

    # Updating the block object:
    
    def reshape_dims(self, input_dims):
        if len(input_dims) <= 3:
            return input_dims[:1] + [1]*(5-len(input_dims)) + input_dims[1:]
        elif len(input_dims) > 5:
            return [input_dims[0]] + [int(np.prod(input_dims[1:-2]))] + input_dims[-2:]
        return input_dims
            
    def update_model(self, model):
        input_dims = model.block.output_dim.copy()
        out = input_dims.copy()
        if len(input_dims) < 4:
            out = [input_dims[0]] + list(self.valid_args["output_size"])
        else:
            out[-3] = self.valid_args["output_size"][0]
            out[-2] = self.valid_args["output_size"][1]
            out[-1] = self.valid_args["output_size"][2]
        model.block.variables.update_x(out)
        
        self.add_reusable_layer(model.block)
        if len(input_dims) not in [4,5]:
            model.block.forward_function[-1][-1] = "(x.view{})".format(tuple(self.reshape_dims(input_dims)))
            model.block.forward_function[-1].append(".view{}".format(tuple(model.block.output_dim)))
