from pytorch_composer.Layer import Layer
import numpy as np


class AdaptiveAvgPool2d(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(dimension_arg, 
                         other_args,
                         variables,
                         layer_type = "adaptiveavgpool2d",
                         nn = "nn.AdaptiveAvgPool2d",
                         description = "Resizing with adaptive average pooling",
                         dimension_key = 'output_size',
                         required_args = ['output_size'],
                         spaces = {
                                'output_size':('list',2),
                            },
                        )

    # Main loop:

    # Valid permutation:

    @staticmethod
    def required_batch_rank(data_dim, data_batch_rank, args):
        # If the batch rank is in the last two position, it is
        # permuted to 0
        if len(data_dim) <= 2:
            return 0
        else:
            if len(data_dim) - 1 == data_batch_rank or len(data_dim) - 2 == data_batch_rank:
                return 0
        return None

    # Valid input dimensions:

    # Creating the layer:

    @property
    def valid_args(self):
        args = self.active_args
        args["output_size"] = self.int_to_tuple(args["output_size"])
        return args
    
    def reshape_dims(self, input_dims):
        if len(input_dims) < 3:
            return input_dims[:1] + [1]*(3-len(input_dims)) + input_dims[1:]
        elif len(input_dims) > 4:
            return [input_dims[0]] + [int(np.prod(input_dims[1:-2]))] + input_dims[-2:]
            
    def update_model(self, model):
        input_dims = model.block.output_dim.copy()
        if len(input_dims) < 3:
            out = [input_dims[0]] + list(self.valid_args["output_size"])
        else:
            out = input_dims.copy()
            out[-2] = self.valid_args["output_size"][0]
            out[-1] = self.valid_args["output_size"][1]
        model.variables.update_x(out)
        
        self.add_reusable_layer(model.block)
        if len(input_dims) not in [3,4]:
            model.block.forward_function[-1][-1] = "(x.view{})".format(tuple(self.reshape_dims(input_dims)))
            model.block.forward_function[-1].append(".view{}".format(tuple(model.block.output_dim)))
