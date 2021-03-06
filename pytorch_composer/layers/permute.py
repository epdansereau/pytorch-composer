from pytorch_composer.Layer import Layer
import numpy as np
from pytorch_composer.CodeSection import Vars


class permute(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(
                 dimension_arg,
                 other_args,
                 variables,
                 layer_type = "permute",
                 dimension_key = "dims",
                 required_args = ["dims"],
                 spaces = {
                    "dims":"list",
                 }
        )

    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    def update_model(self, model):
        input_dim = model.block.output_dim
        out = [input_dim[x] for x in self.valid_args["dims"]]
        new_batch_rank = self.valid_args["dims"].index(model.block.batch_rank)
        model.block.variables.update_x(out, new_batch_rank)
        if input_dim != model.block.output_dim:
            model.block.add_forward(
                ["permute", "x = x.permute{}.contiguous()".format(tuple(self.valid_args["dims"]))])
    
    @staticmethod
    def sloppy_permute_arg(arg, len_):
        if len(arg)> len_:
            arg = arg[:len_]
        sorted_ = sorted(arg)
        order = []
        for i,v in enumerate(arg):
            index = sorted_.index(v)
            sorted_[index] = -1
            order.append(index)
        for i in range(len(arg),len_):
            order.append(i)    
        return order

    @property
    def valid_args(self):
        args = self.active_args
        args["dims"] = self.sloppy_permute_arg(args["dims"],len(self.input_dim))
        return args

