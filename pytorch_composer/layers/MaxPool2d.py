from pytorch_composer.Layer import Layer
import math


class MaxPool2d(Layer):

    def __init__(self, variables):
        self.layer_type = "maxpool2d"
        self.args = None
        self.input_dim = variables.output_dim.copy()
        self.nn = "nn.MaxPool2d"
        self.description = "Pooling layer (2d max)"
        self.variables = variables

        # Arguments:
        self.default_args = {
            "kernel_size": 2,
            "stride": None,
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False,
        }
        self.dimension_key = "kernel_size"
        self.required_args = ["kernel_size"]
        self.kw_args = [
            "stride",
            "padding",
            "dilation",
            "return_indices",
            "ceil_mode"]

    # Main loop:

    # Valid permutation:
    @staticmethod
    def required_batch_rank(data_dim, data_rank, args):
        return 0

    # Valid input dimensions:
    @staticmethod
    def valid_input_dims(input_dims, batch_rank):
        return Layer.change_rank(input_dims, 4, batch_rank)

    # Creating the layer:

    def get_valid_args(self, args):
        if args["stride"] is None:
            args["stride"] = args["kernel_size"]
        to_tuple = ["padding", "kernel_size"]
        args = self.ints_to_tuples(args, to_tuple)
        missing_padding_0, missing_padding_1 = self._missing_padding(
            self.input_dim[2], self.input_dim[3], args["kernel_size"], args["padding"])
        args["padding"] = (
            args["padding"][0] +
            missing_padding_0,
            args["padding"][1] +
            missing_padding_1)
        args = self.tuples_to_ints(args, to_tuple)
        return args

    def update_variables(self, args):
        to_tuple = ["padding", "dilation", "kernel_size", "stride"]
        args_ = self.ints_to_tuples(args.copy(), to_tuple)
        h_out, w_out = self._conv_dim(self.input_dim[2], self.input_dim[3], args_["padding"], args_[
            "dilation"], args_["kernel_size"], args_["stride"])
        self.variables.update_x([self.input_dim[0], self.input_dim[1], h_out, w_out])

    # Updating the block object:

    def update_block(self, block):
        self.add_reusable_layer(block)
