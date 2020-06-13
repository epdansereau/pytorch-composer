from pytorch_composer.Layer import Layer
import math


class Conv2d(Layer):

    def __init__(self, variables):
        self.layer_type = "conv2d"
        self.args = None
        self.input_dim = variables.output_dim.copy()
        self.nn = "nn.Conv2d"
        self.description = "Convolution layer (2d)"
        self.variables = variables

        # Arguments:
        self.default_args = {
            "in_channels": None,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "bias": True,
            "padding_mode": "zeros"
        }
        self.dimension_key = "out_channels"
        self.required_args = ["in_channels", "out_channels", "kernel_size"]
        self.kw_args = [
            "stride",
            "padding",
            "dilation",
            "groups",
            "bias",
            "padding_mode"]

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
        args["in_channels"] = self.input_dim[1]
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
        h_out, w_out = self._conv_dim(self.input_dim[2],
                                      self.input_dim[3], args_["padding"], args_[
            "dilation"], args_["kernel_size"], args_["stride"])
        self.variables.update_x([self.input_dim[0], args_["out_channels"], h_out, w_out])

    # Updating the block object:

    def update_block(self, block):
        return self.add_unique_layer(block)
