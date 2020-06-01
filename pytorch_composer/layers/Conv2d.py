from pytorch_composer.Layer import Layer
import math


class Conv2d(Layer):

    def __init__(self, input_dim):
        self.layer_type = "conv2d"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = None
        self.nn = "nn.Conv2d"
        self.description = "Convolution layer (2d)"

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

    @staticmethod
    def valid_input_dims(input_dims):
        return Layer.change_rank(input_dims, 4)

    def get_valid_args(self, args, input_dim):
        to_tuple = ["padding", "kernel_size"]
        args = self.ints_to_tuples(args, to_tuple)
        missing_padding_0, missing_padding_1 = self._missing_padding(
            input_dim[2], input_dim[3], args["kernel_size"], args["padding"])
        args["padding"] = (
            args["padding"][0] +
            missing_padding_0,
            args["padding"][1] +
            missing_padding_1)
        args = self.tuples_to_ints(args, to_tuple)
        return args

    def get_output_dim(self, input_dim, args):
        to_tuple = ["padding", "dilation", "kernel_size", "stride"]
        args_ = self.ints_to_tuples(args.copy(), to_tuple)
        h_out, w_out = self._conv_dim(input_dim[2], input_dim[3], args_["padding"], args_[
            "dilation"], args_["kernel_size"], args_["stride"])
        return [input_dim[0], args_["out_channels"], h_out, w_out]

    @classmethod
    def create(cls, input_dim, dimension_arg, other_args=None):
        if other_args is None:
            other_args = {}
        layer = cls(input_dim)
        args = layer.active_args(dimension_arg, other_args)
        args["in_channels"] = input_dim[1]
        args["out_channels"] = dimension_arg
        args = layer.get_valid_args(args, input_dim)
        layer.output_dim = layer.get_output_dim(input_dim, args)
        layer.args = layer.write_args(args)
        return layer

    def update_block(self, block):
        return self.add_unique_layer(block)
