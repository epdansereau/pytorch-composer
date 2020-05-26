from pytorch_composer.Layer import Layer
import math


class MaxPool2d(Layer):
    valid_input_dim = "channels_2d"

    def __init__(self, input_dim):
        self.layer_type = "maxpool2d"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = None
        self.nn = "nn.MaxPool2d"
        self.description = "Pooling layer (2d max)"

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

    def get_valid_args(self, args, input_dim):
        if args["stride"] is None:
            args["stride"] = args["kernel_size"]
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
        return [input_dim[0], input_dim[1], h_out, w_out]

    @classmethod
    def create(cls, input_dim, dimension_arg, other_args):
        layer = cls(input_dim)
        args = layer.active_args(dimension_arg, other_args)
        args = layer.get_valid_args(args, input_dim)
        layer.output_dim = layer.get_output_dim(input_dim, args)
        layer.args = layer._write_args(args)
        return layer

    def update_block(self, block):
        return self._add_reusable_layer(block)
