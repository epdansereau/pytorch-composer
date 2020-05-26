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
            "padding": 0,
            "dilation": 1,
        }
        self.required_args = ["in_channels", "out_channels", "kernel_size"]
        self.kw_args = ["stride", "padding", "dilation", "groups", "bias", "padding_mode"]
   
    @classmethod
    def create(cls, input_dim, dimension_arg, other_args):
        layer = cls(input_dim)
        if not(dimension_arg):
            dimension_arg = 2
        if not("stride" in other_args):
            layer.default_args["stride"] = dimension_arg
        real = layer.real_args(layer.default_args, other_args)
        real = {i:layer._int_to_tuple(v) for i,v in real.items()}
        dimension_arg = layer._int_to_tuple(dimension_arg)
        missing_padding_0, missing_padding_1 = layer._missing_padding(
            input_dim[2], input_dim[3], dimension_arg, real["padding"])
        real["padding"] = (
            real["padding"][0] +
            missing_padding_0,
            real["padding"][1] +
            missing_padding_1)
        h_out, w_out = layer._conv_dim(input_dim[2], input_dim[3], real["padding"], real[
                              "dilation"], dimension_arg, real["stride"])
        layer.output_dim = [input_dim[0], input_dim[1], h_out, w_out]
        real = {i:layer._tuple_to_int(v) for i,v in real.items()}
        corrected_args = layer.args_out(layer.default_args, real)
        required_args_out = [layer._tuple_to_int(dimension_arg)]
        for arg in layer.required_args:
            if arg in corrected_args:
                corrected_args.pop(arg)
        layer.args = layer.write_args(required_args_out, corrected_args)
        return layer

    def update_block(self, block):
        return self._add_reusable_layer(block)