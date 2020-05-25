import math

class Create_layer():
    '''
    Subscriptable class that returns functions to compute output dimensions of a layer.
    For example:
        create = Create_layer()
        entry = ["linear, 50]
        input_shape = (1,32)
        create["linear"](input_shape, entry)
    This would return arguements for a linear layer of shape (1,50).
    '''

    def __getitem__(self, item):
        return getattr(self, item.lower())

    def _parse_entry(self, entry, dimension_arg=None, required=False):
        '''
        Valid input formats are : [str], [str, int or tuple], [str, dict], [str, int or tuple, dict]
        '''
        assert isinstance(entry, list)
        assert len(entry) <= 3
        assert isinstance(entry[0], str)

        if len(entry) == 1:
            if required:
                raise Exception(
                    "No {} value was provided".format(dimension_arg))
            return dimension_arg, {}

        if len(entry) == 2:
            if (isinstance(entry[1], int)) or (isinstance(entry[1], tuple)):
                dimension_arg = entry[1]
                return dimension_arg, {}
            elif isinstance(entry[1], dict):
                if dimension_arg in entry[1]:
                    dimension_arg = entry[1][dimension_arg]
                elif required:
                    raise Exception(
                        "No {} value was provided".format(dimension_arg))
                return dimension_arg, entry[1]
            else:
                raise Exception(
                    "Invalid type in entry (expected int or tuple or dict in the second position)")

        if len(entry) == 3:
            if (isinstance(entry[1], int)) or (isinstance(entry[1], tuple)):
                dimension_arg = entry[1]
            else:
                raise Exception(
                    "Invalid type in entry (expected int or tuple in the second position)")
            if not isinstance(entry[2], dict):
                raise Exception(
                    "Invalid type in entry (expected dict in the third position)")
            if dimension_arg in entry[2]:
                if entry[2][dimension_arg] != entry[1]:
                    raise Exception(
                        "The {} value was defined two times".format(dimension_arg))
            return entry[1], entry[2]

    def _int_to_tuple(self, value):
        # if value is an int, returns it two times in a tuple
        if isinstance(value, int):
            return (value, value)
        else:
            return value

    def _tuple_to_int(self, value):
        # collapses tuples into single ints when possible (expects len of 2)
        if isinstance(value, tuple):
            if value[0] == value[1]:
                return value[0]
        return value

    def _conv_dim(self, h_in, w_in, padding, dilation, kernel_size, stride):
        h_out = math.floor((h_in + 2 * padding[0] - dilation[0] * (
            kernel_size[0] - 1) - 1) / stride[0] + 1)
        w_out = math.floor((w_in + 2 * padding[1] - dilation[1] * (
            kernel_size[1] - 1) - 1) / stride[1] + 1)
        return h_out, w_out

    def _missing_padding(self, height, width, kernel_size, padding):
        missing_padding_0 = kernel_size[0] - (height + 2 * padding[0])
        missing_padding_1 = kernel_size[1] - (width + 2 * padding[1])
        missing_padding_0 = math.ceil(max(0, missing_padding_0) / 2)
        missing_padding_1 = math.ceil(max(0, missing_padding_1) / 2)
        return missing_padding_0, missing_padding_1
    

    def real_args(self, default, inp):
        real = {}
        for args in default:
            if args in inp:
                real[args] = inp[args]
            else:
                real[args] = default[args]
        return real
    
    def args_out(self, default, real):
        out = {}
        for args in real:
            if args in default:
                if real[args] != default[args]:
                    out[args] = real[args]
            else:
                out[args] = real[args]
        return out
    
    def write_args(self, args, kwargs):
        args_code = ("{}" + ", {}"*(len(args) - 1)).format(*args)
        for arg, arg_value in kwargs.items():
            args_code += ", {}={}".format(arg, arg_value)
        return args_code
    
    def conv2d(self, input_dim, layer):
        dimension_arg, other_args = self._parse_entry(
            layer, "out_channels", required=True)
        default_args = {
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "dilation": 1
        }
        required_args = ["in_channels", "out_channels", "kernel_size"]
        kw_args = ["stride", "padding", "dilation", "groups", "bias", "padding_mode"]
        real = self.real_args(default_args, other_args)
        real = {i:self._int_to_tuple(v) for i,v in real.items()}
        missing_padding_0, missing_padding_1 = self._missing_padding(
            input_dim[2], input_dim[3], real["kernel_size"], real["padding"])
        real["padding"] = (
            real["padding"][0] +
            missing_padding_0,
            real["padding"][1] +
            missing_padding_1)
        h_out, w_out = self._conv_dim(input_dim[2], input_dim[3], real["padding"], real[
                              "dilation"], real["kernel_size"], real["stride"])

        output_dim = [input_dim[0], dimension_arg, h_out, w_out]
        real = {i:self._tuple_to_int(v) for i,v in real.items()}
        corrected_args = self.args_out(default_args, real)
        required_args_out = [input_dim[1], output_dim[1], real["kernel_size"]]
        for arg in required_args:
            if arg in corrected_args:
                corrected_args.pop(arg)
        layer_args = self.write_args(required_args_out, corrected_args)
        return "conv2d", layer_args, input_dim, output_dim, "nn.Conv2d", "Convolution layer (2d)"
        

    def maxpool2d(self, input_dim, layer):
        dimension_arg, other_args = self._parse_entry(layer, "kernel_size")
        if not(dimension_arg):
            dimension_arg = 2
        default_args = {
            "padding": 0,
            "dilation": 1,
        }
        if not("stride" in other_args):
            default_args["stride"] = dimension_arg
        required_args = ["kernel_size"]
        kw_args = ["stride","padding","dilation", "return_indices", "ceil_mode"]
        real = self.real_args(default_args, other_args)
        real = {i:self._int_to_tuple(v) for i,v in real.items()}
        dimension_arg = self._int_to_tuple(dimension_arg)
        missing_padding_0, missing_padding_1 = self._missing_padding(
            input_dim[2], input_dim[3], dimension_arg, real["padding"])
        real["padding"] = (
            real["padding"][0] +
            missing_padding_0,
            real["padding"][1] +
            missing_padding_1)
        h_out, w_out = self._conv_dim(input_dim[2], input_dim[3], real["padding"], real[
                              "dilation"], dimension_arg, real["stride"])
        output_dim = [input_dim[0], input_dim[1], h_out, w_out]
        real = {i:self._tuple_to_int(v) for i,v in real.items()}
        corrected_args = self.args_out(default_args, real)
        required_args_out = [self._tuple_to_int(dimension_arg)]
        for arg in required_args:
            if arg in corrected_args:
                corrected_args.pop(arg)
        layer_args = self.write_args(required_args_out, corrected_args)
        return "maxpool2d", layer_args, input_dim, output_dim, "nn.MaxPool2d", "Pooling layer (2d max)"

    def linear(self, input_dim, layer):
        dimension_arg, other_args = self._parse_entry(
            layer, "out_features", required=True)
        default_args = {}
        required_args = ["in_features", "out_features"]
        kw_args = ["bias"]
        real = self.real_args(default_args, other_args)
        corrected_args = self.args_out(default_args, real)
        output_dim = input_dim.copy()
        output_dim[-1] = dimension_arg
        for arg in required_args:
            if arg in corrected_args:
                corrected_args.pop(arg)
        required_args_out = [input_dim[-1], dimension_arg]
        layer_args = self.write_args(required_args_out, corrected_args)
        return "linear", layer_args, input_dim, output_dim, "nn.Linear", "Linear layer"
    
    def flat(self, input_dim, layer):
        # No transformation here (occurs during the reshaping step)
        return "flat", None, input_dim, input_dim, None, "Flatenning the data"

    def relu(self, input_dim, layer):
        return "relu", None, input_dim, input_dim, None, "Relu activation"


create = Create_layer()
