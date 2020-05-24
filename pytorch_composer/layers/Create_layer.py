import math


class Create_layer():
    '''
    Subscriptable class that returns functions to compute output dimensions of a layer.
    For example:
        entry = ["linear, 50]
        create = Create_layer()
        create["linear"]((1,32), entry)
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

    def conv2d(self, input_dim, layer):
        ''' The c_out dimension (which is the second dimension) is changed by the convolutional layer
        to the value of the out_channels arguement.
        https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        '''

        c_out, args_in = self._parse_entry(
            layer, "out_channels", required=True)
        args_ = args_in.copy()

        # inserting default values into the arguements, and turning existing
        # values to tuples
        default_args = {
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1)
        }
        for arg, val in default_args.items():
            if not(arg in args_):
                args_[arg] = val
            args_[arg] = self._int_to_tuple(args_[arg])

        # adding padding if the input dims are smaller than the kernel size
        missing_padding_0, missing_padding_1 = self._missing_padding(
            input_dim[2], input_dim[3], args_["kernel_size"], args_["padding"])
        args_["padding"] = (
            args_["padding"][0] +
            missing_padding_0,
            args_["padding"][1] +
            missing_padding_1)
        args_in["padding"] = self._tuple_to_int(args_["padding"])

        # calculating the output of width and height given the parameters

        h_out, w_out = self._conv_dim(input_dim[2], input_dim[3], args_["padding"], args_[
                                      "dilation"], args_["kernel_size"], args_["stride"])

        output_dim = [input_dim[0], c_out, h_out, w_out]
        # turning the kernel_size back from tuple to int for cleaner code:
        kernel_size = self._tuple_to_int(args_["kernel_size"])
        layer_args = "{}, {}, kernel_size={}".format(
            input_dim[1], output_dim[1], kernel_size)
        if "kernel_size" in args_in:
            args_in.pop("kernel_size")
        for arg, arg_value in args_in.items():
            layer_args += ", {}={}".format(arg, arg_value)
        return "conv2d", layer_args, input_dim, output_dim, "nn.Conv2d", "Convolution layer (2d)"

    def maxpool2d(self, input_dim, layer):

        kernel_size, args_in = self._parse_entry(layer, "kernel_size")
        args_ = args_in.copy()
        # if no kernel_size value is provided, it will default to 2
        # inserting a kernel_size provided as int or tuple into the dict:
        if kernel_size:
            kernel_size = self._int_to_tuple(kernel_size)
            args_["kernel_size"] = kernel_size

        default_args = {
            "kernel_size": (2, 2),
            "padding": (0, 0),
            "dilation": (1, 1)
        }
        for arg, val in default_args.items():
            if not(arg in args_):
                args_[arg] = val
            args_[arg] = self._int_to_tuple(args_[arg])

        # by default, the stride is the same size as the kernel size
        if not("stride" in args_):
            args_["stride"] = args_["kernel_size"]
        args_["stride"] = self._int_to_tuple(args_["stride"])

        # adding padding if the input dims are smaller than the kernel size
        missing_padding_0, missing_padding_1 = self._missing_padding(
            input_dim[2], input_dim[3], args_["kernel_size"], args_["padding"])
        args_["padding"] = (
            args_["padding"][0] +
            missing_padding_0,
            args_["padding"][1] +
            missing_padding_1)
        args_in["padding"] = self._tuple_to_int(args_["padding"])

        # computing output dims
        h_out, w_out = self._conv_dim(input_dim[2], input_dim[3], args_["padding"], args_[
                                      "dilation"], args_["kernel_size"], args_["stride"])

        output_dim = [input_dim[0], input_dim[1], h_out, w_out]
        # turning the kernel_size back from tuple to int for cleaner code:
        kernel_size = args_["kernel_size"]
        args_.pop("kernel_size")
        if isinstance(kernel_size, tuple):
            if kernel_size[0] == kernel_size[0]:
                kernel_size = kernel_size[0]
        layer_args = "kernel_size={}".format(kernel_size)
        if "kernel_size" in args_in:
            args_in.pop("kernel_size")
        for arg, arg_value in args_in.items():
            layer_args += ", {}={}".format(arg, arg_value)
        return "maxpool2d", layer_args, input_dim, output_dim, "nn.MaxPool2d", "Pooling layer (2d max)"

    def linear(self, input_dim, layer):
        out_features, args_ = self._parse_entry(
            layer, "out_features", required=True)
        output_dim = input_dim.copy()
        output_dim[-1] = out_features
        layer_args = "{}, {}".format(input_dim[-1], out_features)
        for arg, arg_value in args_.items():
            layer_args += ", {}={}".format(arg, arg_value)
        return "linear", layer_args, input_dim, output_dim, "nn.Linear", "Linear layer"

    def flat(self, input_dim, layer):
        # No transformation here (occurs during the reshaping step)
        return "flat", None, input_dim, input_dim, None, "Flatenning the data"

    def relu(self, input_dim, layer):
        return "relu", None, input_dim, input_dim, None, "Relu activation"


create = Create_layer()
