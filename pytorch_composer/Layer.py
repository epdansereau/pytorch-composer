import math
import warnings


class Layer():
    """ Holds values representing a single layer in the model"""

    def __init__(
            self,
            layer_type,
            layer_args=None,
            input_dim=None,
            output_dim=None,
            nn=None,
            description=None):
        self.layer_type = layer_type
        self.args = layer_args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nn = nn
        self.description = description

    def __bool__(self):
        return bool(self.layer_type)

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

    def ints_to_tuples(self, arguments, keys):
        for key in keys:
            arguments[key] = self._int_to_tuple(arguments[key])
        return arguments

    def tuples_to_ints(self, arguments, keys):
        for key in keys:
            arguments[key] = self._tuple_to_int(arguments[key])
        return arguments

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

    def active_args(self, dimension_arg, other_args):
        args = {}
        if dimension_arg is not None:
            args[self.dimension_key] = dimension_arg
            if self.dimension_key in other_args:
                if other_args[self.dimension_key] != args[self.dimension_key]:
                    warnings.warn(
                        "In {} layer, the argument {} was defined twice. The value in" +
                        " the argument dictionary will be ignored.".format(
                            self.layer_type,
                            self.dimension_key))
                other_args.pop(self.dimension_key)
            for arg in other_args:
                if arg not in self.default_args:
                    warnings.warn(
                        "Unknown argument {} in {} layer will be ignored".format(
                            self.dimension_key, self.layer_type))
            for arg in self.default_args:
                if arg in other_args:
                    args[arg] = other_args[arg]
                else:
                    args[arg] = self.default_args[arg]
        return args

    def write_args(self, args):
        required = []
        for required_arg in self.required_args:
            required.append(args[required_arg])
        args_code = ("{}" + ", {}" * (len(required) - 1)).format(*required)
        for kw_arg in self.kw_args:
            if args[kw_arg] != self.default_args[kw_arg]:
                args_code += ", {}={}".format(kw_arg, args[kw_arg])
        return args_code

    def add_unique_layer(self, block):
        block.count[self.layer_type] += 1
        ind = block.count[self.layer_type]
        block.add_layer(["layer", "self.{}".format(
            self.layer_type), ind, " = {}({})".format(self.nn, self.args)])
        block.add_forward(["comment",
                           "{}: ".format(self.description),
                           tuple(self.input_dim),
                           " -> ",
                           tuple(self.output_dim)])
        block.add_forward(
            ["forward", "x = ", "self.{}{}".format(self.layer_type, ind), "(x)"])
        return block

    def add_reusable_layer(self, block):
        is_new_group = not(self.args in block.groups[self.layer_type])
        if is_new_group:
            block.groups[self.layer_type].append(self.args)
        ind = block.groups[self.layer_type].index(self.args) + 1
        if is_new_group:
            block.add_layer(["layer", "self.{}".format(
                self.layer_type), ind, " = {}({})".format(self.nn, self.args)])
        block.add_forward(["comment",
                           "Reshaping the data: ",
                           tuple(self.input_dim),
                           " -> ",
                           tuple(self.output_dim)])
        block.add_forward(
            ["forward", "x = ", "self.{}{}".format(self.layer_type, ind), "(x)"])
        return block
