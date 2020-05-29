import math
import numpy as np
from sympy import factorint

from .Layer import Layer
from .Block import Block

from .layers import Linear
from .layers import Conv2d
from .layers import MaxPool2d
from .layers import Reshape
from .layers import Flat
from .layers import Relu
from .layers import AdaptiveAvgPool1d
from .layers import AdaptiveAvgPool2d
from .layers import AdaptiveAvgPool3d

# The classes for all the types of layers are saved in a dictionary. The key is the name of the classes.
# Example:
#    layers["Linear"] will return the Linear class.
layers = {x.__name__: x for x in Layer.__subclasses__()}


class CodeSection():
    """ Holds the code ready to be printed """

    def __init__(self, code_text, input_dim, output_dim):
        self.code_text = code_text
        self.input_dim = input_dim
        self.output_dim = output_dim

    def formatted(self):
        "Converts the code saved as a list to a string with the proper indentation"
        str_ = ""
        last = ""
        for line in self.code_text:
            if line[0] == "class":
                if last:
                    str_ += "\n"
                str_ += "\nclass " + "".join([str(x) for x in line[1:]]) + "\n"
            elif line[0] == "def":
                if last != "class":
                    str_ += "\n"
                str_ += " " * 4 + "def " + \
                    "".join([str(x) for x in line[1:]]) + "\n"
            elif line[0] == "comment":
                str_ += " " * 8 + "# " + \
                    "".join([str(x) for x in line[1:]]) + "\n"
            else:
                str_ += " " * 8 + "".join([str(x) for x in line[1:]]) + "\n"
            last = line[0]
        return str_

    def print_formatted(self):
        print(self.formatted())


def parse_entry(entry):
    '''
    Parses [layer_type (str), (dimension_arg (int or tuple)), (other_args(dict))]
    Valid input formats are : [str], [str, int or tuple], [str, dict], [str, int or tuple, dict]
    Output: layer_type(str), dimension(int or tuple or None), other_args(dict or None)
    '''
    try:
        assert isinstance(entry[0], str)
        assert len(entry) <= 3
        if entry[1:]:
            assert type(entry[1]) in [int, tuple, dict]
            if entry[2:]:
                assert not isinstance(
                    entry[1], dict) and isinstance(
                    entry[2], dict)
    except BaseException:
        raise TypeError(
            "Invalid entry in list of entries.\n" +
            "The expected format is [layer_type (str), (dimension_arg (int or tuple)), (other_args(dict))]")
    layer_type = entry[0]
    dimension = None
    other_args = {}
    if len(entry) > 1:
        if type(entry[1]) in [int, tuple]:
            dimension = entry[1]
        if isinstance(entry[-1], dict):
            other_args = entry[-1]
    return layer_type, dimension, other_args


def write_model(input_dim, sequence):
    """
    Writes valid pytorch code for a model, given an arbitrary sequence and input dimensions.
    Input: input_dim(list or tuple), sequence(list of lists)
    Output: CodeSection object
    """
    input_dim = list(input_dim)
    data_dim = input_dim.copy()
    block = Block()

    for entry in sequence:
        layer_type, dimension_arg, other_args = parse_entry(entry)
        valid_input_dims = layers[layer_type].valid_input_dims(data_dim)
        if valid_input_dims is not data_dim:
            reshape = layers["Reshape"].create(data_dim, valid_input_dims)
            data_dim = reshape.output_dim
            block = block.update(reshape)
        layer = layers[layer_type].create(data_dim, dimension_arg, other_args)
        data_dim = layer.output_dim
        block = block.update(layer)

    code = [
        ["class", "Net(nn.Module):"],
        ["def", "__init__(self):"],
        ["code", "super(Net, self).__init__()"],
        *block.layers_list,
        ["def", "forward(self, x):"],
        ["comment", "Input dimensions : ", tuple(input_dim)],
        *block.forward_function,
        ["comment", "Output dimensions : ", tuple(data_dim)],
        ["code", "return x"]
    ]

    return CodeSection(code, input_dim, data_dim)
