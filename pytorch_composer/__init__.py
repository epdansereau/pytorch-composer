import math
import numpy as np
from sympy import factorint

import pytorch_composer.layers as layers
from pytorch_composer.Layer import Layer
from pytorch_composer.Block import Block


class CodeSection():
    """ Holds the code ready to be printed """

    def __init__(self, code_text, input_dim, output_dim):
        self.code_text = code_text
        self.input_dim = input_dim
        self.output_dim = output_dim

    def formatted(self):
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

def match_output_input(function, dimension):
    """
    Returns a layer that will reshape the data if it's dimensions are incompatible with the next input.
    If the data is already compatible, an empty layer that will be ignored is returned.
    """
    assert len(dimension) >= 2
    # reshaping to (batch_size, c, h, w)
    if function in layers.channels_2d:
        # These functions expect data of the shape (batch_size, channels, height, width). If the data received
        # hasn't been resized and doesn't fit, we need to pick a reasonable
        # shape.
        if len(dimension) == 4:
            return Layer("")
        else:
            # distributing the factors between height and width
            new_dim = [dimension[0], 1, 1, 1]
            factors_count = factorint(int(np.prod(dimension[1:])))
            even = True
            for factor, count in factors_count.items():
                for _ in range(count):
                    if even:
                        new_dim[2] *= factor
                        even = False
                    else:
                        new_dim[3] *= factor
                        even = True
            return Layer("reshape", input_dim=dimension, output_dim=new_dim, description="Reshaping the data")

    # reshaping to (batch_size, x)
    elif function in layers.flat:
        if len(dimension) == 2:
            return Layer("")
        features_dim = int(np.prod(dimension[1:]))
        new_dim = [dimension[0], features_dim]
        return Layer("reshape", -1, dimension, new_dim, description="Flattening the data")

    # no reshaping
    else:
        return Layer("")


def write_model(input_dim, sequence):
    """
    Writes valid pytorch code for a model, given an arbitrary sequence and input dimensions.
    """
    data_dim = input_dim.copy()
    block = Block()

    for entry in sequence:
        layer_type = entry[0]
        reshape_layer = match_output_input(layer_type, data_dim)
        if reshape_layer:
            data_dim = reshape_layer.output_dim
            block = block.update(reshape_layer)
        layer = Layer.create(layer_type, data_dim, entry)
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
