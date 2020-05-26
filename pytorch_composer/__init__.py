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

# The classes for all the types of layers are saved in a dictionary. The key is the name of the classes.
# Example: layers["Linear"] will return a linear layer object.
layers = {x.__name__:x for x in Layer.__subclasses__()}

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
        
def parse_entry(entry):
    '''
    Parses [layer_type (str), (dimension_arg (int or tuple)), (other_args(dict))]
    Valid input formats are : [str], [str, int or tuple], [str, dict], [str, int or tuple, dict]
    '''
    try:
        assert type(entry[0]) == str
        assert len(entry) <= 3
        if entry[1:]:
            assert type(entry[1]) in [int, tuple, dict]
            if entry[2:]:
                assert type(entry[1]) != dict and type(entry[2]) == dict
    except:
        raise TypeError("Invalid entry in list of entries.\n" + \
                       "The expected format is [layer_type (str), (dimension_arg (int or tuple)), (other_args(dict))]") 
    layer_type = entry[0]
    dimension = None
    other_args = {}
    if len(entry) > 1:
        if type(entry[1]) in [int, tuple]:
            dimension = entry[1]
        if type(entry[-1]) == dict:
            other_args = entry[-1]
    return layer_type, dimension, other_args   

        
def match_output_input(valid_inputs, dimension):
    """
    Returns a layer that will reshape the data if it's dimensions are incompatible with the next input.
    If the data is already compatible, an empty layer that will be ignored is returned.
    """
    assert len(dimension) >= 2
    # reshaping to (batch_size, c, h, w)
    if valid_inputs == "channels_2d":
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
            return layers["Reshape"]("reshape", input_dim=dimension, output_dim=new_dim, description="Reshaping the data")

    # reshaping to (batch_size, x)
    elif valid_inputs == "flat":
        if len(dimension) == 2:
            return Layer("")
        features_dim = int(np.prod(dimension[1:]))
        new_dim = [dimension[0], features_dim]
        return layers["Reshape"]("reshape", -1, dimension, new_dim, description="Flattening the data")

    # no reshaping
    else:
        return Layer("")


def write_model(input_dim, sequence):
    """
    Writes valid pytorch code for a model, given an arbitrary sequence and input dimensions.
    """
    data_dim = input_dim.copy()  # to do: remove
    block = Block()

    for entry in sequence:
        layer_type, dimension_arg, other_args = parse_entry(entry)
        reshape_layer = match_output_input(layers[layer_type].valid_input_dim, data_dim)
        if reshape_layer:
            data_dim = reshape_layer.output_dim
            block = block.update(reshape_layer)
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
