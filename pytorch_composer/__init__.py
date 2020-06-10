import math
import numpy as np
from collections import Counter, defaultdict

from .Layer import Layer

from .layers import Linear
from .layers import Conv2d
from .layers import MaxPool2d
from .layers import Reshape
from .layers import Flat
from .layers import Relu
from .layers import AdaptiveAvgPool1d
from .layers import AdaptiveAvgPool2d
from .layers import AdaptiveAvgPool3d
from .layers import RNN
from .layers import permute

# The classes for all the types of layers are saved in a dictionary. The key is the name of the classes.
# Example:
# layers["Linear"] will return the Linear class.
layers = {x.__name__: x for x in Layer.__subclasses__()}

from pytorch_composer.CodeSection import CodeSection
from .Classifier import Classifier

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

class Block():
    """
    Object that parses a sequence, creates and holds the code of the model as a list. 
    """

    def __init__(
            self,
            count=None,
            groups=None,
            layers_list=None,
            forward_function=None,
            input_dim = None,
            output_dim = None,
            hidden=None,
            batch_rank = 0
            ):
        if count is None:
            count = Counter()
        if groups is None:
            groups = defaultdict(list)
        if layers_list is None:
            layers_list = []
        if forward_function is None:
            forward_function = []
        if hidden is None:
            hidden = []
        
        self.count = count
        self.groups = groups
        self.layers_list = layers_list
        self.forward_function = forward_function
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.batch_rank = batch_rank
        
        # Recurrent layers
        self.recurrent = 0
           
    @property
    def code(self):
        ''' Code of the model in a list of lists'''
        vars_ = "x"
        hidden_init = []
        if self.hidden:
            vars_hidden = ", ".join([x[0] for x in self.hidden])
            vars_ += ", " + vars_hidden
            # Hidden init:
            hidden_init = [["def", "initHidden(self):"]]
            for var in self.hidden:
                hidden_init.append(["code","{} = torch.zeros{}".format(var[0], var[1])])
            hidden_init.append(["code","return " + vars_hidden])
                
        return [
            ["class", "${model_name}(nn.Module):"],
            ["def", "__init__(self):"],
            ["code", "super(${model_name}, self).__init__()"],
            *self.layers_list,
            ["def", "forward(self, " + vars_ +"):"],
            ["comment", "Input dimensions : ", tuple(self.input_dim)],
            *self.forward_function,
            ["comment", "Output dimensions : ", tuple(self.output_dim)],
            ["code", "return " + vars_],
            *hidden_init
        ]
        
    def __str__(self):
        return self.write(self.code)
    
    @classmethod
    def create(cls, sequence, input_dim, batch_rank):
        ''' The sequence provided is parsed and turned into a block object'''
        block = cls(input_dim = list(input_dim), output_dim = list(input_dim), batch_rank = batch_rank)
        for entry in sequence:
            layer_type, dimension_arg, other_args = parse_entry(entry)
            # Make the input dims fit with the  requested layer:
            permutation = layers[layer_type].permutation(block.output_dim, block.batch_rank, other_args)
            if permutation:
                block = block.update("permute", permutation)
            valid_input_dims = layers[layer_type].valid_input_dims(block.output_dim, block.batch_rank)
            if valid_input_dims is not block.output_dim:
                block = block.update("Reshape", valid_input_dims)
            # Add code for the requested sequence
            block = block.update(layer_type, dimension_arg, other_args)
        return block

    def update(self, layer_type, dimension_arg = None, other_args = None):
        layer = layers[layer_type].create(self.output_dim, dimension_arg, other_args, self.batch_rank)
        block = layer.update_block(self)
        block.output_dim = layer.output_dim
        block.batch_rank = layer.batch_rank

        return block

    def add_forward(self, line):
        self.forward_function.append(line)

    def add_layer(self, line):
        self.layers_list.append(line)
        
    @staticmethod
    def write(code):
        "Converts the code saved as a list to a string with the proper indentation"
        str_ = ""
        last = ""
        for line in code:
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
        
class Model(CodeSection):
    ''' CodeSection object built from sequence'''
    def __init__(self, sequence, data):
        if data is None:
            variables = {"x":[("x0",[4,3,32,32],0)]} # Default variable
        elif isinstance(data, list):
            variables = {"x":[("x0",data,0)]}        
        elif isinstance(data, CodeSection):
            variables = data.variables
        else:
            raise ValueError
        self.block = Block.create(sequence, variables["x"][0][1],variables["x"][0][2])
        variables["x"][0] = (variables["x"][0][0], self.block.output_dim, self.block.batch_rank)
        variables["hidden"] = self.block.hidden
        defaults = {"model_name":"Net","batch_rank":self.block.batch_rank}
        imports = set((
            "torch",
            "torch.nn as nn",
            "torch.nn.functional as F"
        ))
        super().__init__(None, defaults, variables, imports)

    @property
    def template(self):
        return str(self.block)    
    
    def set_output(self,output_dim):
        if output_dim is not block.output_dim:
            reshape = layers["Reshape"].create(self.block.output_dim, output_dim)
            self.block = self.block.update(reshape)        
    
class Code:
    def __init__(self, code_sections):
        self.sections = code_sections
        
    @property
    def str_(self):
        imports = set()
        code = ""
        for section in self.sections:
            imports = imports.union(section.imports)
            code += section.code
        return CodeSection.write_imports(imports) +"\n" + code
    
    def __str__(self):
        return self.str_
    
    def __repr__(self):
        return self.str_
    
    def execute(self):
        exec(str(self), globals(), globals())
    
    