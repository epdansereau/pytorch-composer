from .Classifier import Classifier
from pytorch_composer.CodeSection import CodeSection, SettingsDict
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

    def __init__(self, variables):
        self.count = Counter()
        self.groups = defaultdict(list)
        self.layers_list = []
        self.forward_function = []
        self.variables = variables
        self.input_dim = variables.output_dim.copy()

        self.code = None
        
    @property
    def output_dim(self):
        return self.variables.output_dim
    
    @property
    def batch_rank(self):
        return self.variables.batch_rank

    @property
    def parsed_code(self):
        ''' Code of the model in a list of lists'''
        vars_ = "x"
        hidden_init = []
        if self.variables["h"]:
            vars_hidden = ", ".join(self.variables.names("h"))
            vars_ += ", " + vars_hidden
            # Hidden init:
            hidden_init = [["def", "initHidden(self):"]]
            for var in self.variables["h"]:
                hidden_init.append(
                    ["code", "{} = torch.zeros{}".format(var.name, tuple(var.dim))])
            hidden_init.append(["code", "return " + vars_hidden])

        return [
            ["class", "${model_name}(nn.Module):"],
            ["def", "__init__(self):"],
            ["code", "super(${model_name}, self).__init__()"],
            *self.layers_list,
            ["def", "forward(self, " + vars_ + "):"],
            ["comment", "Input dimensions : ", tuple(self.input_dim)],
            *self.forward_function,
            ["comment", "Output dimensions : ", tuple(self.output_dim)],
            ["code", "return " + vars_],
            *hidden_init
        ]

    def __str__(self):
        return self.write(self.code)

    @classmethod
    def create(cls, sequence, variables):
        ''' The sequence provided is parsed and turned into a block object'''
        block = cls(variables)

        # Main loop:

        for entry in sequence:
            layer_type, dimension_arg, other_args = parse_entry(entry)
            
            # Valid permutation:

            permutation = layers[layer_type].permutation(
                block.output_dim, block.batch_rank, other_args)
            if permutation:
                block = block.update("permute", permutation)

            # Valid input dimensions:
            
            valid_input_dims = layers[layer_type].valid_input_dims(
                block.output_dim, block.batch_rank)
            if valid_input_dims is not block.output_dim:
                block = block.update("Reshape", valid_input_dims)

            # Adding the requested layer:

            block = block.update(layer_type, dimension_arg, other_args)
            
        block.code = block.parsed_code

        return block

    def update(self, layer_type, dimension_arg=None, other_args=None):
        layer = layers[layer_type].create(
            self.variables, dimension_arg, other_args)
        block = layer.update_block(self)
        block.variables = layer.variables

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
        self.sequence = sequence
        defaults = {"model_name": "Net"}
        imports = set((
            "torch",
            "torch.nn as nn",
            "torch.nn.functional as F"
        ))
        super().__init__(data, defaults = defaults, imports = imports)

    def set_default_variables(self):
        self.variables.add_variable("x",[4, 3, 32, 32],0)
        
    def set_variables(self, variables):
        super().set_variables(variables)
        self.read_sequence(self.sequence)
        
    def read_sequence(self, sequence):
        self.block = Block.create(sequence,self.variables)
        self.variables = self.block.variables
        
    @property
    def template(self):
        return str(self.block)

    def set_output(self, output_dim):
        if output_dim is not self.block.output_dim:
            self.block = self.block.update("Reshape", output_dim)
            self.block.code = self.block.parsed_code
        return self
    
class Code:
    def __init__(self, code_sections):
        self.sections = code_sections
        for section in self.sections:
            section.linked_to = self
        if len(self.sections) > 1:
            for i in range(len(self.sections))[1:]:
                self.sections[i].fit(self.sections[i-1])
                       
    def __str__(self):
        # Combining all code sections into a string:
        imports = set()
        code = ""
        for section in self.sections:
            imports = imports.union(section.imports)
            code += section.code
        return CodeSection.write_imports(imports) + "\n" + code

    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self.sections)
    
    def __setitem__(self, key, item):
        if isinstance(key, int):
            self.sections[key] = item
        elif isinstance(key, str):
            self.check_if_found(key)
            for section in self.sections:
                section._update({key:item})
        self.update()
                
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.sections[key]
        elif isinstance(key, str):
            return self.settings[key]
        else:
            raise TypeError
            
    @property
    def settings(self):
        all_settings = {}
        for section in self.sections:
            all_settings.update(section.settings)
        return SettingsDict(all_settings, self)
    
    def fit_all(self):
        if len(self.sections) > 1:
            for i in range(len(self))[1:]:
                self.sections[i].fit(self.sections[i-1])        
    
    def check_if_found(self, settings):
        if isinstance(settings, dict):
            settings = settings.keys()
        elif isinstance(settings,str):
            settings = [settings]
        not_found = set(settings) - set(self.settings.keys())
        if not_found:
            raise KeyError("Settings not found: {}".format(not_found))
    
    def update(self, settings = None):
        # Refits and updates in-place
        if settings:
            self.check_if_found(settings)
            for section in self.sections:
                section._update(settings)
        self.fit_all()

    def save(self, file_name="train.py"):
        with open(file_name, "w") as f:
            f.write(str(self))

    def execute(self):
        exec(str(self), globals(), globals())