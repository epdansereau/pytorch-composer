import math
import warnings
import pytorch_composer
from pytorch_composer.CodeSection import Vars


class Layer():
    """
    Parent class to all layer type classes. Holds values representing a single layer in the model.
    All child classes are expected to have at least self.input_dim (list) and self.output_dim (list)
    attributes, and a classmethod create() that handles instanciation.
    """

    def __init__(self,
                 dimension_arg = None,
                 other_args = None,
                 variables = None,
                 layer_type = None,
                 nn = None,
                 description = None,
                 default_args = None,
                 dimension_key = "",
                 required_args = None,
                 kw_args = None,
                 spaces = None
                ):
        if other_args is None:
            other_args = {}
        if variables is None:
            variables = Vars({})
            variables.add_variable("x",self.default_dim(),self.default_batch_rank())
            
        self.dimension_arg = dimension_arg
        self.other_args = other_args
        
        self.layer_type = layer_type
        self.variables = variables
        self.args = None
        self.input_dim = variables.output_dim.copy()
        self.nn = nn
        self.description = description
        
        if default_args is None:
            default_args = {}
        if required_args is None:
            required_args = []
        if kw_args is None:
            kw_args = []
        if spaces is None:
            spaces = {}
            
        # Arguments:
        self.default_args = default_args
        self.dimension_key = dimension_key
        self.required_args = required_args
        self.kw_args = kw_args
        
        self.spaces = spaces
        
    def __call__(self, variables = None, batch_rank = None):
        return self.layer_model(variables, batch_rank)
    
    def __repr__(self):
        return f'''{self.__class__.__name__}: 
variables:{self.variables}
dimension_arg:{self.dimension_key}:{str(self.dimension_arg)}:
other_args:{self.other_args}
active_args:{self.active_args}
valid_args:{self.valid_args}'''
    
    @property
    def layer_model(self):
        # Single layer model
        return pytorch_composer.Model([[self.__class__.__name__,
                                               self.dimension_arg,
                                               self.other_args]], self.variables)
    
    def get_batch_code(self):
        return self.layer_model.get_batch_code()

    # Main loop:

    # Valid permutation:
    
    @property
    def batch_rank(self):
        return self.variables["x"][0].batch_rank
    
    @property
    def output_dim(self):
        return self.variables["x"][0].dim

    @staticmethod
    def required_batch_rank(data_dim, data_rank, args):
        return None

    @classmethod
    def permutation(cls, data_dim, data_rank, args=None):
        required = cls.required_batch_rank(data_dim, data_rank, args)
        if required is None:
            return False
        if data_rank == required:
            return False
        else:
            perm = [i for i in range(max(len(data_dim), required + 1))]
            perm = perm[:data_rank] + perm[data_rank + 1:]
            perm = perm[:required] + [data_rank] + perm[required:]
            return perm

    # Valid input dimensions:

    @staticmethod
    def valid_input_dims(input_dims, batch_rank):
        # Takes in any input_dims(list of ints), and returns valid input dimensions(list of ints that can
        # contain negative numbers for inference).
        return input_dims

    @staticmethod
    def change_rank(input_dims, new_rank, batch_rank):
        # Takes in any list and returns a list of the desired rank.
        # The dimension at index 0 is assumed to be the batch size and is not changed, unless the new rank
        # is one.
        # The default behavior is to set the dimension at index 2  to 1, and the following dimensions to
        # -1, meaning that they will be infered to be of equal sizes using adaptive resizing.
        # Examples:
        #     input: [4,5,6,7],1     output: [-1]
        #     input: [4,5,6,7],2     output: [4,-1]
        #     input: [4,5,6,7],3     output: [4,1,-1]
        #     input: [4,5,6,7],5     output: [4,1,-1,-1,-1]
        if len(input_dims) == new_rank:
            return input_dims
        elif new_rank == 1:
            return [-1]
        else:
            # TD:if batch_rank > len(new_rank)
            new_dims = [-1] * new_rank
            new_dims[batch_rank] = input_dims[batch_rank]
            if batch_rank == 0 and new_rank > 2:
                new_dims[1] = 1
            return new_dims

    # Creating the layer:

    @classmethod
    def create(cls, dimension_arg, other_args = None, variables = None):
        layer = cls(dimension_arg, other_args, variables)
        layer.update_variables()
        layer.args = layer.write_args(layer.valid_args)
        return layer
    
    @classmethod
    def default_dim(self):
        return [4,3,32,32]
    
    @classmethod
    def default_batch_rank(self):
        return 0   

    @property
    def active_args(self):
        # Joins self.dimension_arg and self.other_args in the same dict.
        # Fills missing arguments with the defaults values
        other_args = self.other_args.copy()
        if (self.dimension_key in self.other_args) and (self.dimension_arg is not None):
            if other_args[self.dimension_key] != self.dimension_arg:
                warnings.warn(("In {} layer, the argument {} was defined twice. The value in" +
                " the argument dictionary will be ignored.").format(
                    self.layer_type,
                    self.dimension_key))
        if self.dimension_arg is not None:
            other_args[self.dimension_key] = self.dimension_arg
        args = self.required_args + self.kw_args
        for arg in list(other_args):
            if arg not in args:
                warnings.warn(
                    "Unknown argument {} in {} layer will be ignored".format(
                        arg, self.layer_type))
                other_args.pop(arg)
        return {**self.default_args, **other_args}

    @property
    def valid_args(self):
        return self.active_args

    def update_variables(self):
        pass

    def write_args(self, args):
        # Converts the layer's arguments into code.
        # Input: args(dict)
        # Output: args_code(str)

        all_args = []
        required = []
        for required_arg in self.required_args:
            required.append(args[required_arg])
        all_args.append(", ".join([str(x) for x in required]))
        kw_code = []
        for kw_arg in self.kw_args:
            if args[kw_arg] != self.default_args[kw_arg]:
                kw_code.append("{}={}".format(kw_arg, args[kw_arg]))
        if kw_code:
            all_args.append(", ".join(kw_code))
        all_args = ", ".join(all_args)
        return all_args

    # Updating the block object:

    def update_block(self, block):
        # The Block.update function creates a Layer object, and calls this
        # function to update itself.
        pass

    def add_unique_layer(self, block, hidden=False):
        # Updates the block when the layer should not be reused in the forward function (i.e. when the
        # layer has weights).
        block.count[self.layer_type] += 1
        ind = block.count[self.layer_type]
        block.add_layer(["layer", "self.{}".format(
            self.layer_type), ind, " = {}({})".format(self.nn, self.args)])
        block.add_forward(["comment",
                           "{}: ".format(self.description),
                           tuple(self.input_dim),
                           " -> ",
                           tuple(self.output_dim)])
        if hidden:
            block.add_forward(
                ["forward", "x, h{} = ".format(ind),
                 "self.{}{}".format(self.layer_type, ind),
                 "(x, h{})".format(ind)])
        else:
            block.add_forward(
                ["forward", "x = ", "self.{}{}".format(self.layer_type, ind), "(x)"])

    def add_reusable_layer(self, block):
        # Updates the block layers of the same type can be reused in the
        # forward function (i.e. activation functions).
        is_new_group = not(self.args in block.groups[self.layer_type])
        if is_new_group:
            block.groups[self.layer_type].append(self.args)
        ind = block.groups[self.layer_type].index(self.args) + 1
        if is_new_group:
            block.add_layer(["layer", "self.{}".format(
                self.layer_type), ind, " = {}({})".format(self.nn, self.args)])
        block.add_forward(["comment",
                           "{}: ".format(self.description),
                           tuple(self.input_dim),
                           " -> ",
                           tuple(self.output_dim)])
        block.add_forward(
            ["forward", "x = ", "self.{}{}".format(self.layer_type, ind), "(x)"])

    # Various utility functions used by layer classes:

    # Functions to compute valid arguments of 2d layers:

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

    # Various functions to deal with int/tuple conversions
    def int_to_tuple(self, value):
        # if value is an int, returns it two times in a tuple
        if isinstance(value, int):
            return (value, value)
        else:
            return value

    def tuple_to_int(self, value):
        # collapses tuples into single ints when possible (expects len of 2)
        if isinstance(value, tuple):
            if value[0] == value[1]:
                return value[0]
        return value

    def ints_to_tuples(self, arguments, keys):
        for key in keys:
            arguments[key] = self.int_to_tuple(arguments[key])
        return arguments

    def tuples_to_ints(self, arguments, keys):
        for key in keys:
            arguments[key] = self.tuple_to_int(arguments[key])
        return arguments