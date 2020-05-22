import math
import numpy as np
from sympy import factorint
from collections import Counter

class CodeSection():
    def __init__(self, code_text, input_dim, output_dim):
        self.code_text = code_text
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def print_formated(self):
        str_ = ""
        for line in self.code_text:
            if line[0] == "class":
                str_ += "class " + "".join([str(x) for x in line[1:]]) + "\n"
            elif line[0] == "def":
                str_ += "\tdef " + "".join([str(x) for x in line[1:]]) + "\n"
            elif line[0] == "comment":
                str_ += "\t\t# " + "".join([str(x) for x in line[1:]]) + "\n"
            else:
                str_ += "\t\t" + "".join([str(x) for x in line[1:]]) + "\n"
        print(str_)
        
class Layer():
    def __init__(self, layer_type, layer_args = None, input_dim = None, output_dim = None, nn = None):
        self.type = layer_type
        self.args = layer_args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nn = nn
    
    def __bool__(self):
        return bool(self.type)


class Layers_output():
    '''Subscritable class that returns functions to compute output dimensions'''
    def __getitem__(self, item):
        return getattr(self, item.lower())
    
    def _parse_entry(self, entry, dimension_arg = None, required = False):
        '''
        Valid input formats are : [str], [str, int or tuple], [str, dict], [str, int or tuple, dict]
        '''
        assert type(entry) == list
        assert len(entry) <= 3
        assert type(entry[0]) == str

        if len(entry) == 1:
            if required:
                raise Exception("No {} value was provided".format(dimension_arg))
            return dimension_arg, {}

        if len(entry) == 2:
            if (type(entry[1]) == int) or (type(entry[1]) == tuple):
                dimension_arg = entry[1]
                return dimension_arg, {}
            elif type(entry[1]) == dict:
                if dimension_arg in entry[1]:
                    dimension_arg = entry[1][dimension_arg]
                elif required:
                    raise Exception("No {} value was provided".format(dimension_arg))
                return dimension_arg, entry[1]
            else:
                raise Exception("Invalid type in entry (expected int or tuple or dict in the second position)")

        if len(entry) == 3:
            if (type(entry[1]) == int) or (type(entry[1]) == tuple):
                dimension_arg = entry[1]
            else:
                raise Exception("Invalid type in entry (expected int or tuple in the second position)")
            if type(entry[2]) != dict:
                raise Exception("Invalid type in entry (expected dict in the third position)")
            if dimension_arg in entry[2]:
                if entry[2][dimension_arg] != entry[1]:
                    raise Exception("The {} value was defined two times".format(dimension_arg))
            return entry[1], entry[2]
    
    def _int_to_tuple(self, value):
        # if value is an int, returns it two times in a tuple
        if type(value) == int:
            return (value, value)
        else:
            return value
        
    def _tuple_to_int(self, value):
        # collapses tuples into single ints when possible (expects len of 2)
        if type(value) == tuple:
            if value[0] == value[1]:
                return value[0]
        return value
    
    def _conv_dim(self, h_in, w_in, padding, dilation, kernel_size, stride):
        h_out = math.floor((h_in + 2*padding[0] - dilation[0]*(
            kernel_size[0] -1) - 1) / stride[0] + 1)
        w_out = math.floor((w_in + 2*padding[1] - dilation[1]*(
            kernel_size[1] -1) - 1) / stride[1] + 1)
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

        c_out, args_in = self._parse_entry(layer, "out_channels", required = True)
        args_ = args_in.copy()
        
        # inserting default values into the arguements, and turning existing values to tuples
        default_args = {
            "kernel_size" : (3,3),
            "stride" : (1,1),
            "padding" : (0,0),
            "dilation" : (1,1)
        }
        for arg, val in default_args.items():
            if not(arg in args_):
                args_[arg] = val
            args_[arg] = self._int_to_tuple(args_[arg])
        
        # adding padding if the input dims are smaller than the kernel size
        missing_padding_0, missing_padding_1 = self._missing_padding(input_dim[2], input_dim[3],
                                                                    args_["kernel_size"], args_["padding"])
        args_["padding"] = (args_["padding"][0] + missing_padding_0, args_["padding"][1] + missing_padding_1)
        args_in["padding"] = self._tuple_to_int(args_["padding"]) 

        # calculating the output of width and height given the parameters
        
        h_out, w_out = self._conv_dim(input_dim[2], input_dim[3], args_["padding"],
                                     args_["dilation"], args_["kernel_size"], args_["stride"])
        

        output_dim = [input_dim[0], c_out, h_out, w_out]
        # turning the kernel_size back from tuple to int for cleaner code:
        kernel_size = self._tuple_to_int(args_["kernel_size"])
        layer_args = "{}, {}, kernel_size = {}".format(input_dim[1], output_dim[1], kernel_size)
        if "kernel_size" in args_in:
            args_in.pop("kernel_size")
        for arg, arg_value in args_in.items():
            layer_args += ", {} = {}".format(arg, arg_value) 
        return Layer("conv2d", layer_args, input_dim, output_dim, "nn.Conv2d")


    def maxpool2d(self, input_dim, layer):

        kernel_size, args_in = self._parse_entry(layer, "kernel_size")
        args_ = args_in.copy()
        # if no kernel_size value is provided, it will default to 2
        # inserting a kernel_size provided as int or tuple into the dict:
        if kernel_size:
            kernel_size = self._int_to_tuple(kernel_size)
            args_["kernel_size"] = kernel_size

        default_args = {
            "kernel_size" : (2,2),
            "padding" : (0,0),
            "dilation" : (1,1)
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
        missing_padding_0, missing_padding_1 = self._missing_padding(input_dim[2], input_dim[3],
                                                                    args_["kernel_size"], args_["padding"])
        args_["padding"] = (args_["padding"][0] + missing_padding_0, args_["padding"][1] + missing_padding_1)
        args_in["padding"] = self._tuple_to_int(args_["padding"]) 
        
        # computing output dims
        h_out, w_out = self._conv_dim(input_dim[2], input_dim[3], args_["padding"],
                                     args_["dilation"], args_["kernel_size"], args_["stride"])
        

        output_dim = [input_dim[0], input_dim[1], h_out, w_out]
        # turning the kernel_size back from tuple to int for cleaner code:
        kernel_size = args_["kernel_size"]
        args_.pop("kernel_size")
        if type(kernel_size) == tuple:
            if kernel_size[0] == kernel_size[0]:
                kernel_size = kernel_size[0]
        layer_args = "kernel_size = {}".format(kernel_size)
        if "kernel_size" in args_in:
            args_in.pop("kernel_size")
        for arg, arg_value in args_in.items():
            layer_args += ", {} = {}".format(arg, arg_value) 
        return Layer("maxpool2d", layer_args, input_dim, output_dim, "nn.MaxPool2d")

    def linear(self, input_dim, layer):
        out_features, args_ = self._parse_entry(layer, "out_features", required = True)
        output_dim = input_dim.copy()
        output_dim[-1] = out_features
        layer_args = "{}, {}".format(input_dim[-1], out_features)
        for arg, arg_value in args_.items():
            layer_args += ", {} = {}".format(arg, arg_value)
        return Layer("linear", layer_args, input_dim, output_dim, "nn.Linear")
    
    def relu(self, input_dim, layer):
        return Layer("relu", None, input_dim, input_dim)
    
output = Layers_output()

class Layers_writer():
    '''Subscritable class that writes the code for each layer'''
    def __init__(self):
        self._layer_count = Counter()
        self._layer_group_count = {}
    
    def __call__(self, layers_list, forward_function, layer):
        return getattr(self, layer.type.lower())(layers_list, forward_function, layer)
    
    def _count_layers(self, layer):
        self._layer_count[layer.type] += 1
        return self._layer_count[layer.type]
    
    def _count_layers_group(self, layer):
        if layer.type in self._layer_group_count:
            if layer.args in self._layer_group_count[layer.type]:
                return self._layer_group_count[layer.type].index(layer.args), False
            else:
                self._layer_group_count[layer.type].append(layer.args)
                return len(self._layer_group_count[layer.type]), True
        else:
            self._layer_group_count[layer.type] = [layer.args]
            return 1, True
        
    def _add_unique_layer(self, layers_list, forward_function, layer):
        ind = self._count_layers(layer)
        layers_list.append(["layer", "self.{}".format(layer.type), ind," = {}({})".format(layer.nn, layer.args)])
        if layer.input_dim == layer.output_dim:
            forward_function.append(["comment","Shape stays at {}".format(layer.input_dim)])
        else:
            forward_function.append(["comment","Shape goes from {} to {}".format(
                layer.input_dim, layer.output_dim)])
        forward_function.append(["forward","x = ", "self.{}{}".format(layer.type, ind),"(x)"])
        return layers_list, forward_function
    
    def _add_reusable_layer(self, layers_list, forward_function, layer):
        ind, new_group = self._count_layers_group(layer)
        if new_group:
            layers_list.append(["layer", "self.{}".format(layer.type), ind," = {}({})".format(layer.nn,layer.args)])
        if layer.input_dim == layer.output_dim:
            forward_function.append(["comment","Shape stays at {}".format(layer.input_dim)])
        else:
            forward_function.append(["comment","Shape goes from {} to {}".format(
                layer.input_dim, layer.output_dim)])
        forward_function.append(["forward","x = ", "self.{}{}".format(layer.type, ind),"(x)"])
        return layers_list, forward_function
    
    def conv2d(self, layers_list, forward_function, layer):
        return self._add_unique_layer(layers_list, forward_function, layer)
        
    def maxpool2d(self, layers_list, forward_function, layer):
        return self._add_reusable_layer(layers_list, forward_function, layer)
        
    def linear(self, layers_list, forward_function, layer):
        return self._add_unique_layer(layers_list, forward_function, layer)

    def reshape(self, layers_list, forward_function, layer):
        forward_function.append(["comment","Reshaping the data from {} to {}:".format(layer.input_dim,
                                                                                     layer.output_dim)])
        if layer.args == -1:
            forward_function.append(["reshape","x = x.view(-1,{})".format(layer.output_dim[1])])
        else:
            forward_function.append(["reshape","x = x.view{}".format(tuple(layer.output_dim))])
        return layers_list, forward_function
    
    def relu(self, layers_list, forward_function, layer):
        forward_function.append(["reshape","x = F.relu(x)"])
        return layers_list, forward_function

def match_output_input(function, dimension):
    '''
    Reshapes the data if it's dimensions are incompatible with the next layer used.
    Output: (reshape_code(str), new_dimensions(list))
    '''
    assert len(dimension) >= 2
    channels_2d = [
        "conv2d",
        "maxpool2d"
    ]
    flat = [
        "linear"
    ]
    
    # reshaping to (batch_size, c, h, w)
    if function in channels_2d:
        # These functions expect data of the shape (batch_size, channels, height, width). If the data received
        # hasn't been resized and doesn't fit, we need to pick a reasonable shape.
        if len(dimension) == 4:
            return Layer("")
        else:
            # distributing the factors between height and width
            new_dim = [dimension[0],1,1,1]
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
            return Layer("reshape", input_dim = dimension, output_dim = new_dim)
        
    # reshaping to (batch_size, x)
    elif function in flat:
        if len(dimension) == 2:
            return Layer("")
        features_dim = int(np.prod(dimension[1:]))
        new_dim = [dimension[0], features_dim]
        return Layer("reshape", -1, dimension, new_dim)
    
    # no reshaping
    else:
        return Layer("")     
        
        
def write_model(input_dim, sequence):
    data_dim = input_dim.copy()
    write = Layers_writer()
    
    layers_list = []
    forward_function = [["comment","Input dimensions : {}".format(data_dim)]]
    
    for entry in sequence:
        layer_type = entry[0]
        reshape_layer = match_output_input(layer_type, data_dim)
        if reshape_layer:
            data_dim = reshape_layer.output_dim
            layers_list, forward_function = write(layers_list, forward_function, reshape_layer)
            # to do: add reshape
        layer = output[layer_type](data_dim, entry)
        data_dim = layer.output_dim
        layers_list, forward_function = write(layers_list, forward_function, layer)

    code = [
        ["class", "Net(nn.Module):"],
        ["def","__init__(self):"],
        ["code", "super(Net, self).__init__()"],
        *layers_list,
        ["def", "forward(self, x):"],
        *forward_function,
        ["code", "return x"]   
    ]
    
    return CodeSection(code, input_dim, data_dim)