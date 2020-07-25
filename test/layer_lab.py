from pytorch_composer import get_layer
from torch import rand

from random import randint, random, choice, sample
from pytorch_composer.get_layer import get_layer, layers_list

import pytorch_composer

from pytorch_composer.CodeSection import Vars

import torch

class RandomLayor:
    def __init__(self, layer_types):
        self.max_int = 10
        
        self.max_list = 5
        
        self.x10prob = 0.05
        self.critical_prob = 0.05
        self.none_prob = 0.25
        
        self.layer_types = layer_types
        
        self.space_dict = {
            "n":self.rand_n,
            "bool":self.rand_bool,
            "list":self.rand_list,
            "float":self.rand_float,
        }
        
    def rand_setting(self, space):
        if isinstance(space, set):
            return sample(space, 1)
        elif isinstance(space,tuple):
            return self.space_dict[space[0]](*space[1:])
        else:
            return self.space_dict[space]()
    
    def rand_float(self):
        if random() < self.critical_prob:
            return choice([0.,1.])
        else:
            return random()
    
    def rand_n(self):
        if random() < self.critical_prob:
            return 1
        else:
            return randint(1,self.max_int)
        
    def rand_bool(self):
        return choice([True,False])
    
    def choice(self,set_):
        return choice(set_)
    
    def rand_list(self, len_ = None, min_ = 1):
        if len_ is None:
            len_ = randint(min_, self.max_list)
        list_ = []
        for i in range(len_):
            list_.append(self.rand_n())
        return list_
    
    def rand(self,type_ = None):
        if type_ is None:
            type_ = self.layer_types
        if isinstance(type_, list):
            type_ = choice(self.layer_types)
        none_prob = self.none_prob
        dim_none_prob = self.none_prob
        layer = get_layer(type_)()
        other_args = {}
        for arg, space in layer.spaces.items():
            other_args[arg] = self.rand_setting(space)
        if random() > self.none_prob and layer.dimension_key:
            dimension_arg = self.rand_setting(layer.spaces[layer.dimension_key])
        else:
            dimension_arg = None
        return [type_, dimension_arg, other_args]
    
    def rand_input_shape(self):
        return self.rand_list(min_ = 2)
    
def rand_input(input_shape = None):
    if input_shape is None:
        input_shape = r.rand_input_shape()
    prob = random()
    if prob < 0.3333:
        return input_shape
    elif prob < 0.6666:
        return rand(input_shape)
    else:
        v = Vars({})
        v.add_variable("x",input_shape)
        return v
            
layer_types = layers_list
layer_types.remove("Embedding")
layer_types.remove("EmbeddingFromPretrained")

r = RandomLayor(layer_types)
    
def test_layer(layer_type,
               dim = 30,
               other_args = None,
               input_shape = None,
               verbose = False):
    if input_shape is None:
        input_shape = r.rand_input_shape()
    input_ = rand_input(input_shape)
    LayerClass = get_layer(layer_type)
    if isinstance(dim, list):
        dim = tuple(dim)
    layer = LayerClass(dim, other_args)
    output = layer(input_)
    if isinstance(output, tuple):
        output = output[0]
    shape = list(output.shape)
    if verbose:
        print("output:",shape)
        print("expected output:", layer.output_dim)
    assert shape == layer.output_dim
    if shape == input_shape:
        print("No changes")
    return layer


def test_layers(layer_types, verbose = "default"):
    if verbose == "default":
        if len(layer_types) == 1:
            verbose = True
        else:
            verbose = False 
    r = RandomLayor(layer_types)
    
    pytorch_composer.warnings_off()
    for layer_type in layer_types:
        for _ in range(50):
            layer = [*r.rand(layer_type), r.rand_input_shape()]
            if verbose:
                print(layer)
            test_layer(*layer, verbose = verbose)
        print("Tested", layer_type)

    print("All tests passed")
    

def rand_layer():
    rand = r.rand(layer_types)
    LayerClass = get_layer(rand[0])
    return LayerClass(rand[1], rand[2])
    

if __name__ == "__main__":
    test_layers(layer_types)
