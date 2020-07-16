from pytorch_composer import get_layer
from torch import rand
from pytorch_composer.CodeSection import Vars, Vocab

from random import randint, random, choice, sample
from pytorch_composer.get_layer import get_layer, layers_list

import pytorch_composer

class RandomLayor:
    def __init__(self):
        self.max_int = 40
        
        self.max_list = 5
        
        self.x10prob = 0.05
        self.critical_prob = 0.05
        self.none_prob = 0.25
        
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
    
    def rand(self,type_):
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

    
def test_layer(layer_type,
               dim = 30,
               other_args = None,
               input_shape = None,
               verbose = False):
    if input_shape is None:
        input_shape = [5,38,10]
    t = rand(input_shape)
    env = Vars({})
    if layer_type != "Embedding":
        env.add_variable("x",input_shape)
    else:
        env.add_variable("x",input_shape,0,[x for x in range(RandomLayer().max_int)])
    if verbose:
        print("input:", env)
    LayerClass = get_layer(layer_type)
    if isinstance(dim, list):
        dim = tuple(dim)
    layer = LayerClass(dim, other_args, env)
    output = layer(t)
#     if verbose:
#         print(layer.layer_model.valid_args)
    if isinstance(output, tuple):
        output = output[0]
    shape = list(output.shape)
    if verbose:
        print("output:",shape)
        print("expected output:", layer.layer_model.variables["x"][0].dim)
    assert shape == layer.layer_model.variables["x"][0].dim
    return layer

def test_layers(layer_types, verbose = "default"):
    if verbose == "default":
        if len(layer_types) == 1:
            verbose = True
        else:
            verbose = False 
    r = RandomLayor()
    
    pytorch_composer.warnings_off()
    for layer_type in layer_types:
        for _ in range(50):
            layer = [*r.rand(layer_type), r.rand_list(min_ = 2)]
            if verbose:
                print(layer)
            test_layer(*layer, verbose = verbose)
        print("Tested", layer_type)

    print("All tests passed")


layer_types = layers_list
layer_types.remove("Embedding")

if __name__ == "__main__":
    test_layers(layer_types)
