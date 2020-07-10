from pytorch_composer import get_layer
from torch import rand
from pytorch_composer.CodeSection import Vars

from random import randint, random, choice
from pytorch_composer.get_layer import get_layer

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
        }
    
    def rand_n(self):
        if random() < self.critical_prob:
            return 1
        else:
            return randint(1,self.max_int)
        
    def rand_bool(self):
        return choice([True,False])
    
    def choice(self,set_):
        return choice(set_)
    
    def rand_list(self):
        list_ = []
        for i in range(randint(1,self.max_list)):
            list_.append(self.rand_n())
        return list_
    
    def rand(self,type_):
        none_prob = self.none_prob
        dim_none_prob = self.none_prob
        layer = get_layer(type_)()
        other_args = {}
        for arg, space in layer.spaces.items():
            if random() > self.none_prob:
                if space in self.space_dict:
                    other_args[arg] = self.space_dict[space]()
        if random() > self.none_prob:
            dimension_arg = self.space_dict[layer.spaces[layer.dimension_key]]()
        else:
            dimension_arg = None
        return [type_, dimension_arg, other_args]

    
def test_layer(layer_type, dim = 30, other_args = None, input_shape = None):
    if input_shape is None:
        input_shape = [5,38,10]
    t = rand(input_shape)
    env = Vars({})
    env.add_variable("x",input_shape)
    print("input:", env)
    LayerClass = get_layer("Linear")
    layer = LayerClass.create(dim, other_args, env)
    print(layer.valid_args)
    print("output:",layer(t).shape)
    return layer

if __name__ == "__main__":
    r = RandomLayor()
    for i in range(50):
        layer = [*r.rand("Linear"), r.rand_list()]

        print(layer)
        print("Test:")
        test_layer(*layer)
        print()
            