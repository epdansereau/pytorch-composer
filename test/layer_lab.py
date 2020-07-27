from pytorch_composer import get_layer, Model
from torch import rand

from random import randint, random, choice, sample
from pytorch_composer.get_layer import get_layer, layers_list

import pytorch_composer

from pytorch_composer.CodeSection import Vars, Vocab

import torch

pytorch_composer.warnings_off()

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
        elif space == 'pretrained_weights':
            return 'pretrained_weights'
        else:
            return self.space_dict[space]()
    
    def rand_float(self):
        if random() < self.critical_prob:
            return choice([0.,1.])
        else:
            return random()
    
    def rand_n(self, max_ = None):
        if max_ is None:
            max_ = self.max_int
        if random() < self.critical_prob:
            return 1
        else:
            return randint(1,max_)
        
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
            
layer_types = layers_list
layer_types_all = layer_types.copy()
layer_types.remove("Embedding")
layer_types.remove("EmbeddingFromPretrained")

r = RandomLayor(layer_types)
    
def test_layer(layer_type,
               dim = 30,
               other_args = None,
               input_shape = None,
               verbose = False,
              ):
    if input_shape is None:
        input_shape = r.rand_input_shape()
    weights = None
    if layer_type == "Embedding":
        input_ = rand_embed(input_shape)
    elif layer_type == "EmbeddingFromPretrained":
        input_, weights = rand_pretrained(input_shape)
    else:
        input_ = rand_input(input_shape)
    LayerClass = get_layer(layer_type)
    if isinstance(dim, list):
        dim = tuple(dim)
    layer = LayerClass(dim, other_args)
    output = layer(input_, weights = weights)
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
    for layer_type in layer_types:
        for _ in range(50):
            layer = [*r.rand(layer_type), r.rand_input_shape()]
            if verbose:
                print(layer)
            test_layer(*layer, verbose = verbose)
        print("Tested", layer_type)

    print("All tests passed")
    
def test_embed():
    for _ in range(50):
        layer = [*r.rand(layer_type), r.rand_input_shape()]
        if verbose:
            print(layer)
        test_layer(*layer, verbose = verbose)
    print("Tested", layer_type)
        
def rand_layer(layer_types = None):
    rand = r.rand(layer_types)
    LayerClass = get_layer(rand[0])
    return LayerClass(rand[1], rand[2])

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
    
def rand_embed(input_shape = None, vocab_size = None):
    if input_shape is None:
        input_shape = r.rand_input_shape()
    if  vocab_size is None:
        vocab_size = r.rand_n()
    v = Vars({})
    v.add_variable("x",input_shape,vocab = Vocab.create(vocab_size))
    return v

def rand_pretrained(input_shape = None, vocab_size = None, embed_dim = None):
    if input_shape is None:
        input_shape = r.rand_input_shape()
    if  vocab_size is None:
        vocab_size = r.rand_n()
    if  embed_dim is None:
        embed_dim = r.rand_n()
    v = Vars({})
    v.add_variable("x",input_shape,vocab = Vocab.from_pretrained("pretrained_weights",[vocab_size,embed_dim]))
    weights = torch.rand([vocab_size,embed_dim])
    return v, weights
        
def rand_input_shape():
    return r.rand_input_shape()

### property tests

def test_property_1(num = 30):
    print("testing property 1")
    for _ in range(num):
        layer = rand_layer()
        input_ = rand_input()
        try:
            output1 = layer(input_).shape
        except AttributeError:
            output1 = layer(input_)[0].shape
        output_dim1 = layer.output_dim
        bc1 = layer.get_batch_code()
        for _ in range(r.rand_n(3)):
            layer(rand_input())
        try:
            output2 = layer(input_).shape
        except AttributeError:
            output2 = layer(input_)[0].shape
        output_dim2 = layer.output_dim
        bc2 = layer.get_batch_code()

        assert output1 == output2
        assert output_dim1 == output_dim2
        assert bc1 == bc2
    print("test passed")

def test_property_2(num = 30):
    print("testing property 2")
    for _ in range(num):
        layer = rand_layer()
        input_ = rand_input()
        model1 = Model([])
        layer.update(model1)
        try:
            output_1 = model1(input_).shape
        except AttributeError:
            output_1 = model1(input_)[0].shape
        output_dim_1 = model1.block.output_dim
        c1 = str(model1)
        
        model2 = Model([], input_)
        layer.update(model2)
        try:
            output_2 = model2(input_).shape
        except AttributeError:
            output_2 = model2(input_)[0].shape
        output_dim_2 = model2.block.output_dim
        c2 = str(model2)
        
    assert output_1 == output_2
    assert output_dim_1 == output_dim_2
    assert c1 == c2
        
    print("test passed")    

def layer_lab():
    test_layers(layer_types_all)
    test_property_1()
    test_property_2()
