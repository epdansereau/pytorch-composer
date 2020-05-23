from collections import Counter

class Layers_writer():
    '''Subscritable class that writes the code for each layer'''
    def __init__(self):
        self._layer_count = Counter()
        self._layer_group_count = {}
    
    def __call__(self, block, layer):
        self._layer_count = block.count
        self._layer_group_count = block.group_count
        layers_list, forward_function = getattr(self, layer.layer_type.lower())(block.layers_list,
                                                                                block.forward_function, layer)
        return self._layer_count, self._layer_group_count, layers_list, forward_function
    
    def _count_layers(self, layer):
        self._layer_count[layer.layer_type] += 1
        return self._layer_count[layer.layer_type]
    
    def _count_layers_group(self, layer):
        if layer.layer_type in self._layer_group_count:
            if layer.args in self._layer_group_count[layer.layer_type]:
                return self._layer_group_count[layer.layer_type].index(layer.args), False
            else:
                self._layer_group_count[layer.layer_type].append(layer.args)
                return len(self._layer_group_count[layer.layer_type]), True
        else:
            self._layer_group_count[layer.layer_type] = [layer.args]
            return 1, True
        
    def _add_unique_layer(self, layers_list, forward_function, layer):
        ind = self._count_layers(layer)
        layers_list.append(["layer", "self.{}".format(layer.layer_type), ind," = {}({})".format(layer.nn, layer.args)])
        if layer.input_dim == layer.output_dim:
            forward_function.append(["comment","Shape stays at {}".format(layer.input_dim)])
        else:
            forward_function.append(["comment","Shape goes from {} to {}".format(
                layer.input_dim, layer.output_dim)])
        forward_function.append(["forward","x = ", "self.{}{}".format(layer.layer_type, ind),"(x)"])
        return layers_list, forward_function
    
    def _add_reusable_layer(self, layers_list, forward_function, layer):
        ind, new_group = self._count_layers_group(layer)
        if new_group:
            layers_list.append(["layer", "self.{}".format(layer.layer_type), ind," = {}({})".format(layer.nn,layer.args)])
        if layer.input_dim == layer.output_dim:
            forward_function.append(["comment","Shape stays at {}".format(layer.input_dim)])
        else:
            forward_function.append(["comment","Shape goes from {} to {}".format(
                layer.input_dim, layer.output_dim)])
        forward_function.append(["forward","x = ", "self.{}{}".format(layer.layer_type, ind),"(x)"])
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
    
write = Layers_writer()