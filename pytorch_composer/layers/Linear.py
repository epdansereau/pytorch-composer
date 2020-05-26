from pytorch_composer.Layer import Layer

class Linear(Layer):
    valid_input_dim = "any"
    
    def __init__(self, input_dim):
        self.layer_type = "linear"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = None
        self.nn = "nn.Linear"
        self.description = "Linear layer"
        
        # Arguments:
        self.default_args = {}
        self.required_args = ["in_features", "out_features"]
        self.kw_args = ["bias"]
   
    @classmethod
    def create(cls, input_dim, dimension_arg, other_args):
        layer = cls(input_dim)
        real = layer.real_args(layer.default_args, other_args)
        corrected_args = layer.args_out(layer.default_args, real)
        layer.output_dim = input_dim.copy()
        layer.output_dim[-1] = dimension_arg
        for arg in layer.required_args:
            if arg in corrected_args:
                corrected_args.pop(arg)
        required_args_out = [input_dim[-1], dimension_arg]
        layer.args = layer.write_args(required_args_out, corrected_args)
        return layer
    
    def update_block(self, block):
        return self._add_unique_layer(block)