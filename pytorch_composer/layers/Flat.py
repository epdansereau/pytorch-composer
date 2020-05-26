from pytorch_composer.Layer import Layer

class Flat(Layer):
    valid_input_dim = "flat"
    
    def __init__(self, input_dim):
        self.layer_type = "flat"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.nn = None
        self.description = "Flatenning the data"
        
        # Arguments:
        self.default_args = {}
        self.required_args = []
        self.kw_args = []
   
    @classmethod
    def create(cls, input_dim, dimension_arg, other_args):
        return cls(input_dim)
    
    def update_block(self, block):
        # Nothing to do here since the reshape happens earlier
        return block