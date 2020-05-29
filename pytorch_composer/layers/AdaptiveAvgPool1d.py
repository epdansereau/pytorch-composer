from pytorch_composer.Layer import Layer


class AdaptiveAvgPool1d(Layer):
    def __init__(self, input_dim):
        self.layer_type = "adaptiveavgpool1d"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = None
        self.nn = "nn.AdaptiveAvgPool1d"
        self.description = "Resizing with adaptive average pooling"

        # Arguments:
        self.default_args = {
        }
        self.dimension_key = 'output_size'
        self.required_args = ['output_size']
        self.kw_args = []

    @classmethod
    def create(cls, input_dim, dimension_arg, other_args = {}):
        layer = cls(input_dim)
        if type(dimension_arg) == tuple:
            dimension_arg = dimension_arg[0]
        out = input_dim.copy()
        out[-1] = dimension_arg
        layer.output_dim = out
        layer.args = layer.write_args({'output_size':dimension_arg})
        return layer

    @staticmethod
    def valid_input_dims(input_dims):
        return Layer.change_rank(input_dims,3)

    
    def update_block(self, block):
        return self.add_reusable_layer(block)
