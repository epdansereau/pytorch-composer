from pytorch_composer.Layer import Layer


class Linear(Layer):

    def __init__(self, input_dim):
        self.layer_type = "linear"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = None
        self.nn = "nn.Linear"
        self.description = "Linear layer"

        # Arguments:
        self.default_args = {
            "bias": True
        }
        self.dimension_key = 'out_features'
        self.required_args = ['in_features', 'out_features']
        self.kw_args = ['bias']

    @classmethod
    def create(cls, input_dim, dimension_arg, other_args={}):
        layer = cls(input_dim)
        args = layer.active_args(dimension_arg, other_args)
        args['in_features'] = input_dim[-1]
        args['out_features'] = dimension_arg
        layer.output_dim = input_dim.copy()
        layer.output_dim[-1] = dimension_arg
        layer.args = layer.write_args(args)
        return layer

    def update_block(self, block):
        return self.add_unique_layer(block)
