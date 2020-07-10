from pytorch_composer.Layer import Layer


class Linear(Layer):

    def __init__(self, dimension_arg= None, other_args = None, variables = None):
        super().__init__(dimension_arg, other_args, variables)
        self.layer_type = "linear"
        self.nn = "nn.Linear"
        self.description = "Linear layer"

        # Arguments:
        self.default_args = {
            "bias": True,
        }
        self.dimension_key = 'out_features'
        self.required_args = ['in_features', 'out_features']
        self.kw_args = ['bias']
        self.spaces = {
            'in_features':'n',
            'out_features':'n',
            'bias':'bool', 
        }

    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    @property
    def valid_args(self):
        args = self.active_args
        args['in_features'] = self.input_dim[-1]
        if not 'out_features' in args:
            args['out_features'] = args['in_features']
        return args

    def update_variables(self):
        out = self.input_dim.copy()
        out[-1] = self.valid_args['out_features']
        self.variables.update_x(out)

    # Updating the block object:

    def update_block(self, block):
        self.add_unique_layer(block)
