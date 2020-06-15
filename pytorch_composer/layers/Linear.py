from pytorch_composer.Layer import Layer


class Linear(Layer):

    def __init__(self, variables):
        self.layer_type = "linear"
        self.args = None
        self.input_dim = variables.output_dim.copy()
        self.nn = "nn.Linear"
        self.description = "Linear layer"
        self.variables = variables

        # Arguments:
        self.default_args = {
            "bias": True
        }
        self.dimension_key = 'out_features'
        self.required_args = ['in_features', 'out_features']
        self.kw_args = ['bias']

    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    def get_valid_args(self, args):
        args['in_features'] = self.input_dim[-1]
        return args

    def update_variables(self, args):
        out = self.input_dim.copy()
        out[-1] = args['out_features']
        self.variables.update_x(out)

    # Updating the block object:

    def update_block(self, block):
        self.add_unique_layer(block)
