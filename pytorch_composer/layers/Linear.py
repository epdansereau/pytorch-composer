from pytorch_composer.Layer import Layer


class Linear(Layer):

    def __init__(self, dimension_arg= None, other_args = None, variables = None):
        super().__init__(
                 dimension_arg,
                 other_args,
                 variables,
                 layer_type = "linear",
                 nn = "nn.Linear",
                 description = "Linear layer",
                 default_args = {
                    "bias": True,
                 },
                 dimension_key = "out_features",
                 required_args = ['in_features', 'out_features'],
                 kw_args = ["bias"],
                 spaces = {
                    'in_features':'n',
                    'out_features':'n',
                    'bias':'bool', 
                 },
        )
        
    @staticmethod
    def has_weights():
        return True

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
        
    def update_model(self, model):
        out = model.block.output_dim.copy()
        out[-1] = self.valid_args['out_features']
        model.block.variables.update_x(out)        
        self.add_unique_layer(model.block)

