from pytorch_composer.Layer import Layer


class RNN(Layer):

    def __init__(self, variables):
        self.layer_type = "rnn"
        self.args = None
        self.input_dim = variables.output_dim.copy()
        self.nn = "nn.RNN"
        self.description = "Recurrent layer"
        self.variables = variables

        # Arguments
        self.default_args = {
            "num_layers": 1,
            "bias": True,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": False,
        }
        self.dimension_key = 'hidden_size'
        self.required_args = ['input_size', 'hidden_size']
        self.kw_args = [
            'num_layers',
            'bias',
            'batch_first',
            'dropout',
            'bidirectional']

        self.hidden_dim = None

    # Main loop:

    # Valid permutation:

    @staticmethod
    def required_batch_rank(data_dim, data_rank, args):
        if "batch_first" in args:
            return int(not(args["batch_first"]))
        return 1

    # Valid input dimensions:

    @staticmethod
    def valid_input_dims(input_dims, batch_rank):
        return Layer.change_rank(input_dims, 3, batch_rank)

    # Creating the layer:

    @classmethod
    def create(cls, variables, dimension_arg, other_args):
        if other_args is None:
            other_args = {}
        layer = cls(variables)
        args = layer.active_args(dimension_arg, other_args)
        args = layer.get_valid_args(args)
        layer.update_variables(args)
        if args["batch_first"]:
            layer.hidden_dim = tuple(
                [1, layer.output_dim[0], layer.output_dim[2]])
        else:
            layer.hidden_dim = tuple([1] + layer.output_dim[1:])
        layer.args = layer.write_args(args)
        return layer

    def get_valid_args(self, args):
        args['input_size'] = self.input_dim[-1]
        return args

    def update_variables(self, args):
        out = self.input_dim.copy()
        out[-1] = args['hidden_size']
        self.variables.update_x(out)

    # Updating the block object:

    def update_block(self, block):
        block.variables.add_variable("h",self.hidden_dim, self.batch_rank)
        return self.add_hidden_layer(block)

    def add_hidden_layer(self, block):
        hidden_var = self.variables["h"][-1].name
        ind = len(self.variables["h"])
        block.count[self.layer_type] += 1
        block.add_layer(["layer", "self.{}".format(
            self.layer_type), ind, " = {}({})".format(self.nn, self.args)])
        block.add_forward(["comment",
                           "{}: ".format(self.description),
                           tuple(self.input_dim),
                           " -> ",
                           tuple(self.output_dim)])
        block.add_forward(
            ["forward", "x, {} = ".format(hidden_var),
             "self.{}{}".format(self.layer_type, ind),
             "(x, {})".format(hidden_var)])
        return block
