from pytorch_composer.Layer import Layer
from pytorch_composer.CodeSection import Vars


class RNN(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(dimension_arg, other_args, variables)
        self.layer_type = "rnn"
        self.nn = "nn.RNN"
        self.description = "Recurrent layer"

        # Arguments
        self.default_args = {
            "num_layers": 1,
            "bias": True,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": False,
            'input_size':32,
            'hidden_size':32,
        }
        self.dimension_key = 'hidden_size'
        self.required_args = ['input_size', 'hidden_size']
        self.kw_args = [
            'num_layers',
            'bias',
            'batch_first',
            'dropout',
            'bidirectional']
        
        self.spaces = {
            "input_size":"n",
            "hidden_size":"n",
            'num_layers':"n",
            'nonlinearity':set(['tanh','relu']),
            "bias": "bool",
            #"batch_first": "bool",    ######TBD
            'dropout':'float',
            #"bidirectional": "bool",  ######TBD
        }

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
    def create(cls, dimension_arg, other_args = None, variables = None):
        layer = cls(dimension_arg, other_args, variables)
        layer.update_variables()
        if layer.valid_args["bidirectional"]:
            num_directions = 2
        else:
            num_directions = 1
        if layer.valid_args["batch_first"]:
            layer.hidden_dim = tuple(
                [layer.valid_args["num_layers"]*num_directions, layer.output_dim[0], layer.output_dim[2]])
        else:
            layer.hidden_dim = tuple([layer.valid_args["num_layers"]*num_directions] + layer.output_dim[1:])
        layer.args = layer.write_args(layer.valid_args)
        return layer

    @property
    def valid_args(self):
        args = self.active_args
        args['input_size'] = self.input_dim[-1]
        return args

    def update_variables(self):
        out = self.input_dim.copy()
        out[-1] = self.valid_args['hidden_size']
        self.variables.update_x(out)

    # Updating the block object:

    def update_block(self, block):
        block.variables.add_variable("h",self.hidden_dim, self.batch_rank)
        self.add_hidden_layer(block)

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
