from pytorch_composer.Layer import Layer


class Embedding(Layer):

    def __init__(self, variables):
        self.layer_type = "embedding"
        self.args = None
        self.input_dim = variables.output_dim.copy()
        self.nn = "nn.Embedding"
        self.description = "Embedding layer"
        self.variables = variables
        self.vocab = variables["x"][0].vocab
        if self.vocab is None:
            raise ValueError("Not supported: the embedding layer has to receive a tensor of int64")

        # Arguments:
        self.default_args = {
            "padding_idx":None,
            "max_norm":None,
            "norm_type":2.0,
            "scale_grad_by_freq":False,
            "sparse":False,
            "_weight":None,
        }
        self.dimension_key = 'embedding_dim'
        self.required_args = ['num_embeddings', 'embedding_dim']
        self.kw_args = ['padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse', '_weight']

    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    def get_valid_args(self, args):
        args['num_embeddings'] = self.vocab.size
        return args

    def update_variables(self, args):
        out = self.input_dim.copy()
        out += [args['embedding_dim']]
        self.variables.update_x(out,vocab = None)

    # Updating the block object:

    def update_block(self, block):
        self.add_unique_layer(block)
