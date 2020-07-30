from pytorch_composer.Layer import Layer


class Embedding(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(
                dimension_arg,
                other_args,
                variables,
                layer_type = "embedding",
                description = "Embedding layer",
                nn = "nn.Embedding",
                default_args = {
                    "padding_idx":None,
                    "max_norm":None,
                    "norm_type":2.0,
                    "scale_grad_by_freq":False,
                    "sparse":False,
                    "_weight":None,
                    'embedding_dim':10, #default size
                },
                dimension_key = "embedding_dim",
                required_args = ['num_embeddings', 'embedding_dim'],
                kw_args = ['padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse', '_weight'],
                spaces = {"embedding_dim":"n"},
                is_embed = True,
        )
        
    @staticmethod
    def has_weights():
        return True

    @property
    def valid_args(self):
        args = self.active_args
        args['num_embeddings'] = self.vocab.size
        return args
        
    def update_model(self, model):
        input_dim = model.block.output_dim.copy()
        out = input_dim.copy()
        args = self.valid_args
        out += [args['embedding_dim']]
        model.block.variables.update_x(out,vocab = None)
        
        self.add_unique_layer(model.block)
        