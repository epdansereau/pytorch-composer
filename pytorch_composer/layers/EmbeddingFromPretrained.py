from pytorch_composer.Layer import Layer


class EmbeddingFromPretrained(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):

        super().__init__(
                dimension_arg,
                other_args,
                variables,
                layer_type = "embedding",
                description = "Embedding layer",
                nn = "nn.Embedding.from_pretrained",
                default_args = {
                    'embeddings':None,
                    "freeze":True,
                    "padding_idx":None,
                    "max_norm":None,
                    "norm_type":2.0,
                    "scale_grad_by_freq":False,
                    "sparse":False,
                },
                dimension_key = "embeddings",
                required_args = ['embeddings'],
                kw_args = ['freeze', 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse'],
                spaces = {},
                is_embed = True
        )   
            
    @property
    def valid_args(self):
        args = self.active_args
        args['embeddings'] = self.vocab.weights
        return args
        
    def update_model(self, model):
        input_dim = model.block.output_dim.copy()
        out = input_dim.copy()
        out += [model.block.variables["x"][0].vocab.embed_dim]
        model.block.variables.update_x(out,vocab = None)
        
        self.add_unique_layer(model.block)
        