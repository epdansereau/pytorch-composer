from pytorch_composer.Layer import Layer


class AdaptiveAvgPool3d(Layer):

    def __init__(self, input_dim, batch_rank):
        self.layer_type = "adaptiveavgpool3d"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = None
        self.nn = "nn.AdaptiveAvgPool3d"
        self.description = "Resizing with adaptive average pooling"
        self.batch_rank = batch_rank

        # Arguments:
        self.default_args = {
        }
        self.dimension_key = 'output_size'
        self.required_args = ['output_size']
        self.kw_args = []

    @staticmethod
    def required_batch_rank(data_dim, data_rank, args):
        return 0  
        
    @classmethod
    def create(cls, input_dim, dimension_arg, other_args, batch_rank):
        if other_args is None:
            other_args = {}
        layer = cls(input_dim, batch_rank)
        new_shape = layer.int_to_tuple(dimension_arg)
        out = input_dim.copy()
        out[-3] = new_shape[0]
        out[-2] = new_shape[1]
        out[-1] = new_shape[2]
        layer.output_dim = out
        layer.args = layer.write_args({'output_size':dimension_arg})
        return layer

    @staticmethod
    def valid_input_dims(input_dims, batch_rank):
        return Layer.change_rank(input_dims, batch_rank)

    def update_block(self, block):
        return self.add_reusable_layer(block)
