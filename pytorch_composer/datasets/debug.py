from pytorch_composer.Dataset import Dataset
from pytorch_composer.CodeSection import Vocab

ROOT = "Path.home() / '.pyt-comp-data'"
NUM_WORKERS = 0


class RandDataset(Dataset):
    def __init__(self, settings = None):
        template = '''
# Tensor with random values
class Rand:
    def __init__(self):
        self.shape = ${shape}
        
    def __getitem__(self, i):
        return torch.rand(self.shape), torch.randint(${classes},[1])[0]
        
    def __len__(self):
        return ${len}


trainset = Rand()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size}, shuffle=False,
                                                        num_workers=0)

testset = Rand()

testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size}, shuffle=False,
                                        num_workers=0)
'''
        defaults = {
            "shape":[3,32,32],
            "batches":2,
            "batch_size":4,
            "classes":10,
        }
        
        imports = set((
            "torch",
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key in ["batch_size", "shape", "classes"]:
            self.set_variables(None)
            
    @property
    def active_settings(self):
        len_ = self["batch_size"]*self["batches"]        
        act_set = {"len":len_}
        return {**self.settings, **{"len":len_}}
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],*self["shape"]],0)
        self.variables.add_variable("y",[self["batch_size"]],0,[x for x in range(self["classes"])])
        
class RandLongDataset(Dataset):
    def __init__(self, settings = None):
        template = '''
# Tensor with random values
class Rand:
    def __init__(self):
        self.shape = ${shape}
        self.range = ${range}
        
    def __getitem__(self, i):
        return torch.randint(0, self.range, self.shape), torch.randint(${classes},[1])[0]
        
    def __len__(self):
        return ${len}


trainset = Rand()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size}, shuffle=False)

testset = Rand()

testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size}, shuffle=False)
'''
        defaults = {
            "shape":[200],
            "batches":2,
            "batch_size":4,
            "classes":10,
            "range":10,
        }
        
        imports = set((
            "torch",
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key in ["batch_size", "shape", "classes"]:
            self.set_variables(None)
            
    @property
    def active_settings(self):
        len_ = self["batch_size"]*self["batches"]        
        act_set = {"len":len_}
        return {**self.settings, **{"len":len_}}
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],*self["shape"]],0,[x for x in range(self["range"])])
        self.variables.add_variable("y",[self["batch_size"]],0,[x for x in range(self["range"])]) 
        

class RandPretrainedDataset(Dataset):
    def __init__(self, settings = None):
        template = '''
# Tensor with random values
class Rand:
    def __init__(self):
        self.shape = ${shape}
        self.range = ${range}
        
    def __getitem__(self, i):
        return torch.randint(0, self.range, self.shape), torch.randint(${classes},[1])[0]
        
    def __len__(self):
        return ${len}
        

trainset = Rand()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size}, shuffle=False)

testset = Rand()

testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size}, shuffle=False)

pretrained_vocab = torch.rand([${embed_size},${embed_dim}])
'''
        defaults = {
            "shape":[200],
            "batches":2,
            "batch_size":4,
            "classes":10,
            "range":10,
            "embed_size":10,
            "embed_dim":20,
        }
        
        imports = set((
            "torch",
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key in ["batch_size", "shape", "classes"]:
            self.set_variables(None)
            
    @property
    def active_settings(self):
        len_ = self["batch_size"]*self["batches"]        
        act_set = {"len":len_}
        return {**self.settings, **{"len":len_}}
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",
                                    [self["batch_size"],*self["shape"]],
                                    0,
                                    Vocab.from_pretrained("pretrained_vocab", [self["embed_size"], self["embed_dim"]]))
        self.variables.add_variable("y",[self["batch_size"]],0,[x for x in range(self["range"])]) 
        