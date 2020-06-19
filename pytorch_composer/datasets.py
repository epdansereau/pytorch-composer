from pytorch_composer.CodeSection import CodeSection

ROOT = "Path.home() / '.pyt-comp-data'"
NUM_WORKERS = 0

# == Vision datasets ==

class CIFAR10(CodeSection):
    def __init__(self, settings = None):
        template = '''
# Load and normalize the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
root = ${root}
if not os.path.exists(root):
    os.makedirs(root)

trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size},
                                          shuffle=True, num_workers=${num_workers})

testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size},
                                         shuffle=False, num_workers=${num_workers})
'''
        defaults = {
            "batch_size":4,
            "num_workers":NUM_WORKERS,
            "root":ROOT
        }
        
        imports = set((
            "torch",
            "torchvision",
            "torchvision.transforms as transforms",
            ("pathlib", "Path"),
            "os",
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key == "batch_size":
            self.set_variables(None)
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],3,32,32],0)
        self.variables.add_variable("y",[self["batch_size"]],0,10)
        
class MNIST(CodeSection):
    def __init__(self, settings = None):
        template = '''
# Load and normalize the dataset
transform = transforms.Compose(
    [transforms.ToTensor()])
    
root = ${root}
if not os.path.exists(root):
    os.makedirs(root)

trainset = torchvision.datasets.MNIST(root=root, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size},
                                          shuffle=True, num_workers=${num_workers})

testset = torchvision.datasets.MNIST(root=root, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size},
                                         shuffle=False, num_workers=${num_workers})
'''
        defaults = {
            "batch_size":4,
            "num_workers":NUM_WORKERS,
            "root":ROOT
        }
        
        imports = set((
            "torch",
            "torchvision",
            "torchvision.transforms as transforms",
            ("pathlib", "Path"),
            "os",
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key == "batch_size":
            self.set_variables(None)
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],1,28,28],0)
        self.variables.add_variable("y",[self["batch_size"]],0,10)
        
# == Text dataset ==

class AG_NEWS(CodeSection):
    def __init__(self, settings = None):
        template = '''
        
root = ${root}
if not os.path.exists(root):
    os.makedirs(root)

train_data_path = root / "AG_NEWS_trainNgram1.data"
test_data_path = root / "AG_NEWS_testNgram1.data"

if train_data_path.exists() and test_data_path.exists():
    trainset = torch.load(train_data_path)
    testset = torch.load(test_data_path)
else:
    trainset, testset = text_classification.DATASETS["AG_NEWS"](root=root)
    print("Saving train data to {}".format(train_data_path))
    torch.save(trainset, train_data_path)
    print("Saving test data to {}".format(test_data_path))
    torch.save(testset, test_data_path)

def generate_batch(batch):
    sequence_length = ${sequence_length}
    pad_token = 0
    
    input_ids = []
    labels = []
    
    for sequence in batch:
        input_id = sequence[1][:sequence_length]
        input_id = torch.cat((input_id, torch.zeros(sequence_length - len(input_id),dtype = torch.int64)))
        input_ids.append(input_id)
        labels.append(sequence[0])
    return torch.stack(input_ids).long(), torch.IntTensor(labels).long()
    
trainloader = DataLoader(
    trainset,
    batch_size=${batch_size},
    collate_fn=generate_batch,
    pin_memory=True)
    
testloader = DataLoader(
    testset,
    batch_size=${batch_size},
    collate_fn=generate_batch,
    pin_memory=True)    
'''
        defaults = {
            "batch_size":4,
            "root":ROOT,
            "sequence_length":200,
            "classes":4,
        }
        
        imports = set((
            "torch",
            ("torchtext.datasets","text_classification"),
            ("torch.utils.data","DataLoader"),
            ("pathlib", "Path"),
            "os",
        ))
        self.vocab_size = 95812
        super().__init__(None, settings, defaults, template, imports)
        
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],self["sequence_length"]],0,self.vocab_size)
        self.variables.add_variable("y",[self["batch_size"]],0,self["classes"])

# == Debugging datasets ==

class RandDataset(CodeSection):
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
        self.variables.add_variable("y",[self["batch_size"]],0,self["classes"])
        
class RandLongDataset(CodeSection):
    def __init__(self, settings = None):
        template = '''
# Tensor with random values
class Rand:
    def __init__(self):
        self.shape = ${shape}
        self.range = ${range}
        
    def __getitem__(self, i):
        return torch.randint(0,self.range - 1,self.shape), torch.randint(${classes},[1])[0]
        
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
        self.variables.add_variable("x",[self["batch_size"],*self["shape"]],0,self["range"])
        self.variables.add_variable("y",[self["batch_size"]],0,self["classes"]) 
        

