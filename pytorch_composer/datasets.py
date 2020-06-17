from pytorch_composer.CodeSection import CodeSection

# == Vision datasets ==

class CIFAR10(CodeSection):
    def __init__(self, settings = None):
        template = '''
# Load and normalize the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='${root}', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size},
                                          shuffle=True, num_workers=${num_workers})

testset = torchvision.datasets.CIFAR10(root='${root}', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size},
                                         shuffle=False, num_workers=${num_workers})
'''
        defaults = {
            "batch_size":4,
            "num_workers":2,
            "root":"./data"
        }
        
        imports = set((
            "torch",
            "torchvision",
            "torchvision.transforms as transforms"
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

trainset = torchvision.datasets.MNIST(root='${root}', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size},
                                          shuffle=True, num_workers=${num_workers})

testset = torchvision.datasets.MNIST(root='${root}', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size},
                                         shuffle=False, num_workers=${num_workers})
'''
        defaults = {
            "batch_size":4,
            "num_workers":2,
            "root":"./data"
        }
        
        imports = set((
            "torch",
            "torchvision",
            "torchvision.transforms as transforms"
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

trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size}, shuffle=False)

testset = Rand()

testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size}, shuffle=False)
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
        if key in ["batch_size", "classes"]:
            self.set_variables(None)
            
    @property
    def active_settings(self):
        len_ = self["batch_size"]*self["batches"]
        return {**self.settings, **{"len":len_}}
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],*self["shape"]],0)
        self.variables.add_variable("y",[self["batch_size"]],0,self["classes"])    
        

