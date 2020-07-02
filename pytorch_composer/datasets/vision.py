from pytorch_composer.CodeSection import CodeSection

ROOT = "Path.home() / '.pyt-comp-data'"
NUM_WORKERS = 0


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
        self.classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key == "batch_size":
            self.set_variables(None)
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],3,32,32],0)
        self.variables.add_variable("y",[self["batch_size"]],0,self.classes)
        
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
        self.classes = [x for x in range(10)]
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key == "batch_size":
            self.set_variables(None)
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],1,28,28],0)
        self.variables.add_variable("y",[self["batch_size"]],0,self.classes)
        
