from pytorch_composer.CodeSection import CodeSection


class CIFAR10(CodeSection):
    def __init__(self, settings = None):
        super().__init__()
        self._template = '''
# Load and normalize the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''
        self.imports = set((
            "torch",
            "torchvision",
            "torchvision.transforms as transforms"
        ))
    
    def set_default_variables(self):
        self.variables.add_variable("x",[4,3,32,32],0)
        self.variables.add_variable("y",[4,10],0)
        

