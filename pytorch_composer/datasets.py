from pytorch_composer.CodeSection import CodeSection


class CIFAR10(CodeSection):
    def __init__(self, settings = None):
        if settings is None:
            settings = {"input_dim":None,"output_dim":[4,3,32,32]}
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
        self.defaults = {}
        self.imports = set((
            "torch",
            "torchvision",
            "torchvision.transforms as transforms"
        ))
        super().__init__(self.template, settings, self.defaults,self.imports)
        