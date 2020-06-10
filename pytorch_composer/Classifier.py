from pytorch_composer.CodeSection import CodeSection
from collections import defaultdict


class Classifier(CodeSection):
    def __init__(self, data):
        self._template = '''
# Define a Loss function and optimizer
net = Net()
criterion = nn.${criterion}()
optimizer = optim.${optimizer}(net.parameters(), lr=${lr}, momentum=${momentum})


# Training
for epoch in range(${epoch}):  # loop over the dataset multiple times
${hidden_init}
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs${hidden_variables} = net(inputs${hidden_variables})
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
${hidden_copy}
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
${debug1}        
        torch.save(net.state_dict(), 'model.pt')

print('Finished Training')
'''
    
        settings = {
            "criterion":"CrossEntropyLoss",
            "optimizer":"SGD",
            "lr":0.001,
            "momentum":0.9,
            "epoch":2,
        }
        
        #adding hidden variables in training loop:
        if "hidden" in data.variables:
            hidden_vars = [x[0] for x in data.variables["hidden"]]
            var_list = ", ".join(hidden_vars)
            settings["hidden_variables"] = ", " + var_list
            settings["hidden_init"] = " "*4 + f"{var_list} = net.initHidden()\n"
            settings["hidden_copy"] = ""
            for var in hidden_vars:
                settings["hidden_copy"] += " "*8 + f"{var} = {var}.data\n"        
        
        imports = set((
            "torch",
            "torch.optim as optim",
            "torch.nn as nn",
        ))
        super().__init__(self.template, settings, data.variables ,imports)
        
                