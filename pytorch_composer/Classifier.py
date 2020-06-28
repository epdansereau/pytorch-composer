from pytorch_composer.CodeSection import CodeSection
from collections import defaultdict


class Classifier(CodeSection):
    def __init__(self, data, settings = None):
        template = '''
# Define a Loss function and optimizer
net = ${model_name}()
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
        if i % ${print_every} == ${print_every} - 1:    # print every ${print_every} mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / ${print_every}))
            running_loss = 0.0
${debug1}
        torch.save(net.state_dict(), '${saving_path}')

print('Finished Training')
'''

        defaults = {
            "criterion": "CrossEntropyLoss",
            "optimizer": "SGD",
            "lr": 0.001,
            "momentum": 0.9,
            "epoch": 2,
            "print_every":2000,
            "model_name": "Net",
            "saving_path":"model.pt"
        }
        
        imports = set((
            "torch",
            "torch.optim as optim",
            "torch.nn as nn",
        ))   
        super().__init__(data, settings, defaults, template, imports)
     

    @property
    def active_settings(self):
        # adding hidden variables in training loop:
        settings = self.settings.copy()
        if "h" in self.variables:
            if self.variables["h"]:
                hidden_vars = [x.name for x in self.variables["h"]]
                var_list = ", ".join(hidden_vars)
                settings["hidden_variables"] = ", " + var_list
                settings["hidden_init"] = " " * 4 + \
                    f"{var_list} = net.initHidden()\n"
                settings["hidden_copy"] = ""
                for var in hidden_vars:
                    settings["hidden_copy"] += " " * \
                        8 + f"{var} = {var}.data\n"
                        
        return settings
            
        
    def require_input(self, input_ = None):
        return input_.variables["y"][0].dim + [input_.variables["y"][0].vocab.size]

