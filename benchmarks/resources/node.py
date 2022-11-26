import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class ODEFunc(nn.Module):

    def __init__(self, input_shape, n_units=30):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape, n_units),
            nn.Tanh(),
            ResNet(
                nn.Sequential(
                    nn.Linear(n_units, n_units),
                    nn.ELU(),
                    nn.Linear(n_units, n_units),
                    nn.ELU(),
                )
            ),
            nn.Linear(n_units, input_shape),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
    
