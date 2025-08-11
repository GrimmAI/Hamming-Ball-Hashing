import torch.nn as nn


class LinearBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, bias=True, use_BN: bool = False):
        super(LinearBnRelu, self).__init__()
        self.fc = nn.Linear(in_planes, out_planes, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class HASHMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_BN: bool = False, use_tanh: bool = False):
        super(HASHMLP, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(LinearBnRelu(in_size, hidden_size))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        if use_BN:
            self.last_bn = nn.BatchNorm1d(output_size, affine=False)
            layers.append(self.last_bn)
        if use_tanh:
            layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
