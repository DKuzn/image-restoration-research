import torch


class RED10(torch.nn.Module):
    def __init__(self, num_layers=5, num_features=64):
        super(RED10, self).__init__()
        conv_layers = []
        deconv_layers = []

        conv_layers.append(torch.nn.Sequential(torch.nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                                               torch.nn.ReLU(inplace=True)))
        for _ in range(num_layers - 1):
            conv_layers.append(self._make_conv_layer(64))

        for _ in range(num_layers - 1):
            deconv_layers.append(self._make_deconv_layer(64))
        deconv_layers.append(torch.nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = torch.nn.Sequential(*conv_layers)
        self.deconv_layers = torch.nn.Sequential(*deconv_layers)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_layers(x)
        out = self.deconv_layers(out)
        out += residual
        out = self.relu(out)
        return out

    def _make_conv_layer(self, num_features):
        return torch.nn.Sequential(torch.nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                   torch.nn.ReLU(inplace=True))

    def _make_deconv_layer(sefl, num_features):
        return torch.nn.Sequential(torch.nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                   torch.nn.ReLU(inplace=True))
