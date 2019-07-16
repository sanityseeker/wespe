import torch
import torch.nn as nn
import torchvision


class Flatten(nn.Module):
    def forward(self, input):
        batch_size = input.shape[0]
        return input.view([batch_size, -1])


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.net(input)
        return output + input


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 9, padding=4),
            nn.ReLU(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 9, padding=4),
            nn.Tanh()
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # self.apply(weights_init)

    def forward(self, input):
        output = self.net(input)
        # The constants are from original paper. Maybe change to linear layer?
        return output * 0.5 + 0.5


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        self.in_channels = channels
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 11, stride=4, padding=5),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(256, 256, 3, stride=2, padding=1),
            # nn.InstanceNorm2d(256, affine=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d((128, 128)),
        )
        self.clf = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(128, 1)
        )

        def weights_init(layer):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        self.apply(weights_init)

    def forward(self, input):
        output = self.net(input)
        output = output.squeeze(2).squeeze(2)
        return self.clf(output).view(input.shape[0])


class Vgg19(torch.nn.Module):
    def __init__(self, device='cpu', bn=True):
        self.bn = bn
        super().__init__()
        if self.bn:
            features = list(torchvision.models.vgg19_bn(pretrained=True).features)
        else:
            features = list(torchvision.models.vgg19(pretrained=True).features)
        self.means = torch.FloatTensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1).to(device)
        self.features = nn.ModuleList(features[:-1])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, input):
        normalized = input * 255 - self.means
        for layer, model in enumerate(self.features):
            normalized = model(normalized)
        return normalized
