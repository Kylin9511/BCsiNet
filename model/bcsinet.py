r""" The proposed BCsiNet
"""

from torch import nn
from collections import OrderedDict

from utils import logger

__all__ = ["bcsinet"]


def conv3x3_bn(in_channels, out_channel, stride=1, groups=1):
    r""" 3x3 convolution with padding, followed by batch normalization
    """

    return nn.Sequential(OrderedDict([
        ("conv3x3", nn.Conv2d(in_channels, out_channel, kernel_size=3,
                              stride=stride, padding=1, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(num_features=out_channel))
    ]))


class RefineBlock(nn.Module):
    def __init__(self):
        super(RefineBlock, self).__init__()
        self.conv1_bn = conv3x3_bn(2, 8)
        self.conv2_bn = conv3x3_bn(8, 16)
        self.conv3_bn = conv3x3_bn(16, 2)
        self.activation = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.activation(self.conv1_bn(x))
        residual = self.activation(self.conv2_bn(residual))
        residual = self.conv3_bn(residual)

        return self.activation(residual + identity)


class TinyRefineBlock(nn.Module):
    r"""
    This is headC for BCsiNet. Residual architecture is included.
    """
    def __init__(self):
        super(TinyRefineBlock, self).__init__()
        self.conv1_bn = conv3x3_bn(2, 4)
        self.conv2_bn = conv3x3_bn(4, 2)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.relu(self.conv1_bn(x))
        residual = self.conv2_bn(residual)

        return self.relu(residual + identity)


class BCsiNet(nn.Module):
    def __init__(self, reduction, encoder_head, num_refinenet):
        super(BCsiNet, self).__init__()
        logger.info(f"=> Model BCsiNet with reduction={reduction}, ")

        in_channels, total_size, w, h = 2, 2048, 32, 32
        if encoder_head == 'A':
            encoder_feature = [
                ("conv3x3_bn", conv3x3_bn(in_channels, 2)),
                ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True))
            ]
        elif encoder_head == 'B':
            encoder_feature = [
                ("conv3x3_bn1", conv3x3_bn(in_channels, 2)),
                ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ("conv3x3_bn2", conv3x3_bn(in_channels, 2)),
                ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True))
            ]
        elif encoder_head == 'C':
            encoder_feature = [
                ("conv3x3_bn1", conv3x3_bn(in_channels, 2)),
                ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ("tiny_refine1", TinyRefineBlock())
            ]
        else:
            raise ValueError(f'Illegal encoder type {encoder_head}')
        self.encoder_feature = nn.Sequential(OrderedDict(encoder_feature))
        self.encoder_binary_fc = nn.Linear(total_size, total_size // reduction)

        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        decoder_feature = []
        for i in range(num_refinenet):
            decoder_feature.append((f"refine{i}", RefineBlock()))
        self.decoder_feature = nn.Sequential(OrderedDict(
            decoder_feature + [
                ("conv3x3_bn", conv3x3_bn(2, in_channels)),
                ("sigmoid", nn.Sigmoid())
            ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _fc_binarization(self):
        r"""
        Note that this PyTorch based binarization only proves the correctness of the
        proposed BCsiNet for simplicity. In order to observe the memory saving and
        inference speed up, C++ codes is needed on general CPU while more customized
        codes are required for ASIC chips at resource limited user equipment.
        """
        
        module = self.encoder_binary_fc
        data = module.weight.data
        mn = data.nelement()
        alpha = data.norm(1).div(mn)
        module.weight.data = data.sign().mul(alpha)

    def forward(self, x):
        assert self.training is False, 'This repo works only for inference'
        n, c, h, w = x.detach().size()

        # For encoder inference at UE
        out = self.encoder_feature(x)
        out = self.encoder_binary_fc(out.view(n, -1))

        # For decoder inference at BS
        out = self.decoder_fc(out)
        out = self.decoder_feature(out.view(n, c, h, w))

        return out


def bcsinet(reduction=4, encoder_head='A', num_refinenet=2):
    r""" Create a proposed BCsiNet model.
    """

    model = BCsiNet(reduction=reduction,
                    encoder_head=encoder_head,
                    num_refinenet=num_refinenet)
    return model
