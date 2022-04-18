import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary


lrelu_value = 0.1
act = nn.LeakyReLU(lrelu_value)


def make_model(args, parent=False):
    return TRAIN_FMEN(args)


def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t


def get_bn_bias(bn_layer):
    gamma, beta, mean, var, eps = bn_layer.weight, bn_layer.bias, bn_layer.running_mean, bn_layer.running_var, bn_layer.eps
    std = (var + eps).sqrt()
    bn_bias = beta - mean * gamma / std

    return bn_bias


class RRRB(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|


    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(RRRB, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, ratio*n_feats, 1, 1, 0)
        self.fea_conv = nn.Conv2d(ratio*n_feats, ratio*n_feats, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(ratio*n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return out


class ERB(nn.Module):
    """ Enhanced residual block for building FEMN.

    Diagram:
        --RRRB--LeakyReLU--RRRB--
        
    Args:
        n_feats (int): Number of feature maps.
        ratio (int): Expand ratio in RRRB.
    """

    def __init__(self, n_feats, ratio=2):
        super(ERB, self).__init__()
        self.conv1 = RRRB(n_feats, ratio)
        self.conv2 = RRRB(n_feats, ratio)

    def forward(self, x):
        out = self.conv1(x)
        out = act(out)
        out = self.conv2(out)

        return out


class HFAB(nn.Module):
    """ High-Frequency Attention Block.

    Diagram:
        ---BN--Conv--[ERB]*up_blocks--BN--Conv--BN--Sigmoid--*--
         |___________________________________________________|

    Args:
        n_feats (int): Number of HFAB input feature maps.
        up_blocks (int): Number of ERBs for feature extraction in this HFAB.
        mid_feats (int): Number of feature maps in ERB.

    Note:
        Batch Normalization (BN) is adopted to introduce global contexts and achieve sigmoid unsaturated area.

    """

    def __init__(self, n_feats, up_blocks, mid_feats, ratio):
        super(HFAB, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_feats)
        self.bn2 = nn.BatchNorm2d(mid_feats)
        self.bn3 = nn.BatchNorm2d(n_feats)

        self.squeeze = nn.Conv2d(n_feats, mid_feats, 3, 1, 0)

        convs = [ERB(mid_feats, ratio) for _ in range(up_blocks)]
        self.convs = nn.Sequential(*convs)

        self.excitate = nn.Conv2d(mid_feats, n_feats, 3, 1, 0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # explicitly padding with bn bias
        out = self.bn1(x)
        bn1_bias = get_bn_bias(self.bn1)
        out = pad_tensor(out, bn1_bias) 

        out = act(self.squeeze(out))
        out = act(self.convs(out))

        # explicitly padding with bn bias
        out = self.bn2(out)
        bn2_bias = get_bn_bias(self.bn2)
        out = pad_tensor(out, bn2_bias)

        out = self.excitate(out)

        out = self.sigmoid(self.bn3(out))

        return out * x


class TRAIN_FMEN(nn.Module):
    """ Fast and Memory-Efficient Network

    Diagram:
        --Conv--Conv-HFAB-[ERB-HFAB]*down_blocks-Conv-+-Upsample--
               |______________________________________|

    Args:
        down_blocks (int): Number of [ERB-HFAB] pairs.
        up_blocks (list): Number of ERBs in each HFAB.
        mid_feats (int): Number of feature maps in branch ERB.
        n_feats (int): Number of feature maps in trunk ERB.
        n_colors (int): Number of image channels.
        scale (list): upscale factor.
        backbone_expand_ratio (int): Expand ratio of RRRB in trunk ERB.
        attention_expand_ratio (int): Expand ratio of RRRB in branch ERB.
    """

    def __init__(self, args):
        super(TRAIN_FMEN, self).__init__()

        self.down_blocks = args.down_blocks

        up_blocks = args.up_blocks
        mid_feats = args.mid_feats
        n_feats = args.n_feats
        n_colors = args.n_colors
        scale = args.scale[0]
        backbone_expand_ratio = args.backbone_expand_ratio
        attention_expand_ratio = args.attention_expand_ratio

        # define head module
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # warm up
        self.warmup = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            HFAB(n_feats, up_blocks[0], mid_feats-4, attention_expand_ratio)
        )

        # define body module
        ERBs = [ERB(n_feats, backbone_expand_ratio) for _ in range(self.down_blocks)]
        HFABs = [HFAB(n_feats, up_blocks[i+1], mid_feats, attention_expand_ratio) for i in range(self.down_blocks)]

        self.ERBs = nn.ModuleList(ERBs)
        self.HFABs = nn.ModuleList(HFABs)

        self.lr_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )


    def forward(self, x):
        x = self.head(x)

        h = self.warmup(x)
        for i in range(self.down_blocks):
            h = self.ERBs[i](h)
            h = self.HFABs[i](h)
        h = self.lr_conv(h)

        h += x
        x = self.tail(h)

        return x 


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class Args:
    def __init__(self):
        self.down_blocks = 4
        self.up_blocks = [2, 1, 1, 1, 1]
        self.n_feats = 50
        self.mid_feats = 16
        self.backbone_expand_ratio = 2
        self.attention_expand_ratio = 2

        self.scale = [4]
        self.rgb_range = 255
        self.n_colors = 3

if __name__ == '__main__':
    args = Args()
    model = TRAIN_FMEN(args).to('cuda')
    in_ = torch.randn(1, 3, round(720/args.scale[0]), round(1280/args.scale[0])).to('cuda')
    summary(model, in_)
