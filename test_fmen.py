import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary


lrelu_value = 0.1
act = nn.LeakyReLU(lrelu_value)


def make_model(args, parent=False):
    return TEST_FMEN(args)


class RRRB(nn.Module):
    def __init__(self, n_feats):
        super(RRRB, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)

        return out


class ERB(nn.Module):
    def __init__(self, n_feats):
        super(ERB, self).__init__()
        self.conv1 = RRRB(n_feats)
        self.conv2 = RRRB(n_feats)

    def forward(self, x):
        res = self.conv1(x)
        res = act(res)
        res = self.conv2(res)

        return res


class HFAB(nn.Module):
    def __init__(self, n_feats, up_blocks, mid_feats):
        super(HFAB, self).__init__()
        self.squeeze = nn.Conv2d(n_feats, mid_feats, 3, 1, 1)
        convs = [ERB(mid_feats) for _ in range(up_blocks)]
        self.convs = nn.Sequential(*convs)
        self.excitate = nn.Conv2d(mid_feats, n_feats, 3, 1, 1)
    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = act(self.squeeze(x))
        out = act(self.convs(out))
        out = self.excitate(out)
        out = self.sigmoid(out)
        out *= x

        return out


class TEST_FMEN(nn.Module):
    def __init__(self, args):
        super(TEST_FMEN, self).__init__()

        self.down_blocks = args.down_blocks

        up_blocks = args.up_blocks
        mid_feats = args.mid_feats
        n_feats = args.n_feats
        n_colors = args.n_colors
        scale = args.scale[0]

        # define head module
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # warm up
        self.warmup = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            HFAB(n_feats, up_blocks[0], mid_feats-4)
        )

        # define body module
        ERBs = [ERB(n_feats) for _ in range(self.down_blocks)]
        HFABs  = [HFAB(n_feats, up_blocks[i+1], mid_feats) for i in range(self.down_blocks)]

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

        self.scale = [4]
        self.rgb_range = 255
        self.n_colors = 3

if __name__ == '__main__':
    args = Args()
    model = TEST_FMEN(args).to('cuda')
    in_ = torch.randn(1, 3, round(720/args.scale[0]), round(1280/args.scale[0])).to('cuda')
    summary(model, in_)
