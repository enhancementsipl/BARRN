import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift


class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, act_type, norm_type):
        super(FeedbackBlock, self).__init__()

        stride = 1
        padding = 2
        kernel_size = 3

        self.num_groups = num_groups

        self.compress_in = ConvBlock(2 * num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.convBlocks = nn.ModuleList()
        self.diconvBlocks_s = nn.ModuleList()
        self.diconvBlocks_l = nn.ModuleList()
        self.fusionBlocks = nn.ModuleList()
        self.convtranBlocks = nn.ModuleList()
        self.diconvtranBlocks_s = nn.ModuleList()
        self.diconvtranBlocks_l = nn.ModuleList()

        for idx in range(self.num_groups):
            self.convBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride,
                                             act_type=act_type, norm_type=norm_type, dilation=1))
            self.diconvBlocks_s.append(ConvBlock(num_features, num_features,
                                                 kernel_size=kernel_size, stride=stride,
                                                 act_type=act_type, norm_type=norm_type, dilation=2))
            self.diconvBlocks_l.append(ConvBlock(num_features, num_features,
                                                 kernel_size=kernel_size, stride=stride,
                                                 act_type=act_type, norm_type=norm_type, dilation=3))
            self.fusionBlocks.append(ConvBlock(num_features * 3, num_features,
                                               kernel_size=1, stride=1,
                                               act_type=act_type, norm_type=norm_type, dilation=1))
            if idx > 0:
                self.convtranBlocks.append(ConvBlock(num_features * (idx + 1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))
                self.diconvtranBlocks_s.append(ConvBlock(num_features * (idx + 1), num_features,
                                                         kernel_size=1, stride=1,
                                                         act_type=act_type, norm_type=norm_type))
                self.diconvtranBlocks_l.append(ConvBlock(num_features * (idx + 1), num_features,
                                                         kernel_size=1, stride=1,
                                                         act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock(num_groups * num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        conv_features = []
        biconv_features_s = []
        biconv_features_l = []
        fusion_features = []
        conv_features.append(x)
        biconv_features_s.append(x)
        biconv_features_l.append(x)

        for idx in range(self.num_groups):
            LD_C = torch.cat(tuple(conv_features), 1)
            LD_D_s = torch.cat(tuple(biconv_features_s), 1)
            LD_D_l = torch.cat(tuple(biconv_features_l), 1)
            if idx > 0:
                LD_C = self.convtranBlocks[idx - 1](LD_C)
                LD_D_s = self.diconvtranBlocks_s[idx - 1](LD_D_s)
                LD_D_l = self.diconvtranBlocks_l[idx - 1](LD_D_l)

            LD_C = self.convBlocks[idx](LD_C)
            LD_D_s = self.diconvBlocks_s[idx](LD_D_s)
            LD_D_l = self.diconvBlocks_l[idx](LD_D_l)

            conv_features.append(LD_C)
            biconv_features_s.append(LD_D_s)
            biconv_features_l.append(LD_D_l)

            LD_F = self.fusionBlocks[idx](torch.cat((LD_C, LD_D_s, LD_D_l), 1))
            fusion_features.append(LD_F)

        output = torch.cat(tuple(fusion_features), 1)
        output = self.compress_out(output)

        del conv_features
        del biconv_features_s
        del biconv_features_l
        del fusion_features

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True


class SRFBN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, act_type='prelu',
                 norm_type=None):
        super(SRFBN, self).__init__()

        self.num_steps = num_steps
        self.num_features = num_features

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.conv_in = ConvBlock(in_channels, 4 * num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        self.block = FeedbackBlock(num_features, num_groups, act_type, norm_type)

        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        self._reset_state()

        x = self.sub_mean(x)

        x_c = self.conv_in(x)
        x_c = self.feat_in(x_c)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x_c)

            h = torch.add(x, self.conv_out(h))
            out_1 = self.add_mean(h)
            outs.append(out_1)

        return outs

    def _reset_state(self):
        self.block.reset_state()


class FeedbackBlock_add(nn.Module):
    def __init__(self, num_features, num_groups, act_type, norm_type):
        super(FeedbackBlock_add, self).__init__()

        stride = 1
        padding = 2
        kernel_size = 3

        self.num_groups = num_groups

        self.compress_in = ConvBlock( num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.convBlocks = nn.ModuleList()
        self.diconvBlocks_s = nn.ModuleList()
        self.diconvBlocks_l = nn.ModuleList()
        self.fusionBlocks = nn.ModuleList()
        self.convtranBlocks = nn.ModuleList()
        self.diconvtranBlocks_s = nn.ModuleList()
        self.diconvtranBlocks_l = nn.ModuleList()

        for idx in range(self.num_groups):
            self.convBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride,
                                             act_type=act_type, norm_type=norm_type, dilation=1))
            self.diconvBlocks_s.append(ConvBlock(num_features, num_features,
                                                 kernel_size=kernel_size, stride=stride,
                                                 act_type=act_type, norm_type=norm_type, dilation=2))
            self.diconvBlocks_l.append(ConvBlock(num_features, num_features,
                                                 kernel_size=kernel_size, stride=stride,
                                                 act_type=act_type, norm_type=norm_type, dilation=3))
            self.fusionBlocks.append(ConvBlock(num_features , num_features,
                                               kernel_size=1, stride=1,
                                               act_type=act_type, norm_type=norm_type, dilation=1))
            if idx > 0:
                self.convtranBlocks.append(ConvBlock(num_features , num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))
                self.diconvtranBlocks_s.append(ConvBlock(num_features , num_features,
                                                         kernel_size=1, stride=1,
                                                         act_type=act_type, norm_type=norm_type))
                self.diconvtranBlocks_l.append(ConvBlock(num_features , num_features,
                                                         kernel_size=1, stride=1,
                                                         act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock(num_groups*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.add(x, self.last_hidden)
        x = self.compress_in(x)

        conv_features = []
        biconv_features_s = []
        biconv_features_l = []
        fusion_features = []
        conv_features.append(x)
        biconv_features_s.append(x)
        biconv_features_l.append(x)
        LD_C = 0
        LD_D_s = 0
        LD_D_l = 0
        for idx in range(self.num_groups):
            for fe in conv_features:
                LD_C=torch.add(LD_C,fe)
            for fe in biconv_features_s:
                LD_D_s=torch.add(LD_D_s,fe)
            for fe in biconv_features_l:
                LD_D_l=torch.add(LD_D_l,fe)

            if idx > 0:
                LD_C = self.convtranBlocks[idx - 1](LD_C)
                LD_D_s = self.diconvtranBlocks_s[idx - 1](LD_D_s)
                LD_D_l = self.diconvtranBlocks_l[idx - 1](LD_D_l)

            LD_C = self.convBlocks[idx](LD_C)
            LD_D_s = self.diconvBlocks_s[idx](LD_D_s)
            LD_D_l = self.diconvBlocks_l[idx](LD_D_l)

            conv_features.append(LD_C)
            biconv_features_s.append(LD_D_s)
            biconv_features_l.append(LD_D_l)

            LD_F = self.fusionBlocks[idx](torch.add(torch.add(LD_C, LD_D_s), LD_D_l))
            fusion_features.append(LD_F)

        output = torch.cat(tuple(fusion_features), 1)
        output = self.compress_out(output)

        del conv_features
        del biconv_features_s
        del biconv_features_l
        del fusion_features

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True


class SRFBN_add(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, act_type='prelu',
                 norm_type=None):
        super(SRFBN_add, self).__init__()

        self.num_steps = num_steps
        self.num_features = num_features

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.conv_in = ConvBlock(in_channels, 4 * num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        self.block = FeedbackBlock_add(num_features, num_groups, act_type, norm_type)

        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        self._reset_state()

        x = self.sub_mean(x)

        x_c = self.conv_in(x)
        x_c = self.feat_in(x_c)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x_c)

            h = torch.add(x, self.conv_out(h))
            out_1 = self.add_mean(h)
            outs.append(out_1)

        return outs

    def _reset_state(self):
        self.block.reset_state()


