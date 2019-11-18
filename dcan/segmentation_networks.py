import torch
import torch.nn as nn


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model


def up_conv(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3,
                           stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model


def double_conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
    )
    return model


def out_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
    )
    return model


class UNet(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(UNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.ReLU(inplace=True)

        self.down_1 = double_conv_block(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.down_2 = double_conv_block(
            self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.down_3 = double_conv_block(
            self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.down_4 = double_conv_block(
            self.num_filter * 4, self.num_filter * 8, act_fn)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        self.bridge = double_conv_block(
            self.num_filter * 8, self.num_filter * 16, act_fn)

        self.trans_1 = up_conv(self.num_filter * 16,
                               self.num_filter * 16, act_fn)
        self.up_1 = double_conv_block(
            self.num_filter * 24, self.num_filter * 8, act_fn)

        self.trans_2 = up_conv(self.num_filter * 8,
                               self.num_filter * 8, act_fn)
        self.up_2 = double_conv_block(
            self.num_filter * 12, self.num_filter * 4, act_fn)

        self.trans_3 = up_conv(self.num_filter * 4,
                               self.num_filter * 4, act_fn)
        self.up_3 = double_conv_block(
            self.num_filter * 6, self.num_filter*2, act_fn)

        self.trans_4 = up_conv(self.num_filter * 2,
                               self.num_filter * 2, act_fn)
        self.up_4 = double_conv_block(
            self.num_filter * 3, self.num_filter, act_fn)


        self.out = out_block(self.num_filter, out_dim)

    def forward(self, x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.up_4(concat_4)

        out = self.out(up_4)

        return out


# Deep Contour Aware Network (DCAN)
class dcanConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_ratio=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(p=dropout_ratio),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class dcanDeConv(nn.Module):
    def __init__(self, in_ch, out_ch, upscale_factor, dropout_ratio=0.2):
        super().__init__()
        self.upscaling = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=upscale_factor, stride=upscale_factor)
        self.conv = dcanConv(out_ch, out_ch, dropout_ratio)

    def forward(self, x):
        x = self.upscaling(x)
        x = self.conv(x)
        return x


class DCAN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = dcanConv(n_channels, 64)
        self.conv2 = dcanConv(64, 128)
        self.conv3 = dcanConv(128, 256)
        self.conv4 = dcanConv(256, 512)
        self.conv5 = dcanConv(512, 512)
        self.conv6 = dcanConv(512, 1024)
        self.deconv3s = dcanDeConv(512, n_classes, 8)  # 8 = 2^3 (3 maxpooling)
        self.deconv3c = dcanDeConv(512, n_classes, 8)
        # 16 = 2^4 (4 maxpooling)
        self.deconv2s = dcanDeConv(512, n_classes, 16)
        self.deconv2c = dcanDeConv(512, n_classes, 16)
        # 32 = 2^5 (5 maxpooling)
        self.deconv1s = dcanDeConv(1024, n_classes, 32)
        self.deconv1c = dcanDeConv(1024, n_classes, 32)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.maxpool(c1))
        c3 = self.conv3(self.maxpool(c2))
        c4 = self.conv4(self.maxpool(c3))
        # s for segment branch, c for contour branch
        u3s = self.deconv3s(c4)
        u3c = self.deconv3c(c4)
        c5 = self.conv5(self.maxpool(c4))
        u2s = self.deconv2s(c5)
        u2c = self.deconv2c(c5)
        c6 = self.conv6(self.maxpool(c5))
        u1s = self.deconv1s(c6)
        u1c = self.deconv1c(c6)
        outs = F.sigmoid(u1s + u2s + u3s)
        outc = F.sigmoid(u1c + u2c + u3c)
        return outs, outc


# net = UNet(3,1,64)

# inputs = torch.ones((3, 3, 400, 400))
# out = net(inputs)
# print(out.shape)
