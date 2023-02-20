import torch
from torch import nn
import torch.nn.functional as F

#Build a nn.Sequential with two conv-BatchN-Relu layers
def baseblock(channel_in, channel_out):
    #     Design a baseblock with conv-batch-relu x 2 (each input is twice convolved as in fig.)
    return nn.Sequential(
        # your code here
        nn.Conv3d(in_channels=channel_in,
                  out_channels=channel_out,
                  kernel_size=3,
                  padding=1
                  ),
        nn.BatchNorm3d(num_features=channel_out),
        nn.ReLU(),

        nn.Conv3d(in_channels=channel_out,
                  out_channels=channel_out,
                  kernel_size=3,
                  padding=1
                  ),
        nn.BatchNorm3d(num_features=channel_out),
        nn.ReLU(),

    )


# Build a downscaling module [Hint: use the above layeredConv after that]
# Add a maxpool before baseblock as in figure
def downsamplePart(channel_in, channel_out):
    return nn.Sequential(
        # your code,
        nn.MaxPool3d(kernel_size=2),
        baseblock(channel_in, channel_out)
    )


# Build a upscaling module [Hint: use the above layeredConv after that]
# - Remember there is also concatenation and size may change so we are padding
class upsampledPart(nn.Module):
    def __init__(self, channel_in, channel_out, bilinear=True):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.up = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=2, stride=2)
        self.conv = baseblock(channel_in, channel_out)

    def forward(self, x1, x2):
        # upscale and then pad to eliminate any difference between upscaled and other feature map coming with skip connection
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        # concatenate (perform concatenation of x1 and x2 --> remember these are skip x2(from encoder) and upssampled image x1)
        x = torch.cat([x2, x1], dim=1)

        # apply baseblock after concatenation --> you do again two convs.? --> baseblock
        x = self.conv(x)

        return x


# Step 4: Compile all of above together
# here output channel should be equal to number of classes
class UNet(nn.Module):
    def __init__(self, channel_in=1, channel_out=3, bilinear=None):
        super(UNet, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out

        # call your base block
        self.initial = baseblock(channel_in, 64)

        # downsampling layers with 2 conv layers
        self.down1 = downsamplePart(64, 128)
        self.down2 = downsamplePart(128, 256)
        self.down3 = downsamplePart(256, 512)
        self.down4 = downsamplePart(512, 1024)

        # your code here
        # upsampling layers with feature concatenation and 2 conv layers
        self.up1 = upsampledPart(1024, 512)
        self.up2 = upsampledPart(512, 256)
        self.up3 = upsampledPart(256, 128)
        self.up4 = upsampledPart(128, 64)

        # output layer
        self.out = nn.Conv3d(64, channel_out, kernel_size=1)

        # build a forward pass here

    # remember to keep your output as you will need to concatenate later in upscaling
    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # your code here for upscaling
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # output
        return self.out(x)