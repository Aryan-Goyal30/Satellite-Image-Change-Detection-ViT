"""Siamese U-Net for binary change detection.

Design, in one paragraph (this is the viva answer):

    A single ImageNet-pretrained ResNet encoder processes BOTH dates with SHARED
    weights, so the two images are mapped into the same feature space. At each of
    the five encoder scales we fuse the two feature maps with

        fuse = conv1x1( concat[ |f_a - f_b| , f_a + f_b ] )

    The absolute difference carries the change signal; the sum carries the scene
    context needed to tell a real structural change from an illumination shift.
    A standard U-Net decoder then upsamples the fused pyramid back to input
    resolution and emits one logit per pixel.

Weight sharing matters: it is what makes the model symmetric and forces both
dates into a comparable representation, instead of learning two unrelated
encoders that happen to be concatenated.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ENCODERS = {
    "resnet18": (torchvision.models.resnet18,
                 torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                 [64, 64, 128, 256, 512]),
    "resnet34": (torchvision.models.resnet34,
                 torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
                 [64, 64, 128, 256, 512]),
    "resnet50": (torchvision.models.resnet50,
                 torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
                 [64, 256, 512, 1024, 2048]),
}


class ResNetEncoder(nn.Module):
    """ResNet trunk exposed as a 5-level feature pyramid (strides 2,4,8,16,32)."""

    def __init__(self, name="resnet34", pretrained=True):
        super().__init__()
        ctor, weights, self.channels = ENCODERS[name]
        net = ctor(weights=weights if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # /2
        self.pool = net.maxpool                                   # /4
        self.layer1, self.layer2 = net.layer1, net.layer2         # /4, /8
        self.layer3, self.layer4 = net.layer3, net.layer4         # /16, /32

    def forward(self, x):
        f0 = self.stem(x)              # /2
        f1 = self.layer1(self.pool(f0))  # /4
        f2 = self.layer2(f1)           # /8
        f3 = self.layer3(f2)           # /16
        f4 = self.layer4(f3)           # /32
        return [f0, f1, f2, f3, f4]


class Fuse(nn.Module):
    """Combine the two dates at one scale: conv1x1(cat[|a-b|, a+b]) -> C."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, fa, fb):
        return self.block(torch.cat([(fa - fb).abs(), fa + fb], dim=1))


class DecoderBlock(nn.Module):
    """Upsample, concat the skip, then two 3x3 convs."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # guard against odd sizes
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """Binary change detection. Input: two RGB tensors. Output: 1-channel logits."""

    NAME = "siamese-unet"

    def __init__(self, encoder="resnet34", pretrained=True, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        self.encoder_name = encoder
        self.encoder = ResNetEncoder(encoder, pretrained)
        ch = self.encoder.channels
        self.fuses = nn.ModuleList([Fuse(c) for c in ch])

        d = list(decoder_channels)
        # bottleneck (/32) -> /16 -> /8 -> /4 -> /2 -> /1
        self.dec4 = DecoderBlock(ch[4], ch[3], d[0])
        self.dec3 = DecoderBlock(d[0], ch[2], d[1])
        self.dec2 = DecoderBlock(d[1], ch[1], d[2])
        self.dec1 = DecoderBlock(d[2], ch[0], d[3])
        self.dec0 = DecoderBlock(d[3], 0, d[4])
        self.head = nn.Conv2d(d[4], 1, 1)

    def forward(self, a, b):
        fa = self.encoder(a)
        fb = self.encoder(b)                      # SHARED weights
        s = [fuse(x, y) for fuse, x, y in zip(self.fuses, fa, fb)]

        x = self.dec4(s[4], s[3])
        x = self.dec3(x, s[2])
        x = self.dec2(x, s[1])
        x = self.dec1(x, s[0])
        x = self.dec0(x)
        return self.head(x)                       # B,1,H,W logits

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(encoder="resnet34", pretrained=True):
    return SiameseUNet(encoder=encoder, pretrained=pretrained)


if __name__ == "__main__":
    m = build_model()
    a = torch.randn(2, 3, 256, 256)
    out = m(a, torch.randn(2, 3, 256, 256))
    print("output:", tuple(out.shape), "| params:", f"{m.n_params/1e6:.2f}M")
    assert out.shape == (2, 1, 256, 256), out.shape
    print("OK")
