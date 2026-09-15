"""Spectral Siamese U-Net for environmental change detection (Phase 4B).

Architecture is the built-environment detector's, unchanged in its essentials:
one SHARED encoder maps both dates into a common feature space, the two feature
pyramids are fused at every scale with

    fuse = conv1x1( concat[ |f_before - f_after| , f_before + f_after ] )

and a U-Net decoder returns one logit per pixel at 256 x 256.

The only thing that varies across the Phase 4B experiments is the INPUT STEM,
which is built for whatever band subset the experiment uses:

    E0  B02 B03 B04              (RGB)
    E1  B02 B03 B04 B08          (RGB + NIR)
    E2  B02 B03 B04 B08 B11 B12  (six-band baseline)

Stem construction, applied identically to every experiment
----------------------------------------------------------
    B02 (blue)  <- pretrained blue  filter   x SCALE
    B03 (green) <- pretrained green filter   x SCALE
    B04 (red)   <- pretrained red   filter   x SCALE
    B08/B11/B12 <- MEAN of the pretrained RGB filters x SCALE

    SCALE = 3 / number_of_bands

The visible bands are mapped by WAVELENGTH, not by position: torchvision stores
conv1 in RGB order while this project's band order is blue-first, so the mapping
is deliberately reversed.

SCALE is magnitude normalisation, not a handicap. A convolution sums over its
input channels, so a stem reading n channels accumulates n contributions where
the pretrained stem accumulated three; scaling by 3/n keeps the expected
response magnitude comparable. It follows that the RGB experiment gets scale 1.0
and therefore the pretrained filters exactly - which is the correct, fair
initialisation for it, arrived at by the same formula rather than by special
treatment.

What is NOT claimed
-------------------
ImageNet pretraining supplies NO semantic knowledge about NIR or SWIR
reflectance. Those filters begin as achromatic edge detectors and any spectral
structure in them has to be learned from the TMF data. The visible mapping is
wavelength-aligned and defensible; the rest is a neutral starting point.

Layer reuse
-----------
Everything after conv1 - bn1, layer1..layer4 - is reused from the ImageNet
checkpoint unchanged in every experiment. conv1 alone is replaced. The fusion
blocks, decoder and head are trained from random initialisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.siamese_unet import SiameseUNet

#: Canonical full band order. Matches environment.data.sentinel2.BANDS.
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

#: The three Phase 4B band sets.
BAND_SETS = {
    "rgb": ("B02", "B03", "B04"),
    "rgb_nir": ("B02", "B03", "B04", "B08"),
    "six_band": BANDS,
}

#: Index of each pretrained ImageNet conv1 channel, which is stored RGB.
_IMAGENET_RGB = {"R": 0, "G": 1, "B": 2}

#: Which pretrained filter seeds each band. None means "achromatic mean".
STEM_SOURCE = {"B02": "B", "B03": "G", "B04": "R",
               "B08": None, "B11": None, "B12": None}


def stem_scale(n_bands: int) -> float:
    """Magnitude normalisation for a stem reading `n_bands` channels."""
    return 3.0 / n_bands


def build_stem(pretrained_conv1: nn.Conv2d, bands) -> nn.Conv2d:
    """Rebuild a 3-channel ImageNet conv1 for an arbitrary band subset."""
    scale = stem_scale(len(bands))
    stem = nn.Conv2d(len(bands), pretrained_conv1.out_channels,
                     kernel_size=pretrained_conv1.kernel_size,
                     stride=pretrained_conv1.stride,
                     padding=pretrained_conv1.padding,
                     bias=pretrained_conv1.bias is not None)
    with torch.no_grad():
        source = pretrained_conv1.weight.data
        achromatic = source.mean(dim=1, keepdim=True)
        for index, band in enumerate(bands):
            key = STEM_SOURCE[band]
            seed = (source[:, _IMAGENET_RGB[key]: _IMAGENET_RGB[key] + 1]
                    if key is not None else achromatic)
            stem.weight.data[:, index: index + 1] = seed * scale
        if pretrained_conv1.bias is not None:
            stem.bias.data.copy_(pretrained_conv1.bias.data)
    return stem


class SpectralSiameseUNet(SiameseUNet):
    """SiameseUNet with a band-subset input stem. Everything else is inherited."""

    def __init__(self, bands=BANDS, encoder: str = "resnet34",
                 pretrained: bool = True, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__(encoder=encoder, pretrained=pretrained,
                         decoder_channels=decoder_channels)
        self.bands = tuple(bands)
        self.encoder.stem[0] = build_stem(self.encoder.stem[0], self.bands)
        self.in_channels = len(self.bands)
        self.pretrained_encoder = pretrained

    @property
    def NAME(self):
        return f"environment-siamese-unet-{self.in_channels}band"

    def describe(self) -> dict:
        """Exactly how this model was built, for the experiment manifest."""
        return {
            "name": self.NAME,
            "encoder": self.encoder_name,
            "in_channels": self.in_channels,
            "bands": list(self.bands),
            "fusion": "conv1x1(concat[|f_before - f_after|, f_before + f_after])",
            "weight_sharing": "single encoder applied to both dates",
            "stem_initialisation": {b: (STEM_SOURCE[b] or "achromatic mean of RGB")
                                    for b in self.bands},
            "stem_scale": stem_scale(self.in_channels),
            "stem_scale_rule": "3 / number_of_bands (magnitude normalisation)",
            "pretrained_layers_reused": ["bn1", "layer1", "layer2", "layer3", "layer4"],
            "pretrained_layers_replaced": [
                f"conv1 (3ch -> {self.in_channels}ch, re-initialised)"],
            "randomly_initialised": ["fuse blocks", "decoder", "head"],
            "imagenet_semantics_claimed_for_nir_swir": False,
            "parameters": self.n_params,
        }


def build_model(bands=BANDS, encoder: str = "resnet34",
                pretrained: bool = True) -> SpectralSiameseUNet:
    return SpectralSiameseUNet(bands=bands, encoder=encoder, pretrained=pretrained)


if __name__ == "__main__":
    for key, bands in BAND_SETS.items():
        model = build_model(bands=bands)
        out = model(torch.randn(2, len(bands), 256, 256),
                    torch.randn(2, len(bands), 256, 256))
        assert out.shape == (2, 1, 256, 256), out.shape
        print(f"{key:<9} {len(bands)} bands  scale {stem_scale(len(bands)):.4f}  "
              f"params {model.n_params/1e6:.2f}M  out {tuple(out.shape)}")
