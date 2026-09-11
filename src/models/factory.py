"""Point d'entrée unique pour instancier un modèle de segmentation par son nom."""

from torch import nn

from src.models.unet import UNet

_SUPPORTED = ("unet_scratch", "unet_resnet34")


def build_model(name: str, in_channels: int = 3, num_classes: int = 1) -> nn.Module:
    if name == "unet_scratch":
        return UNet(in_channels=in_channels, num_classes=num_classes, base_channels=32)

    if name == "unet_resnet34":
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    raise ValueError(f"Modèle inconnu : {name!r}. Choix possibles : {_SUPPORTED}")
