from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers.utils import PIL_INTERPOLATION
from PIL import Image
from PIL.Image import Image as PilImage
from tango.common import Registrable


class TransformImage(Registrable):
    def __init__(
        self,
        image_size: int = 512,
        interpolation: str = "bicubic",
        is_center_crop: bool = False,
        flip_proba: float = 0.5,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.interpolation = interpolation
        self.flip_proba = flip_proba
        self.is_center_crop = is_center_crop

    def __call__(self, image_pl: PilImage) -> Any:
        raise NotImplementedError


@TransformImage.register("textual_inversion")
class TextualInversionTransformImage(TransformImage):
    def __init__(
        self,
        image_size: int = 512,
        interpolation: str = "bicubic",
        is_center_crop: bool = False,
        flip_proba: float = 0.5,
    ) -> None:
        super().__init__(image_size, interpolation, is_center_crop, flip_proba)

    def __call__(self, image_pl: PilImage) -> torch.Tensor:
        if not image_pl.mode == "RGB":
            image_pl = image_pl.convert("RGB")

        image_np = np.array(image_pl).astype(np.uint8)
        if self.is_center_crop:
            crop = min(image_np.shape[0], image_np.shape[1])
            h, w = image_np.shape
            image_np = image_np[
                (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
            ]

        image_pl = Image.fromarray(image_np)
        image_pl = image_pl.resize(
            (self.image_size, self.image_size),
            resample=PIL_INTERPOLATION[self.interpolation],
        )
        image_pl = transforms.RandomHorizontalFlip(p=self.flip_proba)(image_pl)

        image_np = np.array(image_pl).astype(np.uint8)
        image_np = (image_np / 127.5 - 1.0).astype(np.float32)
        image_th = torch.from_numpy(image_np).permute(2, 0, 1)
        return image_th
