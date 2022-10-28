import random
from typing import Dict, List, Optional, TypedDict

import datasets
import numpy as np
import PIL
import torch
from PIL import Image
from PIL.Image import Image as PilImage
from tango import Step
from tango.integrations.transformers import Tokenizer
from torchvision import transforms
from transformers import CLIPTokenizer


class PreprocessedExamples(TypedDict):
    pixel_values: List[torch.Tensor]
    input_ids: List[torch.Tensor]


@Step.register("transform_data")
class TransformData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True

    INTERPOLATION: Dict[str, int] = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
    }
    IMAGENET_TEMPLATES_SMALL: List[str] = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]
    IMAGENET_STYLE_TEMPLATES_SMALL: List[str] = [
        "a painting in the style of {}",
        "a rendering in the style of {}",
        "a cropped painting in the style of {}",
        "the painting in the style of {}",
        "a clean painting in the style of {}",
        "a dirty painting in the style of {}",
        "a dark painting in the style of {}",
        "a picture in the style of {}",
        "a cool painting in the style of {}",
        "a close-up painting in the style of {}",
        "a bright painting in the style of {}",
        "a cropped painting in the style of {}",
        "a good painting in the style of {}",
        "a close-up painting in the style of {}",
        "a rendition in the style of {}",
        "a nice painting in the style of {}",
        "a small painting in the style of {}",
        "a weird painting in the style of {}",
        "a large painting in the style of {}",
    ]

    def preprocess_image(
        self,
        image_pl: PilImage,
        image_size: int,
        interpolation: str,
        is_center_crop: bool,
        flip_proba: float,
    ) -> torch.Tensor:

        if not image_pl.mode == "RGB":
            image_pl = image_pl.convert("RGB")

        image_np = np.array(image_pl).astype(np.uint8)
        if is_center_crop:
            crop = min(image_np.shape[0], image_np.shape[1])
            h, w = image_np.shape
            image_np = image_np[
                (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
            ]

        image_pl = Image.fromarray(image_np)
        image_pl = image_pl.resize(
            (image_size, image_size), resample=self.INTERPOLATION[interpolation]
        )
        image_pl = transforms.RandomHorizontalFlip(p=flip_proba)(image_pl)

        image_np = np.array(image_pl).astype(np.uint8)
        image_np = (image_np / 127.5 - 1.0).astype(np.float32)
        image_th = torch.from_numpy(image_np).permute(2, 0, 1)
        return image_th

    def preprocess_prompt(
        self,
        tokenizer: Tokenizer,
        placeholder_string: str,
        templates: List[str],
    ) -> torch.Tensor:
        text = random.choice(templates).format(placeholder_string)
        input_ids = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return input_ids

    def raw_example_transformer(
        self,
        raw_examples,
        image_size: int,
        interpolation: str,
        is_center_crop: bool,
        flip_proba: float,
        tokenizer: Tokenizer,
        templates: List[str],
        placeholder_string: str,
    ) -> PreprocessedExamples:

        preprocessed_examples: PreprocessedExamples = {
            "pixel_values": [
                self.preprocess_image(
                    img,
                    image_size=image_size,
                    interpolation=interpolation,
                    is_center_crop=is_center_crop,
                    flip_proba=flip_proba,
                )
                for img in raw_examples["image"]
            ],
            "input_ids": [
                self.preprocess_prompt(
                    tokenizer=tokenizer,
                    placeholder_string=placeholder_string,
                    templates=templates,
                )
                for _ in range(len(raw_examples["image"]))
            ],
        }
        return preprocessed_examples

    def run(  # type: ignore
        self,
        dataset: datasets.DatasetDict,
        tokenizer: Tokenizer,
        placeholder_token: str,
        learnable_property: str = "object",
        image_size: int = 512,
        flip_proba: float = 0.5,
        interpolation: str = "bicubic",
        is_center_crop: bool = False,
    ) -> datasets.DatasetDict:

        assert isinstance(tokenizer, CLIPTokenizer), tokenizer
        templates = (
            self.IMAGENET_STYLE_TEMPLATES_SMALL
            if learnable_property == "style"
            else self.IMAGENET_TEMPLATES_SMALL
        )

        def raw_example_transformer_wrapper(raw_examples):
            return self.raw_example_transformer(
                raw_examples=raw_examples,
                image_size=image_size,
                interpolation=interpolation,
                is_center_crop=is_center_crop,
                flip_proba=flip_proba,
                tokenizer=tokenizer,
                templates=templates,
                placeholder_string=placeholder_token,
            )

        dataset = dataset.with_transform(raw_example_transformer_wrapper)
        return dataset
