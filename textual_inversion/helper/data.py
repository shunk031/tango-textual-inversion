import random
from typing import Final, List, TypedDict

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers.utils import PIL_INTERPOLATION
from PIL import Image
from PIL.Image import Image as PilImage
from tango.integrations.transformers import Tokenizer


class PreprocessedExamples(TypedDict):
    pixel_values: List[torch.Tensor]
    input_ids: List[torch.Tensor]


def preprocess_examples(
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
            preprocess_image(
                img,
                image_size=image_size,
                interpolation=interpolation,
                is_center_crop=is_center_crop,
                flip_proba=flip_proba,
            )
            for img in raw_examples["image"]
        ],
        "input_ids": [
            preprocess_prompt(
                tokenizer=tokenizer,
                placeholder_string=placeholder_string,
                templates=templates,
            )
            for _ in range(len(raw_examples["image"]))
        ],
    }
    return preprocessed_examples
