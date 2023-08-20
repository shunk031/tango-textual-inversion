from typing import Any, List, TypedDict

import torch
from PIL.Image import Image as PilImage
from tango.common import Registrable

from textual_inversion.transforms import TransformImage, TransformPrompt


class RawExamples(TypedDict):
    image: List[PilImage]


class PreprocessedExamples(TypedDict):
    pixel_values: List[torch.Tensor]
    input_ids: List[torch.Tensor]


class TransformExamples(Registrable):
    def __init__(
        self,
        image_transformer: TransformImage,
        prompt_transformer: TransformPrompt,
    ) -> None:
        super().__init__()
        self.image_transformer = image_transformer
        self.prompt_transformer = prompt_transformer

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


@TransformExamples.register("textual_inversion")
class TextualInversionTransformExamples(TransformExamples):
    def __init__(
        self, image_transformer: TransformImage, prompt_transformer: TransformPrompt
    ) -> None:
        super().__init__(image_transformer, prompt_transformer)

    def __call__(self, examples: RawExamples) -> PreprocessedExamples:
        pixel_values = [
            self.image_transformer(image_pl=image) for image in examples["image"]
        ]
        input_ids = [self.prompt_transformer() for _ in range(len(examples["image"]))]

        assert len(pixel_values) == len(input_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }
