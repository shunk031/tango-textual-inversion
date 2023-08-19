from typing import Literal, get_args

import datasets
from tango import Step
from tango.integrations.transformers import Tokenizer
from transformers import CLIPTokenizer

from textual_inversion.helper.data import (
    IMAGENET_STYLE_TEMPLATES_SMALL,
    IMAGENET_TEMPLATES_SMALL,
    preprocess_examples,
)

LearnableProperty = Literal["object", "style"]


@Step.register("transform_data")
class TransformData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True

    def run(  # type: ignore
        self,
        dataset: datasets.DatasetDict,
        tokenizer: Tokenizer,
        placeholder_token: str,
        learnable_property: LearnableProperty = "object",
        image_size: int = 512,
        flip_proba: float = 0.5,
        interpolation: str = "bicubic",
        is_center_crop: bool = False,
    ) -> datasets.DatasetDict:
        assert isinstance(tokenizer, CLIPTokenizer), tokenizer
        assert learnable_property in get_args(LearnableProperty)

        templates = (
            IMAGENET_STYLE_TEMPLATES_SMALL
            if learnable_property == "style"
            else IMAGENET_TEMPLATES_SMALL
        )

        def transforms(raw_examples):
            return preprocess_examples(
                raw_examples=raw_examples,
                image_size=image_size,
                interpolation=interpolation,
                is_center_crop=is_center_crop,
                flip_proba=flip_proba,
                tokenizer=tokenizer,
                templates=templates,
                placeholder_string=placeholder_token,
            )

        dataset = dataset.with_transform(transforms)
        return dataset
