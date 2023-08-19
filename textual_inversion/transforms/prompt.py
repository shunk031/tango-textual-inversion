import random
from typing import Literal, get_args

import torch
from tango.common import Registrable
from tango.integrations.transformers import Tokenizer

from textual_inversion.templates import (
    IMAGENET_STYLE_TEMPLATES_SMALL,
    IMAGENET_TEMPLATES_SMALL,
)

LearnableProperty = Literal["object", "style"]


class TransformPrompt(Registrable):
    def __init__(
        self,
        placeholder_string: str,
        tokenizer: Tokenizer,
        learnable_property: LearnableProperty,
    ) -> None:
        super().__init__()
        self.placeholder_string = placeholder_string
        self.tokenizer = tokenizer

        assert learnable_property in get_args(LearnableProperty)
        self.templates = (
            IMAGENET_STYLE_TEMPLATES_SMALL
            if learnable_property == "style"
            else IMAGENET_TEMPLATES_SMALL
        )


@TransformPrompt.register("textual_inversion")
class TextualInversionTransformPrompt(TransformPrompt):
    def __init__(
        self,
        placeholder_string: str,
        tokenizer: Tokenizer,
        learnable_property: LearnableProperty = "object",
    ) -> None:
        super().__init__(placeholder_string, tokenizer, learnable_property)

    def __call__(self, placeholder_string: str) -> torch.Tensor:
        text = random.choice(self.templates).format(placeholder_string)

        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return input_ids
