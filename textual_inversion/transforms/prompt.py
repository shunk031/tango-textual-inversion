import random
from typing import Literal, Union, get_args

import torch
from tango.common import Registrable
from transformers import CLIPTokenizer, CLIPTokenizerFast

from textual_inversion.templates import (
    IMAGENET_STYLE_TEMPLATES_SMALL,
    IMAGENET_TEMPLATES_SMALL,
)

LearnableProperty = Literal["object", "style"]


class TransformPrompt(Registrable):
    def __init__(
        self,
        placeholder_string: str,
        tokenizer: Union[CLIPTokenizer, CLIPTokenizerFast],
        learnable_property: LearnableProperty,
    ) -> None:
        super().__init__()
        assert isinstance(tokenizer, (CLIPTokenizer, CLIPTokenizerFast))

        self.placeholder_string = placeholder_string
        self.tokenizer = tokenizer

        assert learnable_property in get_args(LearnableProperty), learnable_property
        self.templates = (
            IMAGENET_STYLE_TEMPLATES_SMALL
            if learnable_property == "style"
            else IMAGENET_TEMPLATES_SMALL
        )

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError


@TransformPrompt.register("textual_inversion")
class TextualInversionTransformPrompt(TransformPrompt):
    def __init__(
        self,
        placeholder_string: str,
        tokenizer: Union[CLIPTokenizer, CLIPTokenizerFast],
        learnable_property: LearnableProperty = "object",
    ) -> None:
        super().__init__(placeholder_string, tokenizer, learnable_property)

    def __call__(self) -> torch.Tensor:
        text = random.choice(self.templates).format(self.placeholder_string)

        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return input_ids
