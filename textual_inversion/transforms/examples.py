from typing import Any

from tango.common import Registrable

from textual_inversion.transforms import TransformImage, TransformPrompt


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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        breakpoint()
