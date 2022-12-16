from pathlib import Path
from typing import Generic, TypeVar

from diffusers import StableDiffusionPipeline
from tango.common.aliases import PathOrStr
from tango.format import Format

T = TypeVar("T")


@Format.register("diffusers_pipeline")
class DiffusersPipelineFormat(Format[T], Generic[T]):
    VERSION = "001"

    def write(self, artifact: T, dir: PathOrStr):
        assert isinstance(artifact, StableDiffusionPipeline), type(artifact)

        filename = Path(dir) / "pipeline"
        artifact.save_pretrained(filename)

    def read(self, dir: PathOrStr) -> T:
        filename = Path(dir) / "pipeline"
        pipeline = StableDiffusionPipeline.from_pretrained(filename)
        return pipeline
