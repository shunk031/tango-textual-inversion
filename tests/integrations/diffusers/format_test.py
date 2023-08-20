import os

import torch
from diffusers import StableDiffusionPipeline

from textual_inversion.common.testing import TextualInversionTestCase
from textual_inversion.integrations.diffusers import DiffusersPipelineFormat


class TestDiffusersPipelineFormat(TextualInversionTestCase):
    def test_read_write(self):
        model_id = "runwayml/stable-diffusion-v1-5"
        artifact = StableDiffusionPipeline.from_pretrained(
            model_id, revision="fp16", torch_dtype=torch.float16
        )

        pipeline_format = DiffusersPipelineFormat[StableDiffusionPipeline]()

        pipeline_format.write(artifact, self.TEST_DIR)  # type: ignore

        assert os.path.exists(self.TEST_DIR / "pipeline")
        assert pipeline_format.read(self.TEST_DIR)
