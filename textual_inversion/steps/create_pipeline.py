import logging
import os
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from tango import Step
from tango.format import Format
from tango_ext.integrations.diffusers import DiffusersPipelineFormat
from transformers.models.clip import CLIPTextModel

from textual_inversion.models import StableDiffusionModel

logger = logging.getLogger(__name__)


def save_progress(
    text_encoder: CLIPTextModel,
    placeholder_token: str,
    placeholder_token_id: int,
    output_dir: str,
) -> None:
    if not os.path.exists(output_dir):
        logger.info(f"Make directory to {output_dir}")
        os.makedirs(output_dir)

    learned_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_id]  # type: ignore
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}

    learned_embeds_path = os.path.join(output_dir, "learned_embeds.bin")
    logger.info(f"Saving embeddings to {learned_embeds_path}")
    torch.save(learned_embeds_dict, learned_embeds_path)


@Step.register("textual_inversion::create_pipeline")
class CreatePipeline(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = False

    FORMAT: Format = DiffusersPipelineFormat()

    def run(  # type: ignore
        self,
        model_name: str,
        model: StableDiffusionModel,
        output_dir: str,
        placeholder_token: str,
    ) -> StableDiffusionPipeline:
        assert isinstance(model, StableDiffusionModel)

        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            text_encoder=model.text_encoder,
            vae=model.vae,
            unet=model.unet,
            tokenizer=model.tokenizer,
        )

        # Also save the newly trained embeddings
        save_progress(
            text_encoder=model.text_encoder,
            placeholder_token=placeholder_token,
            placeholder_token_id=model.placeholder_token_id,
            output_dir=output_dir,
        )
        return pipeline
