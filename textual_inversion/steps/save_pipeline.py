import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from diffusers import PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionSafetyChecker as SafetyChecker,
)
from tango import Step
from transformers import CLIPFeatureExtractor as FeatureExtractor
from transformers.models.clip import CLIPTextModel

logger = logging.getLogger(__name__)


def save_progress(
    text_encoder: CLIPTextModel,
    placeholder_token: str,
    placeholder_token_id: int,
    output_dir: str,
) -> None:
    logger.info("Saving embeddings")
    learned_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, os.path.join(output_dir, "learned_embeds.bin"))


@Step.register("save_pipeline")
class SavePipeline(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = False

    def run(  # type: ignore
        self,
        model: nn.Module,
        safety_checker_name: str,
        feature_extractor_name: str,
        output_dir: str,
        placeholder_token: str,
    ) -> StableDiffusionPipeline:

        pipeline = StableDiffusionPipeline(
            text_encoder=model.text_encoder,
            vae=model.vae,
            unet=model.unet,
            tokenizer=model.tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                skip_prk_steps=True,
            ),
            safety_checker=SafetyChecker.from_pretrained(safety_checker_name),
            feature_extractor=FeatureExtractor.from_pretrained(feature_extractor_name),
        )
        pipeline.save_pretrained(output_dir)

        # Also save the newly trained embeddings
        save_progress(
            text_encoder=model.text_encoder,
            placeholder_token=placeholder_token,
            placeholder_token_id=model.placeholder_token_id,
            output_dir=output_dir,
        )
        return pipeline
