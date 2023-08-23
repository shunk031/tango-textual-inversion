import itertools
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tango.integrations.torch.model import Model
from transformers import CLIPTokenizer
from transformers.models.clip import CLIPTextModel


def freeze_params(params: Iterable[nn.Parameter]):
    for param in params:
        param.requires_grad = False


@Model.register("textual_inversion::stable_diffusion")
class StableDiffusionModel(Model):
    def __init__(
        self,
        model_name: str,
        tokenizer: CLIPTokenizer,
        placeholder_token: str,
        initializer_token: str,
        scale_factor: float = 0.18215,
    ) -> None:
        super().__init__()

        self.scale_factor = scale_factor
        self.tokenizer = tokenizer

        self.initializer_token_id = self.get_initializer_token_id(
            initializer_token=initializer_token,
        )
        self.placeholder_token_id = self.get_placeholder_token_id(
            placeholder_token=placeholder_token,
        )

        # load pretrained models
        self.text_encoder = self._load_text_encoder(model_name=model_name)
        self.vae = self._load_vae(model_name=model_name)
        self.unet = self._load_unet(model_name=model_name)

        # setup noise scheduler
        self.noise_scheduler = self._load_noise_scheduler(model_name=model_name)

        # setup the models
        self.resize_token_embeddings()
        self.initialize_placeholder_token()

        self.freeze_params()

    def _load_text_encoder(self, model_name: str) -> CLIPTextModel:
        return CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")  # type: ignore

    def _load_vae(self, model_name: str) -> AutoencoderKL:
        return AutoencoderKL.from_pretrained(model_name, subfolder="vae")  # type: ignore

    def _load_unet(self, model_name: str) -> UNet2DConditionModel:
        return UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")  # type: ignore

    def _load_noise_scheduler(self, model_name: str) -> DDPMScheduler:
        return DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")  # type: ignore

    def get_initializer_token_id(self, initializer_token: str) -> int:
        # Convert the initializer_token, placeholder_token to ids
        token_ids = self.tokenizer.encode(initializer_token, add_special_tokens=False)

        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        return token_ids[0]

    def get_placeholder_token_id(self, placeholder_token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(placeholder_token)  # type: ignore

    def resize_token_embeddings(self) -> None:
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

    def initialize_placeholder_token(self) -> None:
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        token_embeds[self.placeholder_token_id] = token_embeds[
            self.initializer_token_id
        ]

    def freeze_params(self) -> None:
        # Freeze vae and unet
        freeze_params(self.vae.parameters())
        freeze_params(self.unet.parameters())

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            self.text_encoder.text_model.encoder.parameters(),
            self.text_encoder.text_model.final_layer_norm.parameters(),
            self.text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)

    def train(self, mode: bool = True) -> "StableDiffusionModel":
        self.text_encoder.train(mode=mode)

        # Keep vae and unet in eval model as we don't train these
        self.vae.eval()
        self.unet.eval()
        return self

    def forward(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Convert image to lantent space
        latents = self.vae.encode(pixel_values).latent_dist.sample().detach()
        latents = latents * self.scale_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

        return {"loss": loss}
