from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from tango import Step
from tango.integrations.torch.util import resolve_device, set_seed_all


@Step.register("textual_inversion::generate_images")
class GenerateImages(Step):
    DETERMINISTIC: bool = False
    CACHEABLE: Optional[bool] = False

    def run(  # type: ignore
        self,
        pipe: StableDiffusionPipeline,
        prompt: str,
        seed: int,
        generated_image_path: str,
        image_size: int = 512,
        grid_rows: int = 1,
        grid_cols: int = 4,
    ) -> None:
        num_images_per_prompt = grid_rows * grid_cols

        set_seed_all(seed)
        device = resolve_device()

        pipe = pipe.to(device)
        generator = torch.Generator().manual_seed(seed)

        images = pipe(
            prompt=prompt,
            width=image_size,
            height=image_size,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        grid = Image.new("RGB", size=(grid_cols * image_size, grid_rows * image_size))

        for i, img in enumerate(images):
            box = (i % grid_cols * image_size, i // grid_cols * image_size)
            grid.paste(img, box=box)

        grid.save(generated_image_path)
