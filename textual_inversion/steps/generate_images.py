from typing import Optional

from diffusers import StableDiffusionPipeline
from PIL import Image
from tango import Step
from tango.integrations.torch.util import resolve_device, set_seed_all


@Step.register("generate_images")
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
        num_images_per_prompt: int = 8,
        grid_rows: int = 2,
        grid_cols: int = 4,
    ) -> None:

        set_seed_all(seed)
        device = resolve_device()

        pipe = pipe.to(device)

        images = pipe(
            prompt=prompt,
            width=image_size,
            height=image_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        grid = Image.new("RGB", size=(grid_cols * image_size, grid_rows * image_size))

        for i, img in enumerate(images):
            box = (i % grid_cols * image_size, i // grid_cols * image_size)
            grid.paste(img, box=box)

        grid.save(generated_image_path)
