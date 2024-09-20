from dataclasses import dataclass, field
from PIL import Image
import imagehash
import torch

@dataclass
class Cache:
    prompt_embeds: torch.Tensor | None = None
    pooled_prompt_embeds: torch.Tensor | None = None
    last_prompt: str = ""
    last_input_image: Image.Image = field(
        default_factory=lambda: Image.new("RGB", (1024, 1024), (255, 255, 255))
    )
    last_output_image: Image.Image = field(
        default_factory=lambda: Image.new("RGB", (1024, 1024), (0, 0, 0))
    )

    _last_input_image_hash: imagehash.ImageHash = field(
        default_factory=lambda: imagehash.average_hash(
            Image.new("RGB", (1024, 1024), (0, 0, 0))
        )
    )

    def no_difference(self, prompt: str, image: Image.Image | None = None) -> bool:
        if image is None:
            return not self.prompt_is_different(prompt)
        else:
            return not self.prompt_is_different(prompt) and not self.image_is_different(
                image
            )

    def prompt_is_different(self, prompt: str) -> bool:
        return self.prompt_embeds is None or self.last_prompt != prompt

    def image_is_different(self, image: Image.Image | None) -> bool:
        return (
            image is not None
            and self._last_input_image_hash - imagehash.average_hash(image) > 2
        )

    def update_prompt_embeds(
        self, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor
    ):
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds

    def update_last_input_image(self, image: Image.Image):
        self.last_input_image = image
        self._last_input_image_hash = imagehash.average_hash(image)

    def update_last_output_image(self, image: Image.Image):
        self.last_output_image = image

    def update_last_prompt(self, prompt: str):
        self.last_prompt = prompt
