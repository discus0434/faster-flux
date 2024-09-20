import logging
import time
from enum import Enum
from typing import Literal

import torch
from diffusers import AutoencoderTiny, FluxImg2ImgPipeline, FluxTransformer2DModel
from PIL import Image
from torchao.quantization import float8_dynamic_activation_float8_weight, quantize_
from torchao.quantization.quant_api import PerRow
from transformers import T5EncoderModel

from ._cache import Cache

logger = logging.getLogger("uvicorn")


class Mode(Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"


class FastPipelineWrapper:
    guidance_scale: float = 3.5
    seed = 42

    def __init__(
        self,
        resolution: int = 1024,
        repo_id: str = "black-forest-labs/FLUX.1-schnell",
        vae_id: str = "madebyollin/taef1",
        device: str = "cuda",
        optimize: bool = True,
    ) -> None:
        self.resolution = resolution
        self.repo_id = repo_id
        self.device = device
        self.pipe = self._load_pipeline(repo_id, vae_id, optimize=optimize)

        self.cache = Cache()
        self.generator = torch.Generator().manual_seed(self.seed)
        self._latents = self._generate_latents()
        self._warmup()

    @torch.inference_mode()
    def __call__(
        self,
        mode: Literal[Mode.TXT2IMG, Mode.IMG2IMG],
        prompt: str,
        image: Image.Image | None = None,
        num_inference_steps: int = 4,
        img2img_strength: float = 0.85,
    ) -> Image.Image:
        if self.cache.no_difference(prompt, image):
            return self.cache.last_output_image

        if self.cache.image_is_different(image):
            self.cache.update_last_input_image(image)
        else:
            image = self.cache.last_input_image

        if self.cache.prompt_is_different(prompt):
            prompt_embeds, pooled_prompt_embeds = self._precomute_embeddings(prompt)
            self.cache.update_prompt_embeds(prompt_embeds, pooled_prompt_embeds)
            self.cache.update_last_prompt(prompt)
        else:
            prompt_embeds = self.cache.prompt_embeds
            pooled_prompt_embeds = self.cache.pooled_prompt_embeds

        if mode == Mode.TXT2IMG:
            strength = 1.0
            image = self._fake_image
            latents = self._latents
        else:
            strength = img2img_strength
            latents = None

        output_image = self.pipe(
            image=image,
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            strength=strength,
            width=self.resolution,
            height=self.resolution,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
        ).images[0]

        self.cache.update_last_output_image(output_image)

        return output_image

    @torch.inference_mode()
    def benchmark(self, mode: Literal[Mode.TXT2IMG, Mode.IMG2IMG] = Mode.TXT2IMG):
        if mode == Mode.TXT2IMG:
            print("Start txt2img benchmark")
            elapsed_times = []
            for _ in range(50):
                start = time.time()
                self.pipe(
                    image=self._fake_image,
                    latents=self._latents,
                    prompt_embeds=self.cache.prompt_embeds,
                    pooled_prompt_embeds=self.cache.pooled_prompt_embeds,
                    num_inference_steps=4,
                    strength=1.0,
                    width=self.resolution,
                    height=self.resolution,
                    guidance_scale=self.guidance_scale,
                    generator=self.generator,
                ).images[0]
                end = time.time()
                print(f"Time: {end - start:.2f}s")
                elapsed_times.append(end - start)

            print(f"Average time: {sum(elapsed_times) / len(elapsed_times):.2f}s")
            print(f"Max time: {max(elapsed_times):.2f}s")
            print(f"Min time: {min(elapsed_times):.2f}s")
        elif mode == Mode.IMG2IMG:
            print("Start img2img benchmark")
            elapsed_times = []
            for _ in range(50):
                start = time.time()
                self.pipe(
                    image=self.cache.last_input_image,
                    latents=None,
                    prompt_embeds=self.cache.prompt_embeds,
                    pooled_prompt_embeds=self.cache.pooled_prompt_embeds,
                    num_inference_steps=4,
                    strength=0.85,
                    width=self.resolution,
                    height=self.resolution,
                    guidance_scale=self.guidance_scale,
                    generator=self.generator,
                ).images[0]
                end = time.time()
                print(f"Time: {end - start:.2f}s")
                elapsed_times.append(end - start)

            print(f"Average time: {sum(elapsed_times) / len(elapsed_times):.2f}s")
            print(f"Max time: {max(elapsed_times):.2f}s")
            print(f"Min time: {min(elapsed_times):.2f}s")

    @torch.inference_mode()
    def _precomute_embeddings(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
        return prompt_embeds.to(
            self.device, dtype=torch.bfloat16
        ), pooled_prompt_embeds.to(self.device, dtype=torch.bfloat16)

    def _load_pipeline(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        vae_id: str = "madebyollin/taef1",
        optimize: bool = True,
    ) -> FluxImg2ImgPipeline:
        pipe: FluxImg2ImgPipeline
        pipe = FluxImg2ImgPipeline.from_pretrained(
            model_id,
            vae=None,
            transformer=None,
            text_encoder_2=None,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        text_encoder_2: T5EncoderModel = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        ).to(self.device)
        vae: AutoencoderTiny = AutoencoderTiny.from_pretrained(
            vae_id, torch_dtype=torch.bfloat16
        )
        transformer: FluxTransformer2DModel = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        transformer.fuse_qkv_projections()
        torch.cuda.empty_cache()

        if optimize:
            quantize_(
                transformer,
                float8_dynamic_activation_float8_weight(granularity=PerRow()),
                device=self.device,
            )
            transformer.to(memory_format=torch.channels_last)
            transformer = torch.compile(
                transformer, mode="max-autotune", fullgraph=True
            )

            quantize_(
                vae,
                float8_dynamic_activation_float8_weight(granularity=PerRow()),
                device=self.device,
            )
            vae.to(memory_format=torch.channels_last)
            vae.decoder = torch.compile(
                vae.decoder, mode="max-autotune", fullgraph=True
            )
        else:
            transformer = transformer.to(self.device)
            vae = vae.to(self.device)

        pipe.vae = vae
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2

        pipe.set_progress_bar_config(disable=True)

        return pipe

    @torch.inference_mode()
    def _warmup(self):
        for _ in range(10):
            self.__call__(mode=Mode.TXT2IMG, prompt="")

    def _generate_latents(self) -> torch.Tensor:
        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        height = 2 * (self.resolution // self.pipe.vae_scale_factor)
        width = 2 * (self.resolution // self.pipe.vae_scale_factor)
        latents = torch.randn(
            (1, num_channels_latents, height, width),
            generator=self.generator,
        )
        latents = latents.view(1, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            1, (height // 2) * (width // 2), num_channels_latents * 4
        )

        return latents.to(self.device, dtype=torch.bfloat16)

    @property
    def _fake_image(self) -> torch.Tensor:
        return (
            torch.distributions.Normal(0, 1)
            .cdf(torch.randn((1, 3, self.resolution, self.resolution)))
            .to(self.device, dtype=torch.float16)
        )
