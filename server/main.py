import asyncio
import base64
import logging
from dataclasses import dataclass
from io import BytesIO

import imagehash
import torch
import uvicorn
from diffusers import AutoencoderTiny, FluxImg2ImgPipeline, FluxTransformer2DModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchao.quantization import float8_dynamic_activation_float8_weight, quantize_
from torchao.quantization.quant_api import PerRow
from transformers import T5EncoderModel

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger("uvicorn")


class PredictRequest(BaseModel):
    image: str | None
    text: str


@dataclass
class FastPipelineCache:
    prompt_embeds: torch.Tensor | None = None
    pooled_prompt_embeds: torch.Tensor | None = None
    last_prompt: str = ""
    last_input_image: Image.Image | None = None
    last_output_image: Image.Image | None = None

    _last_input_image_hash: imagehash.ImageHash = imagehash.phash(
        Image.new("RGB", (1, 1))
    )

    def no_difference(self, prompt: str, image: Image.Image | None) -> bool:
        if image is None:
            return self.prompt_is_different(prompt)
        else:
            return not self.prompt_is_different(prompt) and not self.image_is_different(
                image
            )

    def prompt_is_different(self, prompt: str) -> bool:
        return self.prompt_embeds is None or self.last_prompt != prompt

    def image_is_different(self, image: Image.Image) -> bool:
        return self._last_input_image_hash - imagehash.phash(image) > 5

    def update_prompt_embeds(
        self, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor
    ):
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds

    def update_last_input_image(self, image: Image.Image):
        self.last_input_image = image
        self._last_input_image_hash = imagehash.phash(image)

    def update_last_output_image(self, image: Image.Image):
        self.last_output_image = image

    def update_last_prompt(self, prompt: str):
        self.last_prompt = prompt


class FastPipelineWrapper:
    def __init__(
        self,
        repo_id: str = "black-forest-labs/FLUX.1-schnell",
        vae_id: str = "madebyollin/taef1",
        device: str = "cuda",
        optimize: bool = True,
        offload: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.device = device
        self.offload = offload
        self.pipe = self._load_pipeline(repo_id, vae_id, optimize=optimize)

        self.cache = FastPipelineCache()

        self._warmup()

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        image: Image.Image | None = None,
        num_inference_steps: int = 4,
        strength: float = 0.5,
        width: int = 1024,
        height: int = 1024,
    ) -> Image.Image:
        if self.cache.no_difference(prompt, image):
            return self.cache.last_output_image

        if image is not None and self.cache.image_is_different(image):
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

        image = self.pipe(
            image=image,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=int(num_inference_steps * strength),
            strength=strength,
            width=width,
            height=height,
            guidance_scale=0.0,
        ).images[0]

        self.cache.update_last_output_image(image)

        return image

    @torch.inference_mode()
    def _precomute_embeddings(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device="cpu" if self.offload else self.device,
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
                device="cuda",
            )
            transformer.to(memory_format=torch.channels_last)
            transformer = torch.compile(
                transformer, mode="max-autotune", fullgraph=True
            )

            quantize_(
                vae,
                float8_dynamic_activation_float8_weight(granularity=PerRow()),
                device="cuda",
            )
            vae.to(memory_format=torch.channels_last)
            vae.decoder = torch.compile(
                vae.decoder, mode="max-autotune", fullgraph=True
            )

        pipe: FluxImg2ImgPipeline = FluxImg2ImgPipeline.from_pretrained(
            model_id,
            vae=None,
            transformer=None,
            text_encoder_2=None,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        text_encoder_2: T5EncoderModel = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        ).to(self.device)

        pipe.vae = vae
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2

        pipe.set_progress_bar_config(disable=True)

        return pipe

    @torch.inference_mode()
    def _warmup(self):
        for _ in range(10):
            self.__call__(prompt="A girl", width=512, height=512)


class FasterFluxAPI:
    def __init__(self):
        self.pipeline = FastPipelineWrapper(optimize=True, offload=False)
        self.app = FastAPI()

        self.app.add_api_route("/predict", self.predict, methods=["POST"])
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._predict_lock = asyncio.Lock()

    async def predict(self, request: PredictRequest):
        async with self._predict_lock:
            prompt = request.text

            if request.image:
                _, data = request.image.split(",", 1)
                image_data = base64.b64decode(data)
                input_image = Image.open(BytesIO(image_data))
            else:
                input_image = None

            output_image = self.pipeline(
                prompt=prompt,
                image=input_image,
                num_inference_steps=4,
                strength=0.5,
                width=1024,
                height=1024,
            )

            buffered = BytesIO()
            output_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {"imageBase64": image_base64}


if __name__ == "__main__":
    api = FasterFluxAPI()
    uvicorn.run(api.app, host="0.0.0.0", port=9090)
