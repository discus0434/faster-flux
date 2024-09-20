import argparse
import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from faster_flux import FastPipelineWrapper, Mode

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
ROOT_DIR = Path(__file__).parent

logger = logging.getLogger("uvicorn")


class Txt2ImgRequest(BaseModel):
    text: str


class Img2ImgRequest(BaseModel):
    image: str | None
    text: str


class Response(BaseModel):
    imageBase64: str


class FasterFluxAPI:
    def __init__(
        self, resolution: int = 512, optimize: bool = True, num_inference_steps: int = 1
    ) -> None:
        self.pipeline = FastPipelineWrapper(resolution=resolution, optimize=optimize)
        self.app = FastAPI()

        self.app.add_api_route(
            "/api/txt2img", self.txt2img, methods=["POST"], response_model=Response
        )
        self.app.add_api_route(
            "/api/img2img", self.img2img, methods=["POST"], response_model=Response
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.mount(
            "/txt2img",
            StaticFiles(directory=ROOT_DIR / "view" / "txt2img" / "build", html=True),
            name="txt2img",
        )
        self.app.mount(
            "/img2img",
            StaticFiles(directory=ROOT_DIR / "view" / "img2img" / "build", html=True),
            name="img2img",
        )

        self._txt2img_lock = asyncio.Lock()
        self._img2img_lock = asyncio.Lock()

        self.num_inference_steps = num_inference_steps

    async def txt2img(self, request: Txt2ImgRequest) -> Response:
        async with self._txt2img_lock:
            prompt = request.text
            output_image = self.pipeline(
                mode=Mode.TXT2IMG,
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
            )
            buffered = BytesIO()
            output_image.save(buffered, format="WEBP")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return Response(imageBase64=image_base64)

    async def img2img(self, request: Img2ImgRequest) -> Response:
        async with self._img2img_lock:
            prompt = request.text

            if request.image:
                _, data = request.image.split(",", 1)
                image_data = base64.b64decode(data)
                input_image = Image.open(BytesIO(image_data))
                if input_image.mode != "RGB":
                    input_image = input_image.convert("RGB")
            else:
                input_image = None

            output_image = self.pipeline(
                mode=Mode.IMG2IMG,
                prompt=prompt,
                image=input_image,
                num_inference_steps=self.num_inference_steps,
            )

            buffered = BytesIO()
            output_image.save(buffered, format="WEBP")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return Response(imageBase64=image_base64)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--resolution", type=int, default=512)
    argparser.add_argument("--optimize", type=int, default=1)
    argparser.add_argument("--num_inference_steps", type=int, default=4)
    argparser.add_argument("--server_host", type=str, default="0.0.0.0")
    argparser.add_argument("--server_port", type=int, default=9090)

    args = argparser.parse_args()

    api = FasterFluxAPI(
        resolution=args.resolution,
        optimize=bool(args.optimize),
        num_inference_steps=args.num_inference_steps,
    )
    uvicorn.run(api.app, host=args.server_host, port=args.server_port)
