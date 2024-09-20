import argparse

from faster_flux import FastPipelineWrapper, Mode

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--resolution", type=int, default=512)
    argparser.add_argument("--optimize", type=int, default=1)
    argparser.add_argument("--num_inference_steps", type=int, default=1)
    argparser.add_argument(
        "--mode", type=str, choices=["txt2img", "img2img"], default="txt2img"
    )
    args = argparser.parse_args()

    mode = Mode.TXT2IMG if args.mode == "txt2img" else Mode.IMG2IMG

    pipeline = FastPipelineWrapper(
        resolution=args.resolution, optimize=bool(args.optimize)
    )
    pipeline.benchmark(mode, num_inference_steps=args.num_inference_steps)
