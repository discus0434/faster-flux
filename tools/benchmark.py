import argparse

from faster_flux import FastPipelineWrapper, Mode

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--resolution", type=int, default=512)
    argparser.add_argument("--optimize", type=bool, default=True)
    argparser.add_argument(
        "--mode", type=str, choices=["txt2img", "img2img"], default="txt2img"
    )
    args = argparser.parse_args()

    mode = Mode.TXT2IMG if args.mode == "txt2img" else Mode.IMG2IMG

    pipeline = FastPipelineWrapper(resolution=args.resolution, optimize=args.optimize)
    pipeline.benchmark(mode)
