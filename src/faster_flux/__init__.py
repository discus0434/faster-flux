import torch

from .pipeline_wrapper import FastPipelineWrapper, Mode

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True

__all__ = ["FastPipelineWrapper", "Mode"]
