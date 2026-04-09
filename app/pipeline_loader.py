import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)


DEFAULT_MODEL_ID = "timbrooks/instruct-pix2pix"


def get_device() -> str:
    """Automatically select the available inference device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    """Use float16 on CUDA and float32 on CPU for better compatibility."""
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_instructpix2pix_pipeline(model_id: str = DEFAULT_MODEL_ID):
    """
    Load the pre-trained InstructPix2Pix pipeline.

    This function keeps the logic simple for the V1 demo and avoids
    introducing additional configuration complexity.
    """
    device = get_device()
    torch_dtype = get_torch_dtype(device)

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )
    pipe = pipe.to(device)

    if device == "cuda":
        # Keep V1 simple while adding a safer memory optimization for GPU use.
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

    return pipe, device
