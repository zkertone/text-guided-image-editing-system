import torch
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
)


DEFAULT_GLOBAL_MODEL_ID = "timbrooks/instruct-pix2pix"
DEFAULT_INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
DEFAULT_CONTROLNET_BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_CONTROLNET_CANNY_MODEL_ID = "lllyasviel/sd-controlnet-canny"


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


def optimize_pipeline_memory(pipe, device: str):
    """Apply lightweight memory optimizations for GPU execution."""
    if device == "cuda":
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
    return pipe


def load_instructpix2pix_pipeline(
    global_model_id: str = DEFAULT_GLOBAL_MODEL_ID,
    inpaint_model_id: str = DEFAULT_INPAINT_MODEL_ID,
    controlnet_base_model_id: str = DEFAULT_CONTROLNET_BASE_MODEL_ID,
    controlnet_canny_model_id: str = DEFAULT_CONTROLNET_CANNY_MODEL_ID,
):
    """
    Load the pre-trained pipelines for the current demo.

    The return structure keeps compatibility with the current project entry
    while allowing the editor to support global editing, local inpainting,
    and structure-preserving editing with Canny ControlNet.
    """
    device = get_device()
    torch_dtype = get_torch_dtype(device)

    global_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        global_model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    global_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        global_pipe.scheduler.config
    )
    global_pipe = global_pipe.to(device)
    global_pipe = optimize_pipeline_memory(global_pipe, device)

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    inpaint_pipe = inpaint_pipe.to(device)
    inpaint_pipe = optimize_pipeline_memory(inpaint_pipe, device)

    controlnet = ControlNetModel.from_pretrained(
        controlnet_canny_model_id,
        torch_dtype=torch_dtype,
    )
    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        controlnet_base_model_id,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    controlnet_pipe = controlnet_pipe.to(device)
    controlnet_pipe = optimize_pipeline_memory(controlnet_pipe, device)

    pipelines = {
        "global_edit": global_pipe,
        "local_inpaint": inpaint_pipe,
        "controlnet_canny": controlnet_pipe,
    }

    return pipelines, device
