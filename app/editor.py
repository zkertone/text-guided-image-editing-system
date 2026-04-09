from datetime import datetime
from pathlib import Path

from PIL import Image


class ImageEditor:
    """A lightweight wrapper around the InstructPix2Pix pipeline."""

    def __init__(self, pipeline, image_size=(512, 512)):
        self.pipeline = pipeline
        self.image_size = image_size
        project_root = Path(__file__).resolve().parents[1]
        self.input_dir = project_root / "data" / "input"
        self.output_dir = project_root / "data" / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_image(self, input_image: Image.Image) -> Image.Image:
        """Convert input image to RGB and resize it to the demo size."""
        return input_image.convert("RGB").resize(self.image_size)

    def _generate_timestamp(self) -> str:
        """Generate a timestamp-based filename suffix to avoid overwriting."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _save_image(self, image: Image.Image, save_dir: Path, prefix: str) -> Path:
        """Save an image to the target directory with a timestamp filename."""
        filename = f"{prefix}_{self._generate_timestamp()}.png"
        save_path = save_dir / filename
        image.save(save_path)
        return save_path

    def _build_summary_text(
        self,
        prompt: str,
        num_inference_steps: int,
        image_guidance_scale: float,
        guidance_scale: float,
    ) -> str:
        """Build a readable multi-line summary for the current experiment."""
        return (
            "Experiment Summary\n"
            f"prompt: {prompt.strip()}\n"
            f"num_inference_steps: {int(num_inference_steps)}\n"
            f"image_guidance_scale: {float(image_guidance_scale)}\n"
            f"guidance_scale: {float(guidance_scale)}\n"
            f"image_size: {self.image_size[0]} x {self.image_size[1]}"
        )

    def edit_image(
        self,
        input_image: Image.Image,
        prompt: str,
        num_inference_steps: int = 20,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
    ) -> dict:
        """
        Run text-guided image editing with the loaded diffusion pipeline.
        """
        if input_image is None:
            raise ValueError("Input image cannot be None.")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        image = self.preprocess_image(input_image)
        input_save_path = self._save_image(image, self.input_dir, "input")

        result = self.pipeline(
            prompt=prompt.strip(),
            image=image,
            num_inference_steps=int(num_inference_steps),
            image_guidance_scale=float(image_guidance_scale),
            guidance_scale=float(guidance_scale),
        ).images[0]

        output_save_path = self._save_image(result, self.output_dir, "output")
        summary_text = self._build_summary_text(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
        )

        return {
            "result_image": result,
            "input_save_path": str(input_save_path),
            "output_save_path": str(output_save_path),
            "summary_text": summary_text,
        }
