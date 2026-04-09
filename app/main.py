import os

from app.editor import ImageEditor
from app.pipeline_loader import load_instructpix2pix_pipeline
from app.ui import create_ui


def get_launch_config():
    """Read Gradio launch settings from environment variables."""
    server_name = os.getenv("GRADIO_SERVER_NAME")
    server_port = os.getenv("GRADIO_SERVER_PORT")
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    launch_kwargs = {"share": share}

    if server_name:
        launch_kwargs["server_name"] = server_name

    if server_port:
        launch_kwargs["server_port"] = int(server_port)

    return launch_kwargs


def main():
    print("Loading InstructPix2Pix pipeline...")
    pipeline, device = load_instructpix2pix_pipeline()
    print(f"Pipeline loaded successfully. Running on device: {device}")

    image_editor = ImageEditor(pipeline=pipeline)
    demo = create_ui(image_editor)
    launch_kwargs = get_launch_config()
    print(f"Gradio launch config: {launch_kwargs}")
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
