from app.editor import ImageEditor
from app.pipeline_loader import load_instructpix2pix_pipeline
from app.ui import create_ui


def main():
    print("Loading InstructPix2Pix pipeline...")
    pipeline, device = load_instructpix2pix_pipeline()
    print(f"Pipeline loaded successfully. Running on device: {device}")

    image_editor = ImageEditor(pipeline=pipeline)
    demo = create_ui(image_editor)
    demo.launch()


if __name__ == "__main__":
    main()
