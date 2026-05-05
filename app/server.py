from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from app.database import init_database
from app.editor import ImageEditor
from app.history_manager import HistoryManager
from app.pipeline_loader import load_instructpix2pix_pipeline


app = FastAPI(title="Text-Guided Image Editing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_editor: Optional[ImageEditor] = None
history_manager = HistoryManager()


@app.on_event("startup")
def startup() -> None:
    """Initialize SQLite and load diffusion pipelines once."""
    global image_editor
    init_database()
    pipelines, _ = load_instructpix2pix_pipeline()
    image_editor = ImageEditor(pipeline=pipelines)


def get_editor() -> ImageEditor:
    if image_editor is None:
        raise HTTPException(status_code=503, detail="Model pipelines are still loading.")
    return image_editor


async def read_upload_image(upload_file: UploadFile) -> Image.Image:
    image_bytes = await upload_file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.load()
        return image
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {error}") from error


def image_url(image_id: Optional[int]) -> Optional[str]:
    if image_id is None:
        return None
    return f"/api/images/{image_id}"


def record_image_urls(record: dict) -> dict:
    return {
        "input_image_url": image_url(record.get("input_image_id")),
        "mask_image_url": image_url(record.get("mask_image_id")),
        "control_image_url": image_url(record.get("control_image_id")),
        "output_image_url": image_url(record.get("output_image_id")),
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "message": "server is running",
    }


@app.post("/api/edit")
async def edit_image(
    mode: str = Form(...),
    image: UploadFile = File(...),
    prompt: str = Form(...),
    num_inference_steps: int = Form(20),
    image_guidance_scale: float = Form(1.5),
    guidance_scale: float = Form(7.5),
    mask_image: Optional[UploadFile] = File(None),
) -> dict:
    editor = get_editor()
    input_image = await read_upload_image(image)
    mask = None
    mask_source = "not_applicable"

    if mode not in {"global_edit", "local_inpaint", "controlnet_canny"}:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    if mode == "local_inpaint":
        if mask_image is None:
            raise HTTPException(
                status_code=400,
                detail="mask_image is required when mode is local_inpaint.",
            )
        mask = await read_upload_image(mask_image)
        mask_source = "uploaded_mask"

    try:
        result = editor.edit_image(
            input_image=input_image,
            prompt=prompt,
            mode=mode,
            mask_source=mask_source,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "mode": result["mode"],
        "edit_record_id": result["edit_record_id"],
        "input_image_id": result["input_image_id"],
        "mask_image_id": result["mask_image_id"],
        "control_image_id": result["control_image_id"],
        "output_image_id": result["output_image_id"],
        "output_image_url": image_url(result["output_image_id"]),
        "control_image_url": image_url(result["control_image_id"]),
        "summary_text": result["summary_text"],
    }


@app.get("/api/images/{image_id}")
def get_image(image_id: int) -> Response:
    try:
        image_data = history_manager.load_image_bytes(image_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return Response(content=image_data, media_type="image/png")


@app.get("/api/history")
def get_history() -> list[dict]:
    return history_manager.get_recent_record_dicts(limit=10)


@app.get("/api/history/{record_id}")
def get_history_detail(record_id: int) -> dict:
    record = history_manager.get_record_detail(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Edit record not found.")
    return record


@app.get("/api/history/{record_id}/images")
def get_history_images(record_id: int) -> dict:
    record = history_manager.get_record_detail(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Edit record not found.")
    return record_image_urls(record)


@app.get("/api/input-images")
def get_input_images() -> list[dict]:
    return history_manager.get_recent_input_image_dicts(limit=10)


@app.delete("/api/history/{record_id}")
def delete_history_record(record_id: int) -> dict:
    deleted = history_manager.soft_delete_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Edit record not found or already deleted.")
    return {
        "status": "ok",
        "message": "edit record deleted",
        "record_id": record_id,
    }


@app.delete("/api/images/{image_id}")
def delete_image(image_id: int) -> dict:
    deleted = history_manager.soft_delete_image(image_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Image not found or already deleted.")
    return {
        "status": "ok",
        "message": "image deleted",
        "image_id": image_id,
    }
