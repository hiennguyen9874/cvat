import io
import base64
import json

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Initialize your model
def init_context(context):
    context.logger.info("Init context...  0%")

    # Get worker ID and number of available GPUs
    worker_id_raw = getattr(context, "worker_id", 0)  # Get raw worker_id
    # Convert worker_id to integer if it's a string
    if isinstance(worker_id_raw, str):
        worker_id = int(worker_id_raw)
    else:
        worker_id = worker_id_raw

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    context.logger.info(f"Worker ID: {worker_id}")
    context.logger.info(f"Number of available GPUs: {num_gpus}")

    # Calculate device ID using modulo operation
    if num_gpus > 0:
        device_id = worker_id % num_gpus
        device = f"cuda:{device_id}"
        context.logger.info(f"Worker {worker_id} assigned to GPU device: {device}")
    else:
        device = "cpu"
        context.logger.info(f"Worker {worker_id} using CPU (no GPUs available)")

    # Load model
    model = YOLO("/models/yolov8-traffic/8hlip5bh.pt")

    # Move model to the assigned device
    if torch.cuda.is_available():
        model.to(device)
        context.logger.info("Model moved to GPU")
    else:
        context.logger.info("GPU not available, using CPU")

    context.user_data.model_handler = model
    context.user_data.device = device
    context.logger.info("Init context...100%")


# Inference endpoint
def handler(context, event):
    context.logger.info("Run custom yolov8 model")
    data = event.body
    image_buffer = io.BytesIO(base64.b64decode(data["image"]))
    image = cv2.imdecode(
        np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR
    )

    # Run inference (YOLO handles device placement automatically when model is on GPU)
    results = context.user_data.model_handler(image)
    result = results[0]

    boxes = result.boxes.data[:, :4]
    confs = result.boxes.conf
    clss = result.boxes.cls
    class_name = result.names

    detections = []
    threshold = 0.1
    for box, conf, cls in zip(boxes, confs, clss):
        label = class_name[int(cls)]
        if conf >= threshold:
            # must be in this format
            detections.append(
                {
                    "confidence": float(conf),
                    "label": label,
                    "points": box.tolist(),
                    "type": "rectangle",
                }
            )

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
