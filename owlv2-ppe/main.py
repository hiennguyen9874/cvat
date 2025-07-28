import io
import base64
import json

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


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
        device = torch.device(f"cuda:{device_id}")
        context.logger.info(
            f"Worker {worker_id} assigned to GPU device: cuda:{device_id}"
        )
    else:
        device = torch.device("cpu")
        context.logger.info(f"Worker {worker_id} using CPU (no GPUs available)")

    # Load model and processor
    processor = Owlv2Processor.from_pretrained(
        "/models/owlv2-base-patch16-ensemble", trust_remote_code=True
    )
    model = Owlv2ForObjectDetection.from_pretrained(
        "/models/owlv2-base-patch16-ensemble", trust_remote_code=True
    )

    # Move model to the assigned device
    model = model.to(device)

    context.user_data.processor = processor
    context.user_data.model_handler = model
    context.user_data.device = device
    context.logger.info("Init context...100%")


# Inference endpoint
def handler(context, event):
    context.logger.info("Run custom OWLv2 model")
    data = event.body
    image_buffer = io.BytesIO(base64.b64decode(data["image"]))

    # Convert CV2 image to PIL Image for OWLv2
    image_cv2 = cv2.imdecode(
        np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR
    )
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    # Get text prompts from request data, with default prompts if not provided
    texts = data.get(
        "texts",
        [
            [
                "safety helmet",
                "safety vest",
                "safety pants",
            ]
        ],
    )

    # Process inputs
    inputs = context.user_data.processor(text=texts, images=image, return_tensors="pt")

    # Move inputs to the same device as the model
    device = context.user_data.device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    # Run inference
    with torch.no_grad():
        outputs = context.user_data.model_handler(**inputs)

    # Target image sizes (height, width) to rescale box predictions
    target_sizes = torch.Tensor([image.size[::-1]])

    # Convert outputs to Pascal VOC Format (xmin, ymin, xmax, ymax)
    threshold = data.get("threshold", 0.1)
    results = context.user_data.processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=threshold
    )

    print("results", results)

    # Process results for the first image
    detections = []
    if len(results) > 0:
        result = results[0]
        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
        text_queries = texts[0]  # Get the text queries for the first (and only) image

        # Separate detections by class
        vest_boxes = []
        pants_boxes = []
        other_detections = []

        for box, score, label in zip(boxes, scores, labels):
            label_text = text_queries[label.item()]

            if label_text == "safety vest":
                vest_boxes.append((box, score))
            elif label_text == "safety pants":
                pants_boxes.append((box, score))
            else:
                # Keep other detections as is
                other_detections.append(
                    {
                        "confidence": float(score.item()),
                        "label": label_text,
                        "points": [round(coord, 2) for coord in box.tolist()],
                        "type": "rectangle",
                    }
                )

        # Merge vest and pants boxes if both exist
        if vest_boxes and pants_boxes:
            # Combine all vest and pants boxes
            all_boxes = [box for box, _ in vest_boxes + pants_boxes]
            all_scores = [score for _, score in vest_boxes + pants_boxes]

            # Calculate merged bounding box (min x, min y, max x, max y)
            all_boxes_tensor = torch.stack(all_boxes)
            min_coords = torch.min(all_boxes_tensor, dim=0)[0]  # min x, min y
            max_coords = torch.max(all_boxes_tensor, dim=0)[0]  # max x, max y

            # Create merged box [xmin, ymin, xmax, ymax]
            merged_box = [
                min_coords[0].item(),
                min_coords[1].item(),
                max_coords[2].item(),
                max_coords[3].item(),
            ]

            # Use maximum confidence from all merged boxes
            merged_confidence = max(all_scores).item()

            detections.append(
                {
                    "confidence": float(merged_confidence),
                    "label": "safety vest and safety pants",
                    "points": [round(coord, 2) for coord in merged_box],
                    "type": "rectangle",
                }
            )
        elif vest_boxes:
            # Only vest boxes found - keep them as separate vest detections
            for box, score in vest_boxes:
                detections.append(
                    {
                        "confidence": float(score.item()),
                        "label": "safety vest",
                        "points": [round(coord, 2) for coord in box.tolist()],
                        "type": "rectangle",
                    }
                )
        elif pants_boxes:
            # Only pants boxes found - keep them as separate pants detections
            for box, score in pants_boxes:
                detections.append(
                    {
                        "confidence": float(score.item()),
                        "label": "safety pants",
                        "points": [round(coord, 2) for coord in box.tolist()],
                        "type": "rectangle",
                    }
                )

        # Add other detections (like safety helmet)
        detections.extend(other_detections)

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
