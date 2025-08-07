import io
import base64
import json

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def calculate_iou(box1, box2):
    """
    Tính IoU giữa hai bounding box
    box1, box2: [x1, y1, x2, y2]
    """
    # Tính diện tích từng box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Tính tọa độ giao nhau
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Tính diện tích giao nhau
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Tính IoU
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # Tránh chia cho 0
    return iou


def merge_models_results(result1, result2, iou_threshold=0.6):
    """
    Merge kết quả từ hai mô hình YOLOv8 dựa trên IoU theo từng class

    Args:
        result1, result2: Kết quả từ hai mô hình YOLOv8
        iou_threshold: Ngưỡng IoU (mặc định 0.6)

    Returns:
        merged_data: Dict chứa boxes, confs, clss đã được merge
    """
    # Lấy boxes dưới dạng numpy array
    boxes1 = result1.boxes.data[:, :4] if result1.boxes else torch.empty((0, 4))
    boxes2 = result2.boxes.data[:, :4] if result2.boxes else torch.empty((0, 4))

    confs1 = result1.boxes.conf if result1.boxes else torch.empty((0,))
    confs2 = result2.boxes.conf if result2.boxes else torch.empty((0,))

    clss1 = result1.boxes.cls if result1.boxes else torch.empty((0,))
    clss2 = result2.boxes.cls if result2.boxes else torch.empty((0,))

    # Convert to numpy for easier processing
    boxes1_np = boxes1.cpu().numpy() if len(boxes1) > 0 else np.empty((0, 4))
    boxes2_np = boxes2.cpu().numpy() if len(boxes2) > 0 else np.empty((0, 4))
    confs1_np = confs1.cpu().numpy() if len(confs1) > 0 else np.empty((0,))
    confs2_np = confs2.cpu().numpy() if len(confs2) > 0 else np.empty((0,))
    clss1_np = clss1.cpu().numpy() if len(clss1) > 0 else np.empty((0,))
    clss2_np = clss2.cpu().numpy() if len(clss2) > 0 else np.empty((0,))

    # Lấy tất cả class ID có trong cả hai kết quả
    all_classes = set(clss1_np).union(set(clss2_np))

    merged_boxes = []
    merged_confs = []
    merged_clss = []

    # Xử lý từng class riêng biệt
    for cls in all_classes:
        # Lấy boxes thuộc class hiện tại
        cls_mask1 = clss1_np == cls if len(clss1_np) > 0 else np.array([])
        cls_mask2 = clss2_np == cls if len(clss2_np) > 0 else np.array([])

        cls_boxes1 = (
            boxes1_np[cls_mask1]
            if len(boxes1_np) > 0 and len(cls_mask1) > 0
            else np.empty((0, 4))
        )
        cls_boxes2 = (
            boxes2_np[cls_mask2]
            if len(boxes2_np) > 0 and len(cls_mask2) > 0
            else np.empty((0, 4))
        )
        cls_confs1 = (
            confs1_np[cls_mask1]
            if len(confs1_np) > 0 and len(cls_mask1) > 0
            else np.empty((0,))
        )
        cls_confs2 = (
            confs2_np[cls_mask2]
            if len(confs2_np) > 0 and len(cls_mask2) > 0
            else np.empty((0,))
        )

        # Danh sách theo dõi boxes đã được match
        matched_indices = set()

        # Duyệt qua từng box từ model1
        for i, box1 in enumerate(cls_boxes1):
            best_iou = 0
            best_match_idx = -1

            # Tìm box từ model2 có IoU cao nhất
            for idx, box2 in enumerate(cls_boxes2):
                if idx in matched_indices:
                    continue

                iou = calculate_iou(box1, box2)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx

            # Nếu IoU > ngưỡng, lấy box từ model1 và đánh dấu box2 đã được match
            if best_iou > iou_threshold:
                merged_boxes.append(box1)
                merged_confs.append(cls_confs1[i])
                merged_clss.append(cls)
                matched_indices.add(best_match_idx)
            else:
                # Không match được, giữ lại box từ model1
                merged_boxes.append(box1)
                merged_confs.append(cls_confs1[i])
                merged_clss.append(cls)

        # Thêm boxes từ model2 chưa được match
        for idx, box2 in enumerate(cls_boxes2):
            if idx not in matched_indices:
                merged_boxes.append(box2)
                merged_confs.append(cls_confs2[idx])
                merged_clss.append(cls)

    return {
        "boxes": np.array(merged_boxes) if merged_boxes else np.empty((0, 4)),
        "confs": np.array(merged_confs) if merged_confs else np.empty((0,)),
        "clss": np.array(merged_clss) if merged_clss else np.empty((0,)),
    }


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

    # Load both models
    model1 = YOLO("/models/yolov8-taffic/00yy2lq8.pt")
    model2 = YOLO("/models/yolov8-taffic/covctrhe.pt")

    # Move models to the assigned device
    if torch.cuda.is_available():
        model1.to(device)
        model2.to(device)
        context.logger.info("Models moved to GPU")
    else:
        context.logger.info("GPU not available, using CPU")

    context.user_data.model1_handler = model1
    context.user_data.model2_handler = model2
    context.user_data.device = device
    context.logger.info("Init context...100%")


# Inference endpoint
def handler(context, event):
    context.logger.info("Run custom yolov8 model with merge")
    data = event.body
    image_buffer = io.BytesIO(base64.b64decode(data["image"]))
    image = cv2.imdecode(
        np.frombuffer(image_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR
    )

    # Run inference with both models
    results1 = context.user_data.model1_handler(image)
    results2 = context.user_data.model2_handler(image)

    result1 = results1[0]
    result2 = results2[0]

    # Merge results using IoU-based algorithm
    merged_data = merge_models_results(result1, result2, iou_threshold=0.6)

    # Get class names from first model (assuming both models have same classes)
    class_name = result1.names

    detections = []
    threshold = 0.1

    # Process merged results
    for box, conf, cls in zip(
        merged_data["boxes"], merged_data["confs"], merged_data["clss"]
    ):
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

    context.logger.info(f"Merged detection count: {len(detections)}")
    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
