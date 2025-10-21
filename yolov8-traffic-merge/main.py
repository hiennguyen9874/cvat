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


def nms_on_all_classes(boxes, confs, clss, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression on all classes

    Args:
        boxes: numpy array of bounding boxes [N, 4] in format [x1, y1, x2, y2]
        confs: numpy array of confidence scores [N]
        clss: numpy array of class IDs [N]
        iou_threshold: IoU threshold for NMS (default 0.5)

    Returns:
        Dict containing filtered boxes, confs, and clss after NMS
    """
    if len(boxes) == 0:
        return {
            "boxes": np.empty((0, 4)),
            "confs": np.empty((0,)),
            "clss": np.empty((0,)),
        }

    # Sort by confidence in descending order
    sorted_indices = np.argsort(confs)[::-1]

    keep_indices = []

    while len(sorted_indices) > 0:
        # Take the box with highest confidence
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)

        if len(sorted_indices) == 1:
            break

        # Calculate IoU with remaining boxes
        current_box = boxes[current_idx]
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]

        # Calculate IoU between current box and all remaining boxes
        ious = np.array([
            calculate_iou(current_box, box) for box in remaining_boxes
        ])

        # Keep only boxes with IoU less than threshold
        keep_mask = ious < iou_threshold
        sorted_indices = remaining_indices[keep_mask]

    # Return filtered results
    keep_indices = np.array(keep_indices)
    return {
        "boxes": boxes[keep_indices],
        "confs": confs[keep_indices],
        "clss": clss[keep_indices],
    }


def merge_models_results(
    result1, result2, iou_threshold=0.6, merge_across_classes=False,
    model2_class_filter=None, class_mapping=None
):
    """
    Merge kết quả từ hai mô hình YOLOv8 dựa trên IoU theo từng class hoặc tất cả class

    Args:
        result1, result2: Kết quả từ hai mô hình YOLOv8
        iou_threshold: Ngưỡng IoU (mặc định 0.6)
        merge_across_classes: Nếu True, merge boxes qua tất cả class.
                             Nếu False, chỉ merge trong cùng class (mặc định False)
        model2_class_filter: List các tên class từ model2 muốn giữ lại.
                            Nếu None, giữ tất cả class (mặc định None)
        class_mapping: Dict mapping class names from model2 to model1.
                      Format: {"model2_class": "model1_class"}
                      Nếu None, sử dụng exact name matching (mặc định None)

    Returns:
        merged_data: Dict chứa boxes, confs, clss đã được merge
        (clss sử dụng class ID từ model1)
    """
    # Get class names from both models
    class_names1 = result1.names
    class_names2 = result2.names
    
    # Create mapping from model2 class names to model1 class IDs
    # If a class from model2 doesn't exist in model1, it will be ignored
    class_mapping_2_to_1 = {}
    for cls2_id, cls2_name in class_names2.items():
        # Apply class filter if specified
        if (model2_class_filter is not None and
                cls2_name not in model2_class_filter):
            continue
        
        # Use custom mapping if provided, otherwise use exact name matching
        if class_mapping is not None and cls2_name in class_mapping:
            # Use custom mapping
            target_cls1_name = class_mapping[cls2_name]
        else:
            # Use exact name matching (original behavior)
            target_cls1_name = cls2_name
        
        # Find the target class ID in model1
        for cls1_id, cls1_name in class_names1.items():
            if cls1_name == target_cls1_name:
                class_mapping_2_to_1[cls2_id] = cls1_id
                break
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

    merged_boxes = []
    merged_confs = []
    merged_clss = []

    if merge_across_classes:
        # Merge boxes across all classes (ignore class boundaries)
        matched_indices = set()

        # Duyệt qua từng box từ model1
        for i, box1 in enumerate(boxes1_np):
            best_iou = 0
            best_match_idx = -1

            # Tìm box từ model2 có IoU cao nhất (bất kể class)
            for idx, box2 in enumerate(boxes2_np):
                if idx in matched_indices:
                    continue

                iou = calculate_iou(box1, box2)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx

            # Nếu IoU > ngưỡng, lấy box từ model1 và đánh dấu box2 đã được match
            if best_iou > iou_threshold:
                merged_boxes.append(box1)
                merged_confs.append(confs1_np[i])
                merged_clss.append(clss1_np[i])
                matched_indices.add(best_match_idx)
            else:
                # Không match được, giữ lại box từ model1
                merged_boxes.append(box1)
                merged_confs.append(confs1_np[i])
                merged_clss.append(clss1_np[i])

        # Thêm boxes từ model2 chưa được match, mapping class to model1
        for idx, box2 in enumerate(boxes2_np):
            if idx not in matched_indices:
                cls2_id = int(clss2_np[idx])
                # Only add if the class exists in model1
                if cls2_id in class_mapping_2_to_1:
                    merged_boxes.append(box2)
                    merged_confs.append(confs2_np[idx])
                    merged_clss.append(class_mapping_2_to_1[cls2_id])

    else:
        # Xử lý từng class riêng biệt - chỉ merge classes có cùng tên
        # Get all classes from model1
        all_classes = set(clss1_np) if len(clss1_np) > 0 else set()

        for cls in all_classes:
            # Lấy boxes thuộc class hiện tại từ model1
            cls_mask1 = clss1_np == cls if len(clss1_np) > 0 else np.array([])
            
            cls_boxes1 = (
                boxes1_np[cls_mask1]
                if len(boxes1_np) > 0 and len(cls_mask1) > 0
                else np.empty((0, 4))
            )
            cls_confs1 = (
                confs1_np[cls_mask1]
                if len(confs1_np) > 0 and len(cls_mask1) > 0
                else np.empty((0,))
            )

            # Find equivalent class from model2 by name matching or custom mapping
            equivalent_cls2 = None
            cls1_name = class_names1.get(cls)
            if cls1_name:
                # Apply class filter check
                if (model2_class_filter is not None and
                        cls1_name not in model2_class_filter):
                    # Skip this class from model2 if it's filtered out
                    equivalent_cls2 = None
                else:
                    # Find model2 class that maps to this model1 class
                    for cls2_id, cls2_name in class_names2.items():
                        # Check if there's a custom mapping
                        if (class_mapping is not None and
                                cls2_name in class_mapping):
                            if class_mapping[cls2_name] == cls1_name:
                                equivalent_cls2 = cls2_id
                                break
                        else:
                            # Use exact name matching (original behavior)
                            if cls2_name == cls1_name:
                                equivalent_cls2 = cls2_id
                                break
            
            # Get boxes from model2 with equivalent class (if exists)
            if equivalent_cls2 is not None:
                cls_mask2 = (
                    clss2_np == equivalent_cls2 if len(clss2_np) > 0
                    else np.array([])
                )
                cls_boxes2 = (
                    boxes2_np[cls_mask2]
                    if len(boxes2_np) > 0 and len(cls_mask2) > 0
                    else np.empty((0, 4))
                )
                cls_confs2 = (
                    confs2_np[cls_mask2]
                    if len(confs2_np) > 0 and len(cls_mask2) > 0
                    else np.empty((0,))
                )
            else:
                # No equivalent class in model2
                cls_boxes2 = np.empty((0, 4))
                cls_confs2 = np.empty((0,))

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
    model1 = YOLO("/models/yolov8-traffic/nq44m7uc.pt")
    # model2 = YOLO("/models/yolov8-traffic/covctrhe.pt")
    model2 = YOLO("yolo11l.pt")

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

    # Define class mapping from model2 to model1
    # This maps class names from model2 to corresponding class names in model1
    class_mapping = {
        "person": "pedestrian",
        "bicycle": "vehicle_others",
        "car": "car",
        "motorcycle": "motorbike",
        "bus": "bus",
        "truck": "truck"
    }

    # Merge results using IoU-based algorithm
    # Set merge_across_classes=True to merge boxes across all classes
    # Set merge_across_classes=False to merge only within the same class (default)
    # Use model2_class_filter to specify which classes from model2 to include
    # Example: model2_class_filter=["person"] to only include person detections
    merged_data = merge_models_results(
        result1, result2, iou_threshold=0.5, merge_across_classes=True,
        model2_class_filter=['person'],  # Filter specific classes
        class_mapping=class_mapping  # Map model2 classes to model1 classes
    )

    # Apply NMS on all classes after merging
    nms_data = nms_on_all_classes(
        merged_data["boxes"],
        merged_data["confs"],
        merged_data["clss"],
        iou_threshold=0.5,
    )

    # Get class names from first model (assuming both models have same classes)
    class_name = result1.names

    detections = []
    threshold = 0.1

    # Process NMS results
    for box, conf, cls in zip(
        nms_data["boxes"], nms_data["confs"], nms_data["clss"]
    ):
        cls_id = int(cls)
        # Only use classes that exist in model1
        if cls_id in class_name and conf >= threshold:
            label = class_name[cls_id]
            # must be in this format
            detections.append(
                {
                    "confidence": float(conf),
                    "label": label,
                    "points": box.tolist(),
                    "type": "rectangle",
                }
            )

    context.logger.info(
        f"Final detection count after merge and NMS: {len(detections)}"
    )
    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
