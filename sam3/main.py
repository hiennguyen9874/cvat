# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
import numpy as np
from model_handler import ModelHandler
from skimage.measure import approximate_polygon, find_contours


def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("call handler")
    data = event.body

    # helper to get options
    def get_option(name, default):
        return data.get(name, default)

    # Decode image
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")

    # Parse points
    pos_points = data.get("pos_points", [])
    neg_points = data.get("neg_points", [])
    obj_bbox = data.get("obj_bbox", None)  # [x, y, x, y]

    context.logger.info(
        f"Handler inputs: pos_points={len(pos_points)}, neg_points={len(neg_points)}, obj_bbox={obj_bbox}, image_size={image.size}"
    )

    # Convert to format expected by ModelHandler
    points = []
    labels = []

    for p in pos_points:
        points.append(p)
        labels.append(1)

    for p in neg_points:
        points.append(p)
        labels.append(0)

    # Run inference
    mask = context.user_data.model.handle(image, points, labels, obj_bbox)
    context.logger.info(
        f"Inference returned mask: {'None' if mask is None else mask.shape}"
    )

    # If no mask returned, return empty
    if mask is None:
        print("Mask is None")
        return context.Response(
            body=json.dumps(
                {
                    "points": [],
                    "mask": [],
                }
            ),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    # Polygon
    contours = find_contours(mask, 0.5)
    context.logger.info(f"Contours found: {len(contours)}")
    # Use the largest contour
    if contours:
        contour = max(contours, key=len)
        context.logger.info(f"Largest contour length: {len(contour)}")
        contour = np.flip(contour, axis=1)  # (row, col) -> (x, y)
        contour = approximate_polygon(contour, tolerance=2.5)
        points_list = contour.tolist()
    else:
        points_list = []

    context.logger.info(f"Final points_list length: {len(points_list)}")

    return context.Response(
        body=json.dumps(
            {
                "points": points_list,
                "mask": mask.tolist(),
            }
        ),
        headers={},
        content_type="application/json",
        status_code=200,
    )
