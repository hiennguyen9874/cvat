# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import torch
import numpy as np
from transformers import Sam3TrackerProcessor, Sam3TrackerModel


class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use the specific SAM3 model
        self.model_id = "/models/sam3"

        # Load processor and model
        self.processor = Sam3TrackerProcessor.from_pretrained(self.model_id)
        self.model = Sam3TrackerModel.from_pretrained(self.model_id).to(self.device)

    def handle(self, image, points, labels, box=None):
        # image: PIL Image
        # points: list of [x, y]
        # labels: list of int (1 or 0)
        # box: list of [x_min, y_min, x_max, y_max]
        print(
            f"ModelHandler handle: points={len(points) if points else 0}, labels={len(labels) if labels else 0}, box={box}, image_size={image.size}"
        )

        # Prepare inputs
        prompt_args = {}

        if points:
            # Shape: (batch, num_objects, num_points, 2)
            prompt_args["input_points"] = [[points]]
            # Shape: (batch, num_objects, num_points)
            prompt_args["input_labels"] = [[labels]]

        if box:
            # Flatten box if it's [[xtl, ytl], [xbr, ybr]]
            if len(box) == 2 and isinstance(box[0], list):
                box = [*box[0], *box[1]]
            # Shape: (batch, num_objects, 4)
            prompt_args["input_boxes"] = [[box]]

        if not prompt_args:
            return None

        print(f"prompt_args: {prompt_args}")

        # Preprocess
        inputs = self.processor(
            images=image,
            **prompt_args,
            return_tensors="pt",
        ).to(self.device)
        print(
            f"Preprocess inputs: {[(k, v.shape if hasattr(v, 'shape') else len(v)) for k, v in inputs.items()]}"
        )

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        print(f"Inference outputs keys: {outputs.keys()}")

        # Post-process
        # outputs.pred_masks shape: [batch, num_objects, num_masks, H, W]
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        # masks shape: [num_objects, num_masks, H, W]
        print(f"Post-process masks shape: {masks.shape}")

        if masks.shape[0] > 0:
            # Taking the first mask of the first object
            # Usually index 0 for num_masks is the first prediction
            mask = masks[0][0]
            print(f"Returning mask: shape={mask.shape}, dtype={mask.dtype}")
            return mask.numpy().astype(np.uint8)

        return None
