#!/usr/bin/env python3
"""SAM3 Demo Script"""

import os
import torch
from PIL import Image

# Turn on tfloat32 for Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Loading SAM3 model...")
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Get paths
sam3_root = os.path.dirname(os.path.abspath(__file__))
bpe_path = os.path.join(sam3_root, "assets/bpe_simple_vocab_16e6.txt.gz")
image_path = os.path.join(sam3_root, "assets/images/truck.jpg")

# Build model
with torch.autocast("cuda", dtype=torch.bfloat16):
    model = build_sam3_image_model(bpe_path=bpe_path)

    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    # Create processor
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)

    # Text prompt
    prompt = "truck"
    print(f"Running inference with prompt: '{prompt}'")
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # Get results
    masks = inference_state.get("masks", [])
    boxes = inference_state.get("boxes", [])
    scores = inference_state.get("scores", [])

    print(f"\nResults:")
    print(f"  Number of detections: {len(masks) if masks is not None else 0}")
    if scores is not None and len(scores) > 0:
        print(f"  Scores: {scores}")
    if boxes is not None and len(boxes) > 0:
        print(f"  Boxes: {boxes}")

    # Save result
    if masks is not None and len(masks) > 0:
        import numpy as np
        mask = masks[0].cpu().numpy() if torch.is_tensor(masks[0]) else masks[0]
        if mask.ndim == 3:
            mask = mask[0]

        # Create overlay
        img_array = np.array(image)
        overlay = img_array.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red overlay
        result = (0.6 * img_array + 0.4 * overlay).astype(np.uint8)

        output_path = os.path.join(sam3_root, "demo_output.jpg")
        Image.fromarray(result).save(output_path)
        print(f"\nSaved result to: {output_path}")

print("\nDemo completed!")
