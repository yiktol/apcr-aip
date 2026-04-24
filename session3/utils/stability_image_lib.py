# utils/stability_image_lib.py
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Utility library for generating images with Stability AI models on Amazon Bedrock.
Provides a simplified interface for the Streamlit app.
"""

import base64
import io
import json
import logging
import boto3
from PIL import Image
from botocore.config import Config
from botocore.exceptions import ClientError


class StabilityImageError(Exception):
    """Custom exception for errors returned by Stability AI models"""

    def __init__(self, message):
        self.message = message


logger = logging.getLogger(__name__)

# Available Stability AI text-to-image models on Bedrock
MODELS = {
    "stability.stable-image-core-v1:1": "Stable Image Core — Fast & affordable",
    "stability.sd3-5-large-v1:0": "Stable Diffusion 3.5 Large — High quality",
    "stability.sd3-ultra-v1:1": "Stable Image Ultra — Photorealistic premium",
}

DEFAULT_MODEL_ID = "stability.stable-image-core-v1:1"

# Supported aspect ratios
ASPECT_RATIOS = {
    "1:1": "1:1 (Square)",
    "16:9": "16:9 (Landscape)",
    "9:16": "9:16 (Portrait)",
    "3:2": "3:2 (Classic Photo)",
    "2:3": "2:3 (Portrait Photo)",
    "4:5": "4:5 (Social Media)",
    "5:4": "5:4 (Landscape Photo)",
    "21:9": "21:9 (Ultra-wide)",
    "9:21": "9:21 (Ultra-tall)",
}


def generate_image(
    prompt_content,
    negative_prompt=None,
    model_id=DEFAULT_MODEL_ID,
    aspect_ratio="1:1",
    output_format="png",
    seed=0,
):
    """
    Generate an image using a Stability AI model via Amazon Bedrock.

    Args:
        prompt_content (str): The main prompt describing what to generate.
        negative_prompt (str, optional): What should NOT be in the image.
        model_id (str): The Bedrock model ID. Defaults to Stable Image Core.
        aspect_ratio (str): Aspect ratio string (e.g. "1:1", "16:9").
        output_format (str): "png" or "jpeg". Defaults to "png".
        seed (int): Random seed for reproducibility. 0 = random.

    Returns:
        PIL.Image: The generated image as a PIL Image object.

    Raises:
        StabilityImageError: If image generation fails.
    """
    logger.info("Generating image with model %s", model_id)

    body = {"prompt": prompt_content}

    if negative_prompt and negative_prompt.strip():
        body["negative_prompt"] = negative_prompt.strip()

    if aspect_ratio and aspect_ratio != "1:1":
        body["aspect_ratio"] = aspect_ratio

    if output_format:
        body["output_format"] = output_format

    if seed and seed > 0:
        body["seed"] = seed

    try:
        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-west-2",
            config=Config(read_timeout=300),
        )

        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response["body"].read())

        # Check for content filter or inference errors
        finish_reasons = response_body.get("finish_reasons", [])
        if finish_reasons and finish_reasons[0] is not None:
            raise StabilityImageError(
                f"Generation filtered: {finish_reasons[0]}"
            )

        images = response_body.get("images", [])
        if not images:
            raise StabilityImageError("No images returned from the model")

        image_bytes = base64.b64decode(images[0])
        image = Image.open(io.BytesIO(image_bytes))

        logger.info("Successfully generated image with model %s", model_id)
        return image

    except ClientError as err:
        error_message = err.response.get("Error", {}).get("Message", str(err))
        logger.error("AWS client error: %s", error_message)
        raise StabilityImageError(f"AWS service error: {error_message}")

    except StabilityImageError:
        raise

    except Exception as err:
        logger.error("Unexpected error: %s", str(err))
        raise StabilityImageError(f"Unexpected error: {str(err)}")


def save_image_to_bytes(image, fmt="PNG"):
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


__all__ = [
    "generate_image",
    "StabilityImageError",
    "MODELS",
    "DEFAULT_MODEL_ID",
    "ASPECT_RATIOS",
    "save_image_to_bytes",
]
