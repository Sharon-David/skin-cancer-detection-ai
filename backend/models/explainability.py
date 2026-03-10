"""
explainability.py — Grad-CAM heatmap generation

Generates a heatmap highlighting which regions of the image
influenced the model's prediction most.
"""

import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import albumentations as A

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_XLA"]        = "0"

IMG_SIZE = 224

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    arr = val_transform(image=arr)["image"]
    return np.expand_dims(arr, axis=0).astype(np.float32)

def get_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the name of the last convolutional layer."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Handle nested model (EfficientNet inside functional model)
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    raise ValueError("No Conv2D layer found in model")

def generate_gradcam(
    model: tf.keras.Model,
    image_bytes: bytes,
    alpha: float = 0.4
) -> str:
    """
    Generate Grad-CAM heatmap overlaid on the original image.
    Returns base64-encoded PNG string.
    """
    # Preprocess
    x = preprocess(image_bytes)

    # Find last conv layer
    try:
        last_conv_name = get_last_conv_layer(model)
    except ValueError:
        return None

    # Build grad model
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )
    except Exception:
        # Try nested model
        base = model.layers[1]
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[base.get_layer(last_conv_name).output, model.output]
        )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, 0]

    grads       = tape.gradient(loss, conv_outputs)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out    = conv_outputs[0]
    heatmap     = conv_out @ pooled[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap).numpy()

    # Normalise
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # Resize to original image size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img)

    # Colormap (red = high activation)
    colormap = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    colormap[:, :, 0] = heatmap_arr                          # Red channel
    colormap[:, :, 1] = (heatmap_arr * 0.3).astype(np.uint8) # slight green

    # Overlay on original image
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE))
    orig_arr    = np.array(orig)
    overlay_arr = (orig_arr * (1 - alpha) + colormap * alpha).astype(np.uint8)
    overlay_img = Image.fromarray(overlay_arr)

    # Encode to base64
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
