import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib
# Use a non-interactive backend to avoid GUI/audio side-effects
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Work around a known gradio/gradio_client schema bug by tolerating boolean schemas
try:
    import gradio_client.utils as _gc_utils
    _orig_json_schema_to_python_type = _gc_utils.json_schema_to_python_type
    def _safe_json_schema_to_python_type(schema):
        try:
            return _orig_json_schema_to_python_type(schema)
        except TypeError:
            # Fallback for schemas like {"additionalProperties": true}
            return "object"
    _gc_utils.json_schema_to_python_type = _safe_json_schema_to_python_type
except Exception:
    pass

# Paths and constants
MODEL_PATH = 'models/resnet50_crop_disease.h5'
DATA_DIR = 'data/plantvillage/plantvillage dataset/color'
IMG_SIZE = (224, 224)
PREFERRED_LAST_CONV = 'conv5_block3_out'

# Sanity checks
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure training saved the model correctly.")

# Load model (no need to compile for inference)
model = load_model(MODEL_PATH, compile=False)

# Get class names (fallback to generic if folder missing)
if os.path.isdir(DATA_DIR):
    CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
else:
    # Fallback to generic names using the model output size
    n_classes = int(model.output_shape[-1])
    CLASSES = [f"class_{i}" for i in range(n_classes)]

# index->class and class->index mappings
class_indices = {i: cls for i, cls in enumerate(CLASSES)}
indices_class = {v: k for k, v in class_indices.items()}

# Auto-detect a valid last conv layer if preferred one is not present
_def_last_conv = None
try:
    model.get_layer(PREFERRED_LAST_CONV)
    _def_last_conv = PREFERRED_LAST_CONV
except Exception:
    for layer in reversed(model.layers):
        try:
            shape = getattr(layer, 'output_shape', None)
            if shape is not None and len(shape) == 4:
                _def_last_conv = layer.name
                break
        except Exception:
            continue
if _def_last_conv is None:
    raise ValueError("Could not find a 4D convolutional layer for Grad-CAM.")


def _fig_to_numpy(fig):
    fig.canvas.draw()
    # Use buffer_rgba to avoid deprecation and convert RGBA->RGB
    buf = np.asarray(fig.canvas.buffer_rgba())
    img_rgb = buf[..., :3].copy()
    plt.close(fig)
    return img_rgb


def get_gradcam_heatmap(img_array, model, last_conv_layer_name=_def_last_conv):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # Numerical stability on normalization
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_array[0] * 255

    return heatmap, superimposed_img.astype(np.uint8), int(class_idx.numpy())


def predict_image(img):
    # Guard against empty inputs
    if img is None:
        return np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)

    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_class_idx = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][predicted_class_idx])
    predicted_class = class_indices[predicted_class_idx]

    heatmap, superimposed_img, _ = get_gradcam_heatmap(img_array, model)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img)
    ax1.set_title('Uploaded Image')
    ax1.axis('off')
    ax2.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')
    ax3.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    ax3.set_title(f'Prediction: {predicted_class} ({confidence:.2f})')
    ax3.axis('off')
    plt.tight_layout()

    return _fig_to_numpy(fig)

# Minimal Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Crop Disease Detection")
    gr.Markdown("Upload a leaf image to predict crop disease and view Grad-CAM heatmap.")
    image_input = gr.Image(type="pil")
    output = gr.Image(type="numpy")
    image_input.change(fn=predict_image, inputs=image_input, outputs=output)

# Launch locally, fallback to share if localhost blocked
try:
    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=None, show_error=True)
except ValueError as e:
    if "localhost is not accessible" in str(e):
        demo.launch(share=True, server_name="0.0.0.0", server_port=None, show_error=True)
    else:
        raise
