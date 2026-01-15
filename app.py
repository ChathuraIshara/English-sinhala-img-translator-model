import io
import json
import os
import ftfy
import torch
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from flask_cors import CORS

# Import your provided constants file
import handwritten_constants
from model_utils import SimpleCNN, get_transforms, get_device

# --- CONFIGURATION ---
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pth")
LABEL_MAP_PATH = os.path.join(ARTIFACT_DIR, "label_map.json")
DEFAULT_IMAGE_PATH = os.environ.get("PREDICT_IMAGE_PATH", "new.jpg")

# --- DECODE SINHALA LABELS ---
# This converts the "à¶…" encoding in your file into real characters like "අ"
# We do this once at startup for efficiency.
CLEAN_SINHALA_LABELS = [ftfy.fix_text(label) for label in handwritten_constants.TRUE_LABEL]

app = Flask(__name__)
CORS(app)

device = get_device()
transform = get_transforms()
model = None
idx_to_char = None

UPLOAD_FORM = """
<!doctype html>
<html>
<head>
    <title>Sinhala Character Recognition</title>
    <style>
        body { font-family: sans-serif; text-align: center; padding: 50px; }
        .result-box { margin-top: 20px; padding: 20px; border: 1px solid #ddd; display: inline-block; }
        .sinhala-char { font-size: 72px; color: #2c3e50; display: block; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Sinhala Handwritten Character Prediction</h1>
    <form method="post" enctype="multipart/form-data" action="/predict">
      <input type="file" name="file" accept="image/*">
      <input type="submit" value="Upload and Predict">
    </form>
    <p>Or hit <code>/predict-local</code> to read the default image on disk.</p>
</body>
</html>
"""

def load_artifacts():
    global model, idx_to_char

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Load the mapping from model index to Folder Name
    idx_to_char = checkpoint.get("idx_to_char")
    if idx_to_char is None:
        if os.path.exists(LABEL_MAP_PATH):
            with open(LABEL_MAP_PATH) as f:
                idx_to_char = json.load(f).get("idx_to_char")
    
    if idx_to_char is None:
        raise ValueError("Label map not found in checkpoint or label_map.json")

    # Initialize and load model
    num_classes = len(idx_to_char)
    model_local = SimpleCNN(num_classes=num_classes)
    model_local.load_state_dict(checkpoint["model_state_dict"])
    model_local.to(device)
    model_local.eval()
    model = model_local

def get_label_info_for_model_index(model_index: int):
    """
    Given a model class index (0-based), return:
    - folder_name (string as in dataset)
    - alphabet_index (0-based index into CLEAN_SINHALA_LABELS)
    - sinhala_letter (decoded character)
    """
    # Map model index to folder name via idx_to_char
    if isinstance(next(iter(idx_to_char.keys())), str):
        folder_name = str(idx_to_char.get(str(model_index)))
    else:
        folder_name = str(idx_to_char.get(model_index))

    # Map folder name to alphabet index via constants
    alphabet_index = None
    if folder_name is not None:
        alphabet_index = handwritten_constants.CLASS_INDICES.get(folder_name)

    # Resolve Sinhala letter
    if alphabet_index is not None and alphabet_index < len(CLEAN_SINHALA_LABELS):
        sinhala_letter = CLEAN_SINHALA_LABELS[alphabet_index]
    else:
        sinhala_letter = "Unknown"

    return folder_name, alphabet_index, sinhala_letter

def predict_image(img: Image.Image) -> dict:
    """
    Processes the image and maps the prediction to the actual Sinhala letter.
    """
    img_t = transform(img).unsqueeze(0).to(device)


    with torch.no_grad():
        outputs = model(img_t)
        pred_idx = outputs.argmax(1).item()
        
    # Resolve all mapping via shared helper
    folder_name, alphabet_index, predicted_letter = get_label_info_for_model_index(pred_idx)

    return {
        "folder_name": folder_name,
        "predicted_letter": predicted_letter,
        "class_index": pred_idx,
        "alphabet_index": alphabet_index
    }

def folder_to_sinhala(folder_name: str) -> str:
    """
    Convert dataset folder name (e.g. '17') to real Sinhala character.
    """
    # Step 1 — get alphabet index
    alphabet_index = handwritten_constants.CLASS_INDICES.get(str(folder_name))

    if alphabet_index is None:
        return "Unknown"

    # Step 2 — get broken label
    raw_label = handwritten_constants.TRUE_LABEL[alphabet_index]

    # Step 3 — fix encoding
    return ftfy.fix_text(raw_label)


def get_label_info_for_model_index(model_index: int):
    # model → folder
    folder_name = str(idx_to_char[str(model_index)])

    # folder → Sinhala
    sinhala_letter = folder_to_sinhala(folder_name)

    # also return alphabet index (optional)
    alphabet_index = handwritten_constants.CLASS_INDICES.get(folder_name)

    return folder_name, alphabet_index, sinhala_letter


def load_image_from_disk(path: str) -> Image.Image:
    resolved = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Image not found at {resolved}")
    return Image.open(resolved).convert("RGB")

@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_FORM)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = predict_image(img)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/predict-local", methods=["GET"])
def predict_local():
    try:
        img = load_image_from_disk(DEFAULT_IMAGE_PATH)
        result = predict_image(img)
        result["source"] = DEFAULT_IMAGE_PATH
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc), "source": DEFAULT_IMAGE_PATH}), 500

@app.route("/labels", methods=["GET"])
def get_labels():
    """
    Returns a mapping of all indexes to their corresponding Sinhala letters.
    """
    try:
        if idx_to_char is None:
            return jsonify({"error": "Model not loaded yet"}), 500
        
        # Create a complete mapping: model_index -> folder_name -> alphabet_index -> sinhala_letter
        labels_map = []
        for model_idx, _ in idx_to_char.items():
            # model_idx may be a string key; normalize to int for helper
            try:
                mi = int(model_idx)
            except Exception:
                mi = model_idx

            folder_name, alphabet_index, sinhala_letter = get_label_info_for_model_index(mi)
            labels_map.append({
                "model_index": int(model_idx),
                "folder_name": folder_name,
                "alphabet_index": alphabet_index,
                "sinhala_letter": sinhala_letter
            })
        
        # Sort by model_index for easier reading
        labels_map.sort(key=lambda x: x["model_index"])
        
        return jsonify({
            "total_classes": len(labels_map),
            "labels": labels_map
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    load_artifacts()
    print(f"Loaded model with {len(idx_to_char)} classes on {device}")
    app.run(host="0.0.0.0", port=5000, debug=True)