# Sinhala Handwritten Character Recognition

A Flask API and training pipeline for recognizing Sinhala handwritten characters using a CNN. Includes scripts to train from the dataset and serve predictions.

## Requirements
- Python 3.9+ (tested with 3.10)
- GPU optional; CPU works (slower)
- Install deps:

```bash
pip install -r requirements.txt
```

## Dataset Layout
Dataset source: [Sinhala Letter 454 (Kaggle)](https://www.kaggle.com/datasets/sathiralamal/sinhala-letter-454/)

Place images under `dataset/` with one folder per class ID (1..453). Example:

```
dataset/
  1/    # class folder
    img1.png
    img2.png
  2/
    ...
```

`handwritten_constants.CLASS_INDICES` defines the mapping from folder name (e.g., "1") to the 0-based label index, and `handwritten_constants.TRUE_LABEL` holds the Sinhala characters.

## Training
This trains the CNN, splits 80/20 train/val, and writes artifacts to `artifacts/`.

```bash
python train_model.py \
  --dataset dataset \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-3 \
  --out artifacts
```

Outputs:
- `artifacts/model.pth` — weights + `idx_to_char`
- `artifacts/label_map.json` — serialized `idx_to_char`

## Running the API
The app loads `artifacts/model.pth` and `label_map.json`.

Env vars (optional):
- `ARTIFACT_DIR` (default `artifacts`)
- `PREDICT_IMAGE_PATH` (default `new.jpg` for `/predict-local`)

Start the server:

```bash
python app.py
```

### Endpoints
- `GET /` — Simple upload form
- `POST /predict` — Send an image file field `file`
- `GET /predict-local` — Uses `PREDICT_IMAGE_PATH` on disk
- `GET /labels` — Returns mapping of model index → folder_name → alphabet_index → sinhala_letter

Examples:

```bash
# Predict from a file
curl -X POST -F "file=@path/to/image.png" http://localhost:5000/predict

# Predict using local default image
curl http://localhost:5000/predict-local

# Inspect label mapping
curl http://localhost:5000/labels
```

Response example:

```json
{
  "folder_name": "1",
  "prediction": "අ",
  "class_index": 0,
  "alphabet_index": 0
}
```

## Notes
- Do **not** commit large artifacts (`artifacts/model.pth`) or `dataset/` to Git.
- If you change `handwritten_constants.CLASS_INDICES` or dataset folder names, retrain so `idx_to_char` and the constants stay aligned.
