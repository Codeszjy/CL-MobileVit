# CL-MobileViT: Wheat Leaf Disease Recognition and Safety-Gated Pesticide Retrieval

This project uses MobileViT as its visual backbone for six-class wheat leaf image classification. Training combines cross-entropy with supervised contrastive learning. During inference, the system produces a CL-CAM visualization by default (with Grad-CAM retained as a comparison baseline) and then passes the classification result to a safety-gated retrieval layer backed by a pesticide-registration CSV file.

## System Workflow

```text
Wheat leaf image
    ↓
CL-MobileViT six-class classification
    ↓
CL-CAM visualization (Grad-CAM optional)
    ↓
Confidence, healthy-class, and disease-name safety gates
    ↓
Exact retrieval from the wheat pesticide-registration CSV
    ↓
Candidate registration records and safety notices (not an automatic prescription)
```

The CSV knowledge layer is placed after the neural network. It uses only the Python standard library to read the data, does not participate in the model forward pass, and adds no model parameters. The current implementation is a lightweight and traceable rule-based retrieval prototype; it should not be described as an implemented graph-database knowledge graph.

## Implementation Overview

- **Visual backbone:** MobileViT with `xx_small`, `x_small`, and `small` configurations. ECA channel attention is added to the inverted residual blocks.
- **Training objective:** Two augmented views are generated for each training image. The model outputs both classification logits and a 128-dimensional projection feature. The total loss is `cross-entropy (label smoothing=0.1) + sup_con_weight × supervised contrastive loss`.
- **Validation:** Validation uses a single image view. Loss, Accuracy, macro-averaged Precision, Recall, and F1 are printed, but `best_model.pth` is currently selected using validation Accuracy only.
- **Explanation and retrieval:** The default gradient-free CL-CAM clusters `model.layer_5` feature channels, scores six representative masks, and combines a cognition-base map with a cognition-scissors map. Grad-CAM remains selectable for matched comparisons. The classification result is then passed to a safety-medication rule layer that is separate from the model.
- **Fail-closed behavior:** If the CSV is missing, malformed, or unreadable, classification and CAM visualization can still be produced, but the candidate list remains empty and the report status is `knowledge_base_unavailable_manual_review`.

## Project Structure

| Path | Purpose |
| --- | --- |
| `model.py` | MobileViT, ECA, classification head, and contrastive-learning projection head |
| `model_config.py` | Network configurations for `xx_small`, `x_small`, and `small` |
| `transformer.py` | Transformer encoder implementation |
| `my_dataset.py` | Two-view training data and single-view validation data |
| `utils.py` | Data splitting, supervised contrastive loss, training, and validation |
| `train.py` | Training entry point; currently fixed to `mobile_vit_small` |
| `cam_methods.py` | CL-CAM and Grad-CAM implementations |
| `predict.py` | Classification, selectable CAM visualization, safety-gated retrieval, and report export |
| `medication_knowledge.py` | Disease mapping, CSV validation, candidate filtering, ranking, and report generation |
| `test_cam_methods.py` | CAM shape, finiteness, gradient, and prediction-preservation tests |
| `test_medication_knowledge.py` | Knowledge-layer unit tests that depend on the real CSV file |
| `class_indices.json` | Class-index mapping for the current six-class model |
| `weights/` | Model checkpoints |
| `outputs/` | Inference artifacts |

The current class order is:

```text
0 Brown Rust
1 Healthy
2 Leaf Blight
3 Mildew
4 Septoria
5 Yellow Rust
```

## Environment Setup

Run the commands from the project root. The Python dependencies below are derived from the actual imports in the code:

- PyTorch and Torchvision
- Pillow and NumPy
- tqdm and Matplotlib
- TensorBoard for training logs

First install versions of `torch` and `torchvision` that match your CPU/CUDA environment using the official PyTorch installation instructions. Then install the remaining dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pillow numpy tqdm matplotlib tensorboard
```

Run a quick compilation check:

```powershell
python -m compileall -q .
```

## Data Preparation

The dataset, trained checkpoints, and pesticide-registration CSV are not distributed with this repository. Add them at the paths below or override the corresponding paths from the command line or `predict.py` configuration.

### Training Images

By default, `train.py` reads `data/Wheat`. Each first-level subdirectory represents one class. Directory names are sorted lexicographically to generate labels, and the project-root `class_indices.json` is overwritten:

```text
data/
└── Wheat/
    ├── Brown Rust/
    │   ├── image_001.jpg
    │   └── ...
    ├── Healthy/
    ├── Leaf Blight/
    ├── Mildew/
    ├── Septoria/
    └── Yellow Rust/
```

Data-loading rules:

- Supported extensions are `.jpg`, `.JPG`, `.png`, and `.PNG`.
- Every image must be in RGB mode. Grayscale or other image modes raise an error.
- For each class, 20% of the images are sampled for validation using the fixed random seed `0`; the remaining images are used for training.
- To remain compatible with a six-class checkpoint using the provided mapping, directory names and their sorted order should match the class list above.

The current loader expects this flat class-directory layout and creates its own training/validation split. A dataset that already contains separate `train/` and `test/` directories is not consumed correctly without adapting the loader or reorganizing the input paths.

### Pesticide-Registration CSV

The default path is:

```text
data/knowledge/China_pesticides_wheat.csv
```

The CSV must use UTF-8 or UTF-8 with BOM and contain the following headers:

```text
登记证号, 农药名称, 农药类别, 登记证持有人, 剂型, 毒性,
总有效成分含量, 有效成分, 有效成分英文名, 作物/场所, 防治对象,
用药量（制剂量/亩）, 施用方法, 批准日期, 最新批准日期,
使用技术要求, 注意事项, 中毒急救措施, 储存和运输方法, 备注, 使用方法序号
```

If the CSV is stored elsewhere, specify it with `predict.py --pesticide-csv` or `medication_knowledge.py --csv`.

## Training

The current training entry point is fixed to `mobile_vit_small`. By default, it expects the 1,000-class `weights/mobile_vit_small.pt` checkpoint and removes the incompatible classification-head and projection-head parameters. Supply that checkpoint or use `--weights ""` to train from random initialization:

```powershell
python train.py `
  --data-path "data\Wheat" `
  --num_classes 6 `
  --epochs 50 `
  --batch-size 16 `
  --lr 0.0002 `
  --sup_con_weight 0.5 `
  --weights "weights\mobile_vit_small.pt" `
  --device cuda:0
```

Training produces:

- `weights/best_model.pth`: the checkpoint with the highest validation Accuracy.
- `weights/latest_model.pth`: the checkpoint from the most recent epoch.
- `runs/`: TensorBoard logs.
- `class_indices.json`: a class mapping regenerated from the dataset directory names.

View the logs with:

```powershell
tensorboard --logdir runs
```

Training notes:

- `train.py` always writes `weights/best_model.pth` and `weights/latest_model.pth`. Back up any files with those names before starting a new run.
- Add `--freeze-layers` to train only the classifier and projection head; omit it for full fine-tuning.
- Start training from the project root. Otherwise, `class_indices.json`, `weights/`, and `runs/` are written relative to the current working directory.

## Inference

### Code-Side Inference Configuration

All routine inference and CL-CAM settings are grouped in the `DEFAULT_CONFIG` block near the top of `predict.py`. After placing a compatible checkpoint and image at the configured paths, run:

```powershell
python predict.py
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `cam_method` | `cl-cam` | Explanation method; `grad-cam` is also available |
| `cam_target_layer` | `layer_5` | Feature layer used to construct the map |
| `cam_clusters` | `6` | Number of channel clusters and representative masks |
| `cam_beta` | `0.5` | Balance between the base and scissors maps |
| `cam_seed` | `0` | K-means initialization seed |
| `cam_max_iterations` | `100` | Maximum K-means iterations |
| `cam_overlay_alpha` | `0.4` | Heatmap opacity in the overlay |

Command-line arguments remain available and override the code-side defaults for a single run. For example:

```powershell
python predict.py --image "D:\wheat_images\sample.jpg"
```

Full example:

```powershell
python predict.py `
  --image "D:\wheat_images\sample.jpg" `
  --weights "weights\best_model.pth" `
  --model-variant auto `
  --class-indices "class_indices.json" `
  --pesticide-csv "data\knowledge\China_pesticides_wheat.csv" `
  --cam-method cl-cam `
  --cam-target-layer layer_5 `
  --cam-clusters 6 `
  --cam-beta 0.5 `
  --cam-seed 0 `
  --cam-max-iterations 100 `
  --cam-overlay-alpha 0.4 `
  --confidence-threshold 0.85 `
  --top-k 5 `
  --device cpu `
  --output-dir "outputs\sample"
```

Each successful model run generates:

- `cl_cam_result.png` for CL-CAM by default, or `grad_cam_result.png` with `--cam-method grad-cam`: the model-aligned center crop, heatmap, and overlay.
- `medication_recommendation.json`: model information, per-class probabilities, `cam_method`, `cam_path`, CAM parameters/diagnostics, knowledge-layer status, and candidate records.
- `medication_recommendation.txt`: a human-readable text report.

### CL-CAM Method Selection

CL-CAM clusters feature channels by spatial activation pattern, averages each cluster, scores the representative masks with additional forward passes, and combines the highest-scoring cognition-base map with the lowest-scoring cognition-scissors map. The `beta` parameter is a balance factor in `[0, 1]`; `0.5` is the neutral implementation default. Before masking, this implementation min-max normalizes each representative map to `[0, 1]` so that MobileViT activation scale does not arbitrarily rescale the normalized image tensor.

Use the retained baseline with:

```powershell
python predict.py --image "D:\wheat_images\sample.jpg" --cam-method grad-cam
```

CL-CAM scores `Q` masked inputs (`Q=6` by default). This implementation evaluates those inputs together in one batched model call, whereas Grad-CAM uses one ordinary forward pass and one backward pass. The maps can be sensitive to `Q`; therefore, `Q=6` is a starting point rather than a proven optimum for this MobileViT wheat dataset, and any benefit must be established with matched evaluation.


## Pesticide Retrieval

The rule layer can be queried without loading the neural network:

```powershell
python medication_knowledge.py --disease Mildew --confidence 0.95
```

Specify an external CSV with:

```powershell
python medication_knowledge.py `
  --disease "Yellow Rust" `
  --confidence 0.95 `
  --csv "D:\data\China_pesticides_wheat.csv" `
  --output-dir "outputs\knowledge_test"
```

## Usage Boundaries

- CL-CAM and Grad-CAM are diagnostic visualizations. On their own, they do not demonstrate reliable lesion localization, segmentation accuracy, or causal explanation.
- The output contains candidate registration records, not an automatic prescription. The CSV does not include registration expiry dates or live registration status, so the system cannot claim that a record remains valid on the day of use.
- Before any real-world application, verify the registration number, current official label, crop, control target, dosage, application method, pre-harvest interval, and local regulations, and consult a qualified crop-protection professional.
- Candidate ranking considers exact matching, a more conservative toxicity order, newer approval dates, and registration numbers. It is used only for display and does not represent efficacy ranking or product endorsement.
