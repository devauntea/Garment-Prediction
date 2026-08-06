# Garment Prediction

Image classification pipeline that predicts garment category from clothing photos,
built on the [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
category-attribute dataset using ResNet-18 transfer learning.

The project is structured as four independent, rerunnable stages rather than a single
notebook, so any stage can be re-executed without redoing the ones before it.

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Manifest | `src/build_manifest.py` | Parses DeepFashion annotation files (`list_category_img.txt`, `list_eval_partition.txt`) into a single `manifests/items.csv` with image path, category, and train/val/test split |
| 2. Preprocess | `src/preprocess_images.py` | Materializes the manifest into an `ImageFolder` layout under `data/processed/{train,val,test}/<class>/` |
| 3. Train | `src/train_model.py` | Fine-tunes a pretrained ResNet-18, writes checkpoints to `models/` and run metadata to `reports/knowledge/` |
| 4. Evaluate | `src/evaluate_model.py` | Scores the checkpoint and emits accuracy, a per-class classification report, and a confusion-matrix figure to `reports/` |

## Model

- **Backbone:** ResNet-18, ImageNet pretrained weights (`ResNet18_Weights.DEFAULT`)
- **Input:** 224×224, ImageNet channel normalization
- **Optimizer:** Adam, lr 1e-3, batch size 32
- **Output:** final FC layer resized to the dataset's class count

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the DeepFashion *Category and Attribute Prediction* benchmark and place it at
`data/raw/deepfashion/` so that `img/`, `list_category_img.txt`, and
`list_eval_partition.txt` sit directly inside.

## Usage

Run the stages in order:

```bash
python src/build_manifest.py
python src/preprocess_images.py
python src/train_model.py
python src/evaluate_model.py
```

Evaluation writes a per-class report and confusion matrix to `reports/figures/`, which
makes regressions between training runs visible at a glance.

## Layout

```text
.
├── manifests/          # generated items.csv
├── src/
│   ├── build_manifest.py
│   ├── preprocess_images.py
│   ├── train_model.py
│   └── evaluate_model.py
└── reports/
    ├── figures/        # confusion matrices
    └── knowledge/      # run metadata
```

## License

MIT
