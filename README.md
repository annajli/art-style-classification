# Art Style Classification

Deep learning models for classifying art styles using the [WikiArt dataset](https://www.kaggle.com/datasets/steubk/wikiart).

## Repo structure

```
art-style-classification/
├── models/                 # Promoted model architectures (classes ready to import)
│   ├── baseline_cnn.py     # ResNet18, ResNet50 — frozen backbone + tunable MLP head
│   └── vit.py              # ViT_B16 — partial unfreeze + tunable MLP head
├── utils/                  # Shared utilities
│   ├── dataset.py          # WikiArtDataset + data path helper
│   ├── metrics.py          # Accuracy, confusion matrix, classification report
│   └── train_val.py        # train_loop, test_loop, plot_history
├── notebooks/              # Colab experiment notebooks
│   ├── 01_baseline_training_template.ipynb   # CNN baseline template — do not edit
│   ├── 02_vit_experiment_template.ipynb      # ViT template — do not edit
│   └── <name>_<experiment>.ipynb             # individual experiment notebooks
├── requirements.txt
└── README.md
```

## Experiment workflow

**Code lives here. Compute and data live in Colab.**

1. **Pick a template** — open the relevant template in Colab (`01` for CNNs, `02` for ViT) and save a copy with a descriptive name (e.g. `anna_resnet50_unfrozen.ipynb`)
2. **Tune the config cell** — adjust `HIDDEN_DIM`, `DROPOUT`, `LR`, `NUM_EPOCHS` etc. without touching anything else
3. **Run your experiment** — checkpoints save automatically to Google Drive
4. **Push your notebook** — commit and push your `.ipynb` to `notebooks/` so the team can see results
5. **Promote winning architectures** — once a model performs well, add it as a class in `models/` so others can import and build on it

## Getting started in Colab

1. Go to [Google Colab](https://colab.research.google.com/) → **File → Open notebook → GitHub**
2. Paste this repo's URL and open your chosen template notebook
3. **File → Save a copy in Drive**, then rename it to `<yourname>_<experiment>.ipynb`
4. Set runtime to GPU: **Runtime → Change runtime type → A100**
5. Add credentials (see below) and run all cells

## Kaggle credentials

You need a Kaggle API key to download the dataset.

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**
2. In Colab, open the 🔑 **Secrets** panel (left sidebar) and add two secrets:
   - `KAGGLE_USERNAME` — your Kaggle username
   - `KAGGLE_KEY` — the API key from the downloaded `kaggle.json`
3. The credentials cell reads them automatically via `google.colab.userdata.get()`

## Running locally (optional)

```bash
pip install -r requirements.txt
export WIKIART_PATH=/path/to/wikiart
```

Then import from `utils` and `models` directly.
