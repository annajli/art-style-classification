import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


def get_data_path(colab_path: str = None) -> Path:
    """
    Returns the path to the WikiArt dataset.

    In Colab: pass the path returned by kagglehub.dataset_download().
    Locally: set WIKIART_PATH env var, or pass it explicitly.

    Example (Colab):
        import kagglehub
        raw_path = kagglehub.dataset_download("steubk/wikiart")
        data_path = get_data_path(colab_path=raw_path)
    """
    if colab_path:
        return Path(colab_path)
    env_path = os.environ.get("WIKIART_PATH")
    if env_path:
        return Path(env_path)
    raise ValueError(
        "No data path found. Either pass colab_path= or set the WIKIART_PATH env var."
    )


class WikiArtDataset(Dataset):
    """
    WikiArt dataset loader.

    Expects the dataset root to contain one subdirectory per art style,
    each holding the corresponding images (standard ImageFolder layout).

    Args:
        root:      Path to dataset root (returned by get_data_path).
        transform: torchvision transform pipeline. Defaults to a
                   standard 224×224 ImageNet-normalised transform.
        split:     'train', 'val', or 'test' — if the root contains
                   these subdirs. Leave None to use root directly.
    """

    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, root: Path, transform=None, split: str = None):
        self.root = Path(root) / split if split else Path(root)
        self.transform = transform or self.DEFAULT_TRANSFORM

        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = [
            (img_path, self.class_to_idx[img_path.parent.name])
            for cls in self.classes
            for img_path in (self.root / cls).iterdir()
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Augmentation transform for training — geometric only, no colour shifts
# (colour palette is style-defining in WikiArt)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class TransformSubset(Dataset):
    """
    Wraps a Subset and applies a custom transform, bypassing the base
    dataset's transform. Use this to give train/val splits different
    augmentation pipelines without scanning the filesystem twice.
    """
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label


def stratified_split(dataset, val_split=0.1, seed=42):
    """
    Split dataset indices into train/val preserving per-class proportions.

    Returns:
        train_indices, val_indices — lists of integer indices into dataset.samples.
    """
    rng = random.Random(seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_idx, val_idx = [], []
    for indices in class_indices.values():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    return train_idx, val_idx


def make_sampler(dataset, indices):
    """
    Returns a WeightedRandomSampler that equalises class frequencies,
    oversampling minority classes so each class is seen equally per epoch.
    Pass to DataLoader as sampler= and set shuffle=False.
    """
    labels        = [dataset.samples[i][1] for i in indices]
    class_counts  = Counter(labels)
    weights       = [1.0 / class_counts[dataset.samples[i][1]] for i in indices] 
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
