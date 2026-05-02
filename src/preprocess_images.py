from pathlib import Path
import shutil
import pandas as pd
from PIL import Image

MANIFEST_PATH = Path("manifests/items.csv")
OUT_DIR = Path("data/processed")
IMAGE_SIZE = (224, 224)


def safe_filename(image_path: str) -> str:
    """
    Converts nested DeepFashion paths into safe unique filenames.
    Example:
    data/raw/deepfashion/img/Striped_Dress/img_00000001.jpg
    becomes:
    Striped_Dress_img_00000001.jpg
    """
    path = Path(image_path)
    parent = path.parent.name
    return f"{parent}_{path.name}"


def preprocess_image(src_path: Path, dst_path: Path):
    """
    Opens an image, converts it to RGB, resizes it, and saves it.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img.save(dst_path, quality=95)


def main():
    df = pd.read_csv(MANIFEST_PATH)

    copied = 0
    skipped = 0

    for _, row in df.iterrows():
        src_path = Path(row["image_path"])
        split = row["split"]
        label = row["label"]

        dst_name = safe_filename(row["image_path"])
        dst_path = OUT_DIR / split / label / dst_name

        if not src_path.exists():
            skipped += 1
            continue

        preprocess_image(src_path, dst_path)
        copied += 1

    print(f"Preprocessing complete.")
    print(f"Copied/resized images: {copied}")
    print(f"Skipped missing images: {skipped}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()