from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/deepfashion")
IMG_DIR = RAW_DIR / "img"
CATEGORY_FILE = RAW_DIR / "list_category_img.txt"
SPLIT_FILE = RAW_DIR / "list_eval_partition.txt"
OUT_FILE = Path("manifests/items.csv")


def read_category_file(path: Path) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # DeepFashion annotation files usually have:
    # line 1 = number of rows
    # line 2 = column names
    # remaining lines = data
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            image_name = parts[0]
            category_id = int(parts[1])
            rows.append({"image_name": image_name, "category_id": category_id})

    return pd.DataFrame(rows)


def read_split_file(path: Path) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            image_name = parts[0]
            split = parts[1]
            rows.append({"image_name": image_name, "split": split})

    return pd.DataFrame(rows)


def simplify_category(category_id: int) -> str:
    """
    MVP label mapping.
    DeepFashion has 50 category IDs. For tonight, we group them into a smaller
    set of easier-to-explain garment classes.
    """

    # Approximate grouping for MVP report/demo.
    if category_id in {3, 4, 5, 6, 7, 8, 11, 17, 18, 19, 20, 21}:
        return "top"
    if category_id in {16, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}:
        return "dress"
    if category_id in {41, 42, 43}:
        return "skirt"
    if category_id in {23, 24, 25, 26, 27, 28}:
        return "pants_or_shorts"
    return "other"


def main():
    category_df = read_category_file(CATEGORY_FILE)
    split_df = read_split_file(SPLIT_FILE)

    df = category_df.merge(split_df, on="image_name", how="inner")

    df["image_path"] = df["image_name"].apply(lambda x: str(IMG_DIR / x.replace("img/", "")))
    df["label"] = df["category_id"].apply(simplify_category)

    # Keep only a small MVP set.
    df = df[df["label"].isin(["top", "dress", "skirt", "pants_or_shorts"])].copy()

    # Keep a small balanced subset so training is fast.
    max_per_label_split = 600
    df = (
        df.groupby(["split", "label"], group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max_per_label_split), random_state=42))
        .reset_index(drop=True)
    )

    # Confirm files exist.
    df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    missing = (~df["exists"]).sum()

    if missing > 0:
        print(f"Warning: {missing} image paths do not exist. Check folder structure.")

    df = df[df["exists"]].drop(columns=["exists"])

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"Saved manifest: {OUT_FILE}")
    print()
    print("Rows by split:")
    print(df["split"].value_counts())
    print()
    print("Rows by label:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()