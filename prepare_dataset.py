import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import random
import yaml
import shutil
from csv import writer

from train_easyocr.utils import AttrDict
from process_images import (
    load_image_as_array,
    save_image,
    get_image_cropped_by_rectangle
)


def get_arguments():
    parser = argparse.ArgumentParser(description="prepare_dataset")

    parser.add_argument("--dataset", required=True,
                        help="Path to the root Dataset folder")
    parser.add_argument("--training", action="store_true",
                        help="Generate training patches")
    parser.add_argument("--validation", action="store_true",
                        help="Generate validation patches")
    parser.add_argument("--evaluation", action="store_true",
                        help="Copy evaluation set")

    return parser.parse_args()


def parse_json_file(json_path):
    """Load a JSON label file and return a DataFrame of bboxes + text."""
    with open(json_path, "r", encoding="utf8") as f:
        label = json.load(f)

    df = pd.DataFrame(
        [(*ann["annotation.bbox"], ann["annotation.text"])
         for ann in label["annotations"]],
        columns=["xmin", "ymin", "w", "h", "text"]
    )
    # convert w,h → xmax,ymax
    df["xmax"] = df["xmin"] + df["w"]
    df["ymax"] = df["ymin"] + df["h"]
    return df[["xmin", "ymin", "xmax", "ymax", "text"]]


def save_image_patches(output_dir: Path, split: str, file_list, select_data: str):
    print(f"→ Generating image patches for {split} set")
    save_dir = output_dir / split / select_data
    img_out_dir = save_dir / "images"
    save_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = save_dir / "labels.csv"
    pd.DataFrame(columns=["filename", "words"]).to_csv(labels_csv, index=False)

    for img_path in tqdm(file_list):
        # img_path: .../images_train/images/nid_1.png  (or images_val/images/*.png)
        # find its corresponding JSON:
        imgs_parent = img_path.parent           # .../.../images
        group_dir  = imgs_parent.parent         # .../.../images_train
        labels_dir = group_dir.parent / group_dir.name.replace("images", "labels") / "labels"
        json_path  = labels_dir / f"{img_path.stem}.json"
        if not json_path.exists():
            continue

        img = load_image_as_array(str(img_path))
        gt = parse_json_file(json_path)

        for xmin, ymin, xmax, ymax, text in gt.values:
            xmin, ymin = max(0, xmin), max(0, ymin)
            try:
                patch = get_image_cropped_by_rectangle(
                    img=img, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                )
                out_name = f"{img_path.stem}_{xmin}-{ymin}-{xmax}-{ymax}.png"
                out_path = img_out_dir / out_name
                if not out_path.exists():
                    save_image(img=patch, path=str(out_path))
                    with open(labels_csv, "a", newline="", encoding="utf8") as f:
                        writer(f).writerow((out_name, text))
            except Exception:
                print(f"    ✗ Failed to save patch {out_name}")

    print(f"✔ Completed {split} patches\n")


def prepare_evaluation_set(eval_files, dataset_root: Path):
    print("→ Preparing evaluation set")
    # out_root = dataset_root / "evaluation_set"
    out_root = Path("/kaggle/working/evaluation_set")
    for img_path in tqdm(eval_files):
        # copy JSON
        imgs_parent = img_path.parent
        group_dir  = imgs_parent.parent
        labels_dir = group_dir.parent / group_dir.name.replace("images", "labels") / "labels"
        json_path  = labels_dir / f"{img_path.stem}.json"

        # target paths
        rel = img_path.relative_to(dataset_root / "validation")
        target_img = out_root / "images" / rel
        target_json = out_root / "labels" / rel.with_suffix(".json")

        target_img.parent.mkdir(parents=True, exist_ok=True)
        target_json.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, target_img)
        shutil.copy(json_path, target_json)

    print("✔ Evaluation set ready\n")


if __name__ == "__main__":
    args = get_arguments()
    ds_root = Path(args.dataset)

    # load config
    cfg_path = Path(__file__).parent / "train_easyocr/config_files/config.yaml"
    config = AttrDict(yaml.safe_load(cfg_path.open("r", encoding="utf8")))

    random.seed(config.seed)

    # collect all pngs under training/ and validation/
    train_imgs = list((ds_root / "training").rglob("*.png"))
    train_imgs = [p for p in train_imgs if "images_train" in str(p)]
    
    val_imgs   = list((ds_root / "validation").rglob("*.png"))
    val_imgs = [p for p in val_imgs if "images_val" in str(p)]

    print(f"→ Found {len(train_imgs)} training images")
    print(f"→ Found {len(val_imgs)} validation images\n")

    train_set = random.sample(train_imgs, k=config.train_images)
    val_set   = random.sample(val_imgs,   k=config.val_images)
    # eval = remaining from val
    remaining = list(set(val_imgs) - set(val_set))
    eval_set  = random.sample(remaining,   k=config.eval_images)

    # out_base = ds_root.parent / "training_and_validation_set" 
    out_base = Path("/kaggle/working/training_and_validation_set")

    if args.training:
        save_image_patches(out_base, "training", train_set, config.select_data)
    if args.validation:
        save_image_patches(out_base, "validation", val_set,   config.select_data)
    if args.evaluation:
        prepare_evaluation_set(eval_set, ds_root)
