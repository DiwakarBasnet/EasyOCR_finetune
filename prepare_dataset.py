import argparse
import random
import shutil
from pathlib import Path
from csv import writer
import yaml

from train_easyocr.utils import AttrDict


def get_arguments():
    parser = argparse.ArgumentParser(description="prepare_dataset for single-character EasyOCR finetuning")
    parser.add_argument("--dataset", required=True,
                        help="Path to DevanagariHandwrittenCharacterDataset root")
    parser.add_argument("--training", action="store_true",
                        help="Generate training set")
    parser.add_argument("--validation", action="store_true",
                        help="Generate validation set")
    parser.add_argument("--evaluation", action="store_true",
                        help="Generate evaluation set")
    return parser.parse_args()


def collect_and_copy(images, dst_dir: Path, labels_csv: Path):
    """
    Copy images list to dst_dir/images and write labels.csv with filename,label
    """
    img_out_dir = dst_dir / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # Write header
    with open(labels_csv, 'w', newline='', encoding='utf8') as f:
        csv_w = writer(f)
        csv_w.writerow(["filename", "label"])

    for img_path, label in images:
        dst_name = img_path.name
        dst_path = img_out_dir / dst_name
        shutil.copy(str(img_path), str(dst_path))
        with open(labels_csv, 'a', newline='', encoding='utf8') as f:
            csv_w = writer(f)
            csv_w.writerow([dst_name, label])


def main():
    args = get_arguments()
    ds_root = Path(args.dataset)

    # load config
    cfg_path = Path(__file__).parent / "train_easyocr/config_files/config.yaml"
    config = AttrDict(yaml.safe_load(cfg_path.open('r', encoding='utf8')))
    random.seed(config.seed)

    # gather all (image_path, label) tuples
    train_folder = ds_root / 'Train'
    test_folder = ds_root / 'Test'

    all_train = []
    for class_dir in train_folder.iterdir():
        if class_dir.is_dir():
            for img in class_dir.glob('*.png'):
                all_train.append((img, class_dir.name))

    all_test = []
    for class_dir in test_folder.iterdir():
        if class_dir.is_dir():
            for img in class_dir.glob('*.png'):
                all_test.append((img, class_dir.name))

    # sample sets
    train_sel = random.sample(all_train, k=min(config.train_images, len(all_train)))
    val_sel   = random.sample(all_test,  k=min(config.val_images,   len(all_test)))
    # for evaluation, sample from remaining test
    remaining = list(set(all_test) - set(val_sel))
    eval_sel  = random.sample(remaining, k=min(config.eval_images, len(remaining)))

    out_base = Path.cwd() / 'training_and_validation_set'
    eval_base = Path.cwd() / 'evaluation_set'

    if args.training:
        print("→ Generating training set")
        train_dst = out_base / 'training'
        train_dst.mkdir(parents=True, exist_ok=True)
        collect_and_copy(train_sel, train_dst, train_dst / 'labels.csv')
        print(f"✔ Training set ready at {train_dst}")

    if args.validation:
        print("→ Generating validation set")
        val_dst = out_base / 'validation'
        val_dst.mkdir(parents=True, exist_ok=True)
        collect_and_copy(val_sel, val_dst, val_dst / 'labels.csv')
        print(f"✔ Validation set ready at {val_dst}")

    if args.evaluation:
        print("→ Generating evaluation set")
        eval_dst = eval_base
        # copy images only (no labels.csv)
        img_out = eval_dst / 'images'
        img_out.mkdir(parents=True, exist_ok=True)
        for img_path, _ in eval_sel:
            dst = img_out / img_path.name
            shutil.copy(str(img_path), str(dst))
        print(f"✔ Evaluation images ready at {eval_dst}")

if __name__ == '__main__':
    main()
