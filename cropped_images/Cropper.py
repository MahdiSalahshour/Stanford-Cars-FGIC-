#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import cv2
from tqdm import tqdm
from collections import Counter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Inspect and crop Stanford Car dataset")
    p.add_argument(
        "--root-dir", 
        type=str, 
        required=True,
        help="Path to `stanford-car-dataset-by-classes-folder` root"
    )
    p.add_argument(
        "--output-dir", 
        type=str, 
        default="cropped_images",
        help="Where to save the cropped images"
    )
    p.add_argument(
        "--show-classes", 
        type=int, 
        default=20,
        help="How many classes to visualize"
    )
    return p.parse_args()

def inspect_dataset(train_dir, num_show=20):
    # build torchvision dataset just to inspect
    ds = datasets.ImageFolder(train_dir)
    print(f"Found {len(ds.classes)} classes, {len(ds.samples)} images total.")
    
    counts = Counter(label for _, label in ds.samples)
    for idx, cls in enumerate(ds.classes[:num_show]):
        print(f"{idx:3d}: {cls:30s} → {counts[idx]:4d} images")
    
    # simple grid of first image from each of the first num_show classes
    transform = transforms.Compose([transforms.ToTensor()])
    fig, axs = plt.subplots(
        num_show//5 + (num_show%5>0),
        5,
        figsize=(15, 3*(num_show//5 + 1))
    )
    axs = axs.flatten()
    for i, cls in enumerate(ds.classes[:num_show]):
        path = os.path.join(train_dir, cls, os.listdir(os.path.join(train_dir, cls))[0])
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].set_title(cls, fontsize=8)
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()

def load_metadata(csv_path):
    cols = ['filename','x1','y1','x2','y2','class_id']
    return pd.read_csv(csv_path, header=None, names=cols)

def load_names(names_csv):
    df = pd.read_csv(names_csv, header=None)
    id_to_name = {
        i+1: row.replace("/", "-")
        for i, row in enumerate(df.iloc[:,0])
    }
    return id_to_name

def crop_and_save(df, id_to_name, src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)
    print(f"Cropping {len(df)} images…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping"):
        cls_id = int(row['class_id'])
        class_folder = id_to_name[cls_id]
        src_path = os.path.join(src_root, class_folder, row['filename'])
        dst_path = os.path.join(dst_root, class_folder, row['filename'])
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img = cv2.imread(src_path)
        if img is None:
            continue
        x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(dst_path, crop)
    print("Done! Cropped images are in:", dst_root)

def main():
    args = parse_args()

    # paths
    train_dir   = os.path.join(args.root_dir, "car_data", "car_data", "train")
    anno_csv    = os.path.join(args.root_dir, "anno_train.csv")
    names_csv   = os.path.join(args.root_dir, "names.csv")
    out_dir     = args.output_dir

    # inspect
    inspect_dataset(train_dir, num_show=args.show_classes)

    # prepare & crop
    df_meta    = load_metadata(anno_csv)
    id_to_name = load_names(names_csv)
    crop_and_save(df_meta, id_to_name, train_dir, out_dir)

if __name__ == "__main__":
    main()
