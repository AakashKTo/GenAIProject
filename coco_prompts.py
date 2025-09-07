import os
import zipfile
import urllib.request
import json
from tqdm import tqdm
from PIL import Image
import pandas as pd

# URLs
COCO_IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Paths
IMG_ZIP = "val2017.zip"
ANN_ZIP = "annotations_trainval2017.zip"
IMG_DIR = "val2017"
ANN_DIR = "annotations"
REAL_IMG_DIR = "real_images"
CSV_OUTPUT = "coco_val2017_prompts.csv"

# Download utility
def download(url, path):
    if os.path.exists(path):
        print(f"✅ {path} already exists.")
        return
    print(f"⬇️ Downloading {path}...")
    with urllib.request.urlopen(url) as response, open(path, 'wb') as out_file:
        total = int(response.getheader('Content-Length').strip())
        with tqdm(total=total, unit='B', unit_scale=True, desc=f"Downloading {path}") as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

# Unzip utility
def unzip(zip_path, target_dir="."):
    print(f"📦 Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

# Resize images and convert to PNG
def resize_and_convert_images(src_dir, dest_dir, size=(512, 512)):
    os.makedirs(dest_dir, exist_ok=True)
    for fname in tqdm(os.listdir(src_dir), desc="🖼️ Resizing Images"):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(src_dir, fname)
        out_path = os.path.join(dest_dir, fname.replace(".jpg", ".png"))
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size, Image.BICUBIC)
            img.save(out_path)
        except Exception as e:
            print(f"⚠️ Failed to process {fname}: {e}")

# Parse COCO annotations and map captions to image file paths
def build_dataframe(ann_file, img_dir):
    with open(ann_file, "r") as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"].replace(".jpg", ".png") for img in data["images"]}
    rows = []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        caption = ann["caption"].strip()
        if img_id in images:
            rows.append({
                "image_id": img_id,
                "file_name": os.path.join(img_dir, images[img_id]),
                "caption": caption
            })

    df = pd.DataFrame(rows)
    return df

# Main script
def main():
    # Download files
    download(COCO_IMG_URL, IMG_ZIP)
    download(COCO_ANN_URL, ANN_ZIP)

    # Unzip if necessary
    if not os.path.exists(IMG_DIR):
        unzip(IMG_ZIP)
    if not os.path.exists(ANN_DIR):
        unzip(ANN_ZIP)

    # Resize and save real images as .png
    resize_and_convert_images(IMG_DIR, REAL_IMG_DIR)

    # Build dataframe from annotations
    captions_path = os.path.join(ANN_DIR, "captions_val2017.json")
    df = build_dataframe(captions_path, REAL_IMG_DIR)

    # Save to CSV
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"\n✅ Dataset saved to {CSV_OUTPUT}")
    print(df.head())

if __name__ == "__main__":
    main()
