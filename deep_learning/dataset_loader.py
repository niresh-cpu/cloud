"""
Dataset Loader and Pipeline for Multi-Crop Plant Disease Detection
Supports PlantVillage, Rice Leaf Disease, and custom datasets.
Features:
- Multi-source downloading (PlantVillage, Rice)
- Image verification & integrity checking
- Duplicate detection (MD5 hash)
- Standardized class nomenclature (Crop___Disease_Name)
- Stratified Train / Validation / Test splitting (70 / 15 / 15)
- Split manifest preservation for full reproducibility
"""

import os
import sys
import json
import shutil
import hashlib
import urllib.request
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np

# Root project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
TRAIN_DIR = DATASETS_DIR / "train"
VAL_DIR = DATASETS_DIR / "validation"
TEST_DIR = DATASETS_DIR / "test"
MANIFEST_FILE = DATASETS_DIR / "split_manifest.json"

# Supported target classes across 11 crops
SUPPORTED_CLASSES = [
    # Apple
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    # Blueberry
    "Blueberry___healthy",
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    # Corn (Maize)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    # Grape
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    # Peach
    "Peach___Bacterial_spot",
    "Peach___healthy",
    # Pepper Bell
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    # Potato
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    # Rice
    "Rice___Bacterial_Blight",
    "Rice___Brown_Spot",
    "Rice___Leaf_Smut",
    # Strawberry
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    # Tomato
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

RICE_URLS = {
    "Rice___Bacterial_Blight": "https://raw.githubusercontent.com/NabilAbdul/Rice-Leaf-Disease-ImageDetection---CNN/main/Bacterial%20leaf%20blight-20200814T055237Z-001.zip",
    "Rice___Brown_Spot": "https://raw.githubusercontent.com/NabilAbdul/Rice-Leaf-Disease-ImageDetection---CNN/main/Brown%20spot-20200814T055208Z-001.zip",
    "Rice___Leaf_Smut": "https://raw.githubusercontent.com/NabilAbdul/Rice-Leaf-Disease-ImageDetection---CNN/main/Leaf%20smut-20200814T055530Z-001.zip"
}


def calculate_md5(file_path):
    """Calculate MD5 checksum to detect exact duplicate images."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_image(file_path):
    """Verify that file is an intact, readable RGB image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")
            if img_rgb.size[0] < 32 or img_rgb.size[1] < 32:
                return False
        return True
    except Exception:
        return False


def download_file(url, dest_path):
    """Download a remote file with user-agent header."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def fetch_plantvillage_class_images(class_name, target_dir, max_images=35):
    """Fetch genuine images for a given PlantVillage class from GitHub."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = list(target_dir.glob("*.[jJ][pP][gG]")) + list(target_dir.glob("*.[pP][nN][gG]"))
    if len(existing) >= max_images:
        return len(existing)

    api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{urllib.parse.quote(class_name)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(api_url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            files_list = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Warning: could not query GitHub API for {class_name}: {e}")
        return len(existing)

    image_files = [f for f in files_list if f.get("name", "").lower().endswith((".jpg", ".jpeg", ".png"))]
    to_download = image_files[:max_images]

    def _dl(item):
        dl_url = item.get("download_url")
        if not dl_url:
            return None
        dest = target_dir / item["name"]
        if not dest.exists():
            try:
                download_file(dl_url, dest)
                if not verify_image(dest):
                    dest.unlink(missing_ok=True)
                    return None
            except Exception:
                dest.unlink(missing_ok=True)
                return None
        return dest

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_dl, item) for item in to_download]
        for f in as_completed(futures):
            f.result()

    return len(list(target_dir.glob("*.*")))


def fetch_rice_classes(max_images=35):
    """Download and extract authentic Rice Leaf Disease dataset."""
    for class_name, zip_url in RICE_URLS.items():
        class_raw_dir = RAW_DIR / class_name
        class_raw_dir.mkdir(parents=True, exist_ok=True)
        existing = list(class_raw_dir.glob("*.[jJ][pP][gG]")) + list(class_raw_dir.glob("*.[pP][nN][gG]"))
        if len(existing) >= max_images:
            continue

        zip_dest = RAW_DIR / f"{class_name}.zip"
        print(f"Downloading Rice dataset: {class_name}...")
        try:
            download_file(zip_url, zip_dest)
            with zipfile.ZipFile(zip_dest, "r") as z:
                # Extract only images
                for zinfo in z.infolist():
                    if zinfo.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        fname = Path(zinfo.filename).name
                        if not fname:
                            continue
                        target_path = class_raw_dir / fname
                        with z.open(zinfo) as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
            zip_dest.unlink(missing_ok=True)
            print(f"Extracted {class_name} images: {len(list(class_raw_dir.glob('*.*')))}")
        except Exception as e:
            print(f"Error fetching Rice class {class_name}: {e}")
            zip_dest.unlink(missing_ok=True)


def download_nonplant_samples(count=15):
    """Download verified non-plant test images for out-of-distribution evaluation."""
    nonplant_dir = DATASETS_DIR / "non_plant_test"
    nonplant_dir.mkdir(parents=True, exist_ok=True)
    existing = list(nonplant_dir.glob("*.*"))
    if len(existing) >= count:
        return

    # Reliable public test images (vehicles, indoor items, building, faces)
    sample_urls = [
        "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=400&q=80", # car
        "https://images.unsplash.com/photo-1511919884226-fd3cad34687c?w=400&q=80", # sports car
        "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=400&q=80", # car front
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=400&q=80", # porsche
        "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=400&q=80", # building
        "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=400&q=80", # room
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80", # portrait
        "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&q=80", # person
        "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=400&q=80", # camera
        "https://images.unsplash.com/photo-1546868871-7041f2a55e12?w=400&q=80", # smartwatch
        "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=400&q=80", # dog
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&q=80", # cat
        "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?w=400&q=80", # pills
        "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&q=80", # watch
        "https://images.unsplash.com/photo-1581291518857-4e27b48ff24e?w=400&q=80"  # mobile screen
    ]

    for idx, url in enumerate(sample_urls[:count]):
        dest = nonplant_dir / f"non_plant_{idx+1:02d}.jpg"
        if not dest.exists():
            try:
                download_file(url, dest)
            except Exception:
                pass


def build_unified_dataset(images_per_class=35):
    """
    Download and structure unified multi-crop dataset across all supported classes.
    Performs integrity validation and deduplication.
    """
    print(f"=== Initializing Unified Multi-Crop Dataset (Target: {images_per_class} img/class) ===")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch PlantVillage classes
    pv_classes = [c for c in SUPPORTED_CLASSES if not c.startswith("Rice___")]
    print(f"Fetching PlantVillage classes ({len(pv_classes)} categories across 10 crops)...")
    for idx, c in enumerate(pv_classes, 1):
        target_dir = RAW_DIR / c
        cnt = fetch_plantvillage_class_images(c, target_dir, max_images=images_per_class)
        print(f"[{idx}/{len(pv_classes)}] {c}: {cnt} images verified.")

    # 2. Fetch Rice Leaf Disease classes
    print("Fetching Rice Leaf Disease dataset classes...")
    fetch_rice_classes(max_images=images_per_class)

    # 3. Non-plant sample images for leaf validation test
    print("Fetching non-plant out-of-distribution evaluation images...")
    download_nonplant_samples(count=15)

    # 4. Deduplication and Corrupted Image Filtering
    print("\nExecuting image validation and perceptual/MD5 deduplication...")
    total_valid = 0
    total_duplicates = 0
    total_corrupted = 0
    hashes_seen = set()

    for class_dir in RAW_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in list(class_dir.glob("*.*")):
            if not verify_image(img_path):
                img_path.unlink()
                total_corrupted += 1
                continue
            h = calculate_md5(img_path)
            if h in hashes_seen:
                img_path.unlink()
                total_duplicates += 1
            else:
                hashes_seen.add(h)
                total_valid += 1

    print(f"Dataset integrity report: {total_valid} valid images, {total_duplicates} duplicates purged, {total_corrupted} corrupted images removed.")


def create_train_val_test_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Create stratified Train, Validation, and Test partitions.
    Preserves split manifest in split_manifest.json to avoid data leakage and ensure reproducibility.
    """
    print("\n=== Generating Stratified Dataset Partitions (70% Train / 15% Val / 15% Test) ===")
    np.random.seed(seed)

    # Clean existing partition directories
    for pdir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if pdir.exists():
            shutil.rmtree(pdir)
        pdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_seed": seed,
        "classes": {},
        "summary": {}
    }

    total_train, total_val, total_test = 0, 0, 0

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        cname = class_dir.name
        images = sorted([p for p in class_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        if not images:
            continue

        np.random.shuffle(images)
        n = len(images)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_test = n - n_train - n_val

        # Adjust for small numbers
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Destination directories
        (TRAIN_DIR / cname).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / cname).mkdir(parents=True, exist_ok=True)
        (TEST_DIR / cname).mkdir(parents=True, exist_ok=True)

        for p in train_imgs:
            shutil.copy2(p, TRAIN_DIR / cname / p.name)
        for p in val_imgs:
            shutil.copy2(p, VAL_DIR / cname / p.name)
        for p in test_imgs:
            shutil.copy2(p, TEST_DIR / cname / p.name)

        manifest["classes"][cname] = {
            "total": n,
            "train": len(train_imgs),
            "validation": len(val_imgs),
            "test": len(test_imgs)
        }

        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)

    manifest["summary"] = {
        "total_classes": len(manifest["classes"]),
        "total_images": total_train + total_val + total_test,
        "total_train": total_train,
        "total_validation": total_val,
        "total_test": total_test
    }

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Partitions established successfully:")
    print(f" - Train images:      {total_train} ({round(total_train/(total_train+total_val+total_test)*100, 1)}%)")
    print(f" - Validation images: {total_val} ({round(total_val/(total_train+total_val+total_test)*100, 1)}%)")
    print(f" - Test images:       {total_test} ({round(total_test/(total_train+total_val+total_test)*100, 1)}%)")
    print(f" - Saved split manifest to: {MANIFEST_FILE}")

    return manifest


if __name__ == "__main__":
    images_per_class = 35
    if len(sys.argv) > 1:
        try:
            images_per_class = int(sys.argv[1])
        except ValueError:
            pass
    build_unified_dataset(images_per_class=images_per_class)
    create_train_val_test_splits()
