"""
Dataset Analysis and Statistics Generator for Real-Time Plant Disease Detection
Generates:
1. results/dataset_statistics.csv with columns:
   - Plant Name
   - Disease Name
   - Number of Images
2. results/class_distribution.png (Horizontal bar chart of actual image counts)
"""

import os
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "raw"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / "dataset_statistics.csv"
CHART_FILE = RESULTS_DIR / "class_distribution.png"

def clean_plant_and_disease(folder_name):
    parts = folder_name.split("___")
    plant_raw = parts[0]
    disease_raw = parts[1] if len(parts) > 1 else "Unknown"

    plant = plant_raw.replace("_", " ").strip()
    if "corn" in plant.lower():
        plant = "Corn (Maize)"
    elif "cherry" in plant.lower():
        plant = "Cherry (including sour)"
    elif "pepper" in plant.lower():
        plant = "Pepper, Bell"

    disease = disease_raw.replace("_", " ").strip()
    if disease.lower() == "healthy":
        disease = "Healthy"
    elif "spider_mite" in disease_raw.lower() or "two-spotted" in disease_raw.lower():
        disease = "Spider Mites (Two-Spotted Spider Mite)"
    elif "gray_leaf_spot" in disease_raw.lower() or "cercospora" in disease_raw.lower():
        disease = "Cercospora Leaf Spot (Gray Leaf Spot)"
    elif "isariopsis" in disease_raw.lower():
        disease = "Leaf Blight (Isariopsis Leaf Spot)"
    elif "black_measles" in disease_raw.lower():
        disease = "Esca (Black Measles)"
    else:
        disease = " ".join(word.capitalize() if not word.isupper() else word for word in disease.split())

    return plant, disease

def analyze_and_export():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {DATASET_DIR}")

    classes = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in {DATASET_DIR}")

    records = []
    for c in classes:
        # Count actual valid image files
        imgs = [f for f in c.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]]
        count = len(imgs)
        plant, disease = clean_plant_and_disease(c.name)
        records.append({
            "plant": plant,
            "disease": disease,
            "raw_name": c.name,
            "count": count
        })

    # 1. Export dataset_statistics.csv
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Plant Name", "Disease Name", "Number of Images"])
        for r in records:
            writer.writerow([r["plant"], r["disease"], r["count"]])

    print(f"Generated {CSV_FILE} successfully with {len(records)} classes.")

    # 2. Generate Horizontal Bar Chart (class_distribution.png)
    # Sort classes by count ascending for horizontal bar chart (highest at top)
    sorted_records = sorted(records, key=lambda x: (x["count"], x["plant"], x["disease"]))

    labels = [f"{r['plant']} - {r['disease']}" for r in sorted_records]
    counts = [r["count"] for r in sorted_records]
    total_images = sum(counts)
    total_classes = len(counts)

    plt.figure(figsize=(14, 16), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Clean emerald green bars with dark borders
    bars = plt.barh(
        range(len(labels)),
        counts,
        color="#16a34a",
        edgecolor="#14532d",
        linewidth=0.8,
        height=0.72,
        alpha=0.92
    )

    # Y-axis
    plt.yticks(range(len(labels)), labels, fontsize=9.5, fontweight="medium", color="#1e293b")
    plt.ylabel("Plant & Disease Classification", fontsize=12, fontweight="bold", labelpad=12, color="#0f172a")

    # X-axis
    max_count = max(counts)
    plt.xlim(0, max_count * 1.14)
    plt.xlabel("Number of Verified Images", fontsize=12, fontweight="bold", labelpad=10, color="#0f172a")

    # Image count at the end of each bar
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        plt.text(
            w + (max_count * 0.012),
            y,
            f"{int(w)}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#15803d"
        )

    # Title
    plt.title(
        f"Multi-Crop Plant Disease Dataset Class Distribution\n"
        f"Total Classes: {total_classes}  |  Total Actual Images: {total_images:,}",
        fontsize=14,
        fontweight="bold",
        pad=16,
        color="#0f172a"
    )

    # Grid & Spines
    plt.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e1")
    plt.gca().set_axisbelow(True)
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        plt.gca().spines[spine].set_color("#94a3b8")

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated {CHART_FILE} successfully (300 DPI).")
    print(f"Total categories: {total_classes}, Total images: {total_images}")

if __name__ == "__main__":
    analyze_and_export()
