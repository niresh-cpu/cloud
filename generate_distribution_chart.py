"""
Generates high-resolution horizontal bar chart of the actual dataset distribution.
Requirements:
1. Read dataset folders and count images in each disease class
2. Disease class names on Y-axis
3. Number of images on X-axis
4. All actual classes in the dataset included
5. Green bars with readable labels
6. Image count at the end of each bar
7. Export as high-resolution PNG
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATASET_RAW_DIR = BASE_DIR / "datasets" / "raw"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Read dataset folders directly from disk
class_counts = {}
for class_folder in sorted(DATASET_RAW_DIR.iterdir()):
    if not class_folder.is_dir():
        continue
    # Count verified image files
    imgs = [f for f in class_folder.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    if imgs:
        class_counts[class_folder.name] = len(imgs)

# Format readable labels: e.g. "Apple___Apple_scab" -> "Apple · Apple Scab"
def format_class_label(raw_name):
    parts = raw_name.split("___")
    crop = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    # Capitalize disease nicely
    if disease.lower() == "healthy":
        disease = "Healthy"
    return f"{crop} · {disease}"

# Sort classes by count descending for clean visualization
sorted_items = sorted(class_counts.items(), key=lambda x: (x[1], x[0]), reverse=False)
raw_names = [item[0] for item in sorted_items]
formatted_labels = [format_class_label(item[0]) for item in sorted_items]
counts = [item[1] for item in sorted_items]

total_classes = len(class_counts)
total_images = sum(counts)

print(f"Total Verified Categories: {total_classes}")
print(f"Total Dataset Images: {total_images}")

# 2. Setup figure
plt.figure(figsize=(14, 16), dpi=300)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# 5. Use vibrant green bars with subtle border
bar_color = "#16a34a"  # Emerald / Forest Green
edge_color = "#14532d"  # Dark green outline

bars = plt.barh(
    range(len(formatted_labels)),
    counts,
    color=bar_color,
    edgecolor=edge_color,
    linewidth=0.8,
    height=0.72,
    alpha=0.92
)

# Y-axis ticks and labels
plt.yticks(range(len(formatted_labels)), formatted_labels, fontsize=10, fontweight="medium", color="#1e293b")
plt.ylabel("Crop & Disease Pathological Classification", fontsize=12, fontweight="bold", labelpad=12, color="#0f172a")

# X-axis
max_count = max(counts)
plt.xlim(0, max_count * 1.14)
plt.xlabel("Number of Verified Leaf Sample Images", fontsize=12, fontweight="bold", labelpad=10, color="#0f172a")

# 6. Display exact image count at the end of each bar
for idx, bar in enumerate(bars):
    w = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    plt.text(
        w + (max_count * 0.012),
        y,
        f"{int(w):,}",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#15803d"
    )

# Title & subtitle
plt.title(
    f"Multi-Crop Plant Disease Dataset Class Distribution\n"
    f"Total Categories: {total_classes} Classes  |  Total Ground-Truth Samples: {total_images:,} Images",
    fontsize=14,
    fontweight="bold",
    pad=16,
    color="#0f172a"
)

# Clean grid
plt.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e1")
plt.gca().set_axisbelow(True)

# Spines
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    plt.gca().spines[spine].set_color("#94a3b8")

plt.tight_layout()

# 7. Export high-resolution PNG
out_png = RESULTS_DIR / "dataset_class_distribution_green.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"Exported high-resolution chart to: {out_png}")
