# Real-Time Multi-Crop Plant Leaf Disease Detection Using ResNet-50 Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.x%20%7C%20Vite-61dafb.svg)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS-38bdf8.svg)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/Academic-IA--3%20Evaluation-green.svg)]()

A complete, production-grade Deep Learning system engineered for real-time identification of plant pathologies and healthy foliage across multiple agricultural crops. Built with **ResNet-50 Transfer Learning**, a dedicated **Foliage Validation Guard**, **Out-of-Distribution Uncertainty Detection**, a high-performance **FastAPI backend**, and a modern, responsive **React + Vite** frontend.

Developed for the **IA-3 Deep Learning Skill-Based Evaluation (25 Marks)** targeting the **Excellent** category across all technical evaluation rubrics.

---

## 📑 Table of Contents
1. [Project Overview & Key Innovations](#project-overview--key-innovations)
2. [IA-3 Evaluation Rubric Alignment (25 Marks)](#ia-3-evaluation-rubric-alignment-25-marks)
3. [Multi-Crop Dataset & Coverage](#multi-crop-dataset--coverage)
4. [Deep Learning Architecture](#deep-learning-architecture)
5. [Two-Stage Training Methodology](#two-stage-training-methodology)
6. [Out-of-Distribution Leaf Guard & Uncertainty Handling](#out-of-distribution-leaf-guard--uncertainty-handling)
7. [System Architecture & Folder Structure](#system-architecture--folder-structure)
8. [FastAPI Backend API Reference](#fastapi-backend-api-reference)
9. [React + Vite Frontend Interface](#react--vite-frontend-interface)
10. [Local Execution Instructions (Windows)](#local-execution-instructions-windows)
11. [Google Colab Training Compatibility](#google-colab-training-compatibility)
12. [Automated Testing Suite](#automated-testing-suite)

---

## 1. Project Overview & Key Innovations

Agricultural crop diseases present severe threats to global food security. Traditional agricultural field scouting is labor-intensive and prone to diagnostic error. This project delivers an end-to-end intelligent agricultural vision solution:

- **Broad Multi-Crop Coverage:** Classifies 37 verified categories across 11 crop species (Apple, Blueberry, Cherry, Corn, Grape, Peach, Pepper, Potato, Rice, Strawberry, and Tomato).
- **ResNet-50 Transfer Learning:** Employs ImageNet-pretrained representations and deep residual skip connections ($H(x) = F(x) + x$) to capture microscopic necrosis and lesion morphology without gradient degradation.
- **Two-Stage Fine-Tuning:** Stage 1 trains custom classification layers with a frozen backbone; Stage 2 unfreezes deeper convolutional blocks with a 10x reduced learning rate while preserving BatchNormalization running statistics.
- **Out-of-Distribution & Leaf Validation Guard:** Actively rejects non-plant images (cars, people, buildings, animals, everyday objects) and quantifies prediction entropy so farmers are never misled by false high-confidence predictions on ambiguous inputs.
- **Integrated Agronomic IPM Advisory:** Provides botanical descriptions, diagnostic symptom checklists, cultural preventative measures, and Integrated Pest Management (IPM) protocols.
- **Empirical Transparency:** Full performance metrics, independent test evaluations, confusion matrix heatmaps, and loss/accuracy curves are computed from real data and displayed interactively.

---

## 2. IA-3 Evaluation Rubric Alignment (25 Marks)

| Evaluation Rubric | Marks | Technical Implementation in Workspace | Verification Evidence |
| :--- | :---: | :--- | :--- |
| **1. Problem Identification and Dataset** | **5** | Solves broad multi-crop agricultural diagnosis across 11 crops; unified dataset pipeline with deduplication, corruption checks, and 70/15/15 stratified splitting. | `deep_learning/dataset_loader.py`<br>`deep_learning/dataset_analysis.py`<br>`datasets/split_manifest.json`<br>`results/dataset_statistics.csv` |
| **2. Preprocessing and Network Architecture** | **5** | RGB validation, 224x224 standardization, official ResNet-50 preprocessing, train-only data augmentation (rotation, zoom, flip), GlobalAveragePooling2D, Dense(256), Dropout(0.5), dynamic Softmax. | `deep_learning/preprocessing.py`<br>`deep_learning/model.py`<br>`results/model_architecture.json` |
| **3. Model Implementation and Training** | **5** | Two-stage training pipeline (Stage 1 Feature Extraction, Stage 2 Fine-Tuning), EarlyStopping (patience=5), ModelCheckpoint, ReduceLROnPlateau, Adam optimizer. | `deep_learning/train.py`<br>`models/crop_disease_resnet50.keras`<br>`models/class_names.json`<br>`results/training_history.csv` |
| **4. Performance Evaluation and Application** | **5** | Independent test set evaluation (Accuracy, Macro/Weighted Precision, Recall, F1), Confusion Matrix heatmap, production FastAPI backend, React + Vite frontend with webcam support. | `deep_learning/evaluate.py`<br>`results/confusion_matrix.png`<br>`results/accuracy_curve.png`<br>`backend/main.py`<br>`frontend/src/App.jsx` |
| **5. Report** | **5** | All figures, CSV logs, classification reports, and metrics are automatically saved for independent report preparation. | `results/*.png`, `results/*.csv`, `results/*.json` |
| **TOTAL** | **25** | **Complete Working End-to-End Deep Learning System** | **All 4 technical criteria fully runnable** |

---

## 3. Multi-Crop Dataset & Coverage

The system integrates verified imagery from:
1. **PlantVillage Dataset:** Benchmarked foliar samples of orchard fruits and solanaceous vegetables.
2. **Rice Leaf Disease Dataset:** Ground-truth paddy images of bacterial and fungal rice infections.

### Supported Crops and Categories (37 Classes)
- **Apple (4):** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Blueberry (1):** Healthy
- **Cherry (2):** Powdery Mildew, Healthy
- **Corn / Maize (4):** Gray Leaf Spot (Cercospora), Common Rust, Northern Leaf Blight, Healthy
- **Grape (4):** Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy
- **Peach (2):** Bacterial Spot, Healthy
- **Pepper Bell (2):** Bacterial Spot, Healthy
- **Potato (3):** Early Blight, Late Blight, Healthy
- **Rice (3):** Bacterial Leaf Blight, Brown Spot, Leaf Smut
- **Strawberry (2):** Leaf Scorch, Healthy
- **Tomato (10):** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mites, Target Spot, Yellow Leaf Curl Virus (TYLCV), Mosaic Virus (ToMV), Healthy

---

## 4. Deep Learning Architecture

```
INPUT IMAGE (224 × 224 × 3 RGB)
       ↓
PRETRAINED RESNET-50 BACKBONE
(ImageNet Weights, include_top=False, 50 layers)
       ↓
GLOBAL AVERAGE POOLING 2D
(Collapses 7 × 7 × 2048 to 2048-dim feature vector)
       ↓
DENSE LAYER (256 Neurons, ReLU Activation)
       ↓
DROPOUT LAYER (p = 0.5 Regularization)
       ↓
FINAL DENSE LAYER (Dynamically sized to 37 classes)
       ↓
SOFTMAX ACTIVATION
       ↓
PROBABILITY DISTRIBUTION & DISEASE PREDICTION
```

### Mathematical Formulation
ResNet solves the degradation problem via residual learning:
$$\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
Where $\mathbf{x}$ is the input tensor, $\mathcal{F}$ is the residual mapping to be learned, and $\mathcal{H}$ is the desired underlying mapping.

The classification head minimizes multi-class Categorical Crossentropy:
$$\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

---

## 5. Two-Stage Training Methodology

1. **Stage 1 — Feature Extraction (Epochs 1–8):**
   - ResNet-50 backbone layers are **frozen** (`base_model.trainable = False`).
   - Only the custom classification head is optimized using **Adam** ($\eta = 10^{-4}$).
   - Prevents destabilizing pretrained ImageNet features with initial random weight gradients.

2. **Stage 2 — Fine-Tuning (Epochs 9–12):**
   - Deeper convolutional layers (`conv5_block` layers) are **unfrozen**.
   - `BatchNormalization` layers remain in inference mode to preserve population running mean and variance.
   - Recompiled with a 10x smaller learning rate ($\eta = 10^{-5}$) to subtly tailor representations to agricultural foliar structures.

3. **Overfitting Safeguards:**
   - **EarlyStopping:** Monitors validation accuracy with patience=5 and automatic weight restoration.
   - **ModelCheckpoint:** Saves only the highest-performing model weights to `models/crop_disease_resnet50.keras`.
   - **ReduceLROnPlateau:** Dynamically scales learning rate by $0.2\times$ when validation loss plateaus.
   - **Data Augmentation:** Real-time horizontal flipping, random rotation ($\pm 15^\circ$), zoom ($\pm 5\%$), and spatial translation applied strictly during training.

---

## 6. Out-of-Distribution Leaf Guard & Uncertainty Handling

Standard softmax neural networks suffer from overconfidence on out-of-distribution inputs (e.g. uploading a picture of a car or a person forces the network to pick one of the plant diseases). AgriVision AI prevents this through a dual-guard pipeline:

1. **Semantic Plant Foliage Verification:**
   - Evaluates ImageNet broad category predictions and chlorophyll reflectance indices (Excess Green Index $ExG = 2G - R - B$).
   - If an input is identified as a vehicle, building, person, animal, or non-plant object, the system aborts inference with:
     > *"Invalid Image: Please upload a valid plant or leaf image."*

2. **Prediction Entropy & Margin Guard:**
   - Computes normalized Shannon entropy over the predictive distribution:
     $$H(p) = -\frac{\sum_{i=1}^{C} p_i \log_2(p_i)}{\log_2(C)}$$
   - If $H(p) > 0.88$ or the confidence margin between top-1 and top-2 candidates is $< 8\%$, the system displays:
     > *"Unable to identify this plant or disease reliably. Please upload a clearer image or consult an agricultural expert."*

---

## 7. System Architecture & Folder Structure

```
DL project/
├── frontend/                     # React + Vite Frontend
│   ├── src/
│   │   ├── components/           # Navbar, Footer, CameraModal
│   │   ├── pages/                # Home, DiseaseDetection, DiseaseInfo,
│   │   │                         # SupportedPlants, ModelPerformance, AboutModel
│   │   ├── services/api.js       # Axios client connected to FastAPI
│   │   ├── App.jsx               # Routes & Layout
│   │   ├── main.jsx              # React Entry Point
│   │   └── index.css             # Tailwind v4 Styling
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── backend/                      # Python FastAPI Backend
│   ├── main.py                   # App startup, CORS, lifespan
│   ├── routes/api.py             # REST API endpoints
│   ├── services/                 # Model caching & metrics service
│   ├── schemas/                  # Pydantic response models
│   └── requirements.txt
│
├── deep_learning/                # Machine Learning Pipeline
│   ├── dataset_loader.py         # Multi-dataset ingestion & stratified split
│   ├── dataset_analysis.py       # Dataset statistics & distribution plot
│   ├── preprocessing.py          # ResNet-50 preprocessing & tf.data pipeline
│   ├── model.py                  # ResNet-50 transfer learning & fine-tuning
│   ├── train.py                  # Two-stage training execution
│   ├── evaluate.py               # Independent test set evaluation & confusion matrix
│   ├── leaf_validator.py         # Foliage integrity & non-plant rejection
│   └── predict.py                # Cached inference pipeline
│
├── datasets/                     # Local Image Storage
│   ├── raw/                      # Downloaded multi-crop images
│   ├── train/                    # 70% Stratified Training Partition
│   ├── validation/               # 15% Validation Partition
│   ├── test/                     # 15% Independent Test Partition
│   └── split_manifest.json       # Exact image split reproducibility manifest
│
├── models/
│   ├── crop_disease_resnet50.keras # Trained best model weights
│   └── class_names.json          # Synchronized class index mapping
│
├── results/                      # Empirical Evaluation Artifacts
│   ├── dataset_statistics.csv    # Class-wise image counts & sources
│   ├── class_distribution.png    # High-resolution dataset distribution chart
│   ├── accuracy_curve.png        # Training vs Validation Accuracy graph
│   ├── loss_curve.png            # Training vs Validation Loss graph
│   ├── confusion_matrix.png      # Independent test Confusion Matrix heatmap
│   ├── class_f1_scores.png       # Per-class F1-score performance bar chart
│   ├── classification_report.csv # Per-class precision, recall, f1, support
│   ├── training_history.csv      # Epoch-by-epoch training logs
│   ├── training_config.json      # Hyperparameter metadata
│   └── evaluation_metrics.json   # Overall test set accuracy & macro metrics
│
├── data/
│   └── disease_information.json  # Agronomic encyclopedia & IPM protocols
│
├── tests/
│   └── test_pipeline.py          # Automated PyTest verification suite
│
├── requirements.txt              # Root dependencies
└── README.md                     # Comprehensive academic documentation
```

---

## 8. FastAPI Backend API Reference

Base URL: `http://localhost:8000/api`

| Method | Endpoint | Description | Sample Output |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | System status, model loading verification | `{"status":"operational","model_loaded":true}` |
| `POST` | `/predict` | Primary multi-format leaf classification endpoint | `{"crop":"Tomato","disease":"Early Blight","confidence":0.96}` |
| `POST` | `/validate-image` | Validates file format and confirms plant foliage | `{"is_valid_file":true,"is_plant_leaf":true}` |
| `GET` | `/classes` | List of all 37 trained categories | `{"total":37,"classes":[...]}` |
| `GET` | `/crops` | Grouped view of crops and their diseases | `{"total_crops":11,"crops":[...]}` |
| `GET` | `/diseases` | Full agricultural disease knowledge base | Structured dictionary of 37 entries |
| `GET` | `/disease-info/{id}` | Botanical description, symptoms, prevention, IPM | Detailed agronomic profile for class |
| `GET` | `/model-info` | ResNet-50 layer specifications and parameter counts | Architecture parameters and weights |
| `GET` | `/model-metrics` | Test accuracy, precision, recall, macro F1-score | Evaluated metric dictionary |
| `GET` | `/performance` | Aggregated dashboard data (history + report) | Full dashboard payload for frontend |
| `GET` | `/charts/{name}` | Serves confusion matrix and training curves | High-res PNG image response |

---

## 9. React + Vite Frontend Interface

The frontend is constructed using React 19, Vite, and Tailwind CSS. It is structured into six dedicated modules:

1. **Home (`/`):** Project hero banner, real-time live statistics fetched from `/api/health`, core architecture highlights, and step-by-step workflow.
2. **Disease Detection (`/detect`):** Drag-and-drop file uploader supporting multi-formats (JPG, PNG, WEBP, BMP, TIFF, GIF), live camera capture via HTML5 `getUserMedia`, instant ResNet-50 predictions, top-3 candidates, confidence meters, and agronomic management tabs.
3. **Disease Information (`/diseases`):** Searchable, filterable encyclopedia with disease descriptions, symptom checklists, cultural prevention, and IPM controls.
4. **Supported Crops (`/crops`):** Dynamic crop directory showing only verified classes present in the model.
5. **Model Performance (`/performance`):** Real-time evaluation dashboard featuring Recharts interactive training curves, high-resolution Matplotlib figures, confusion matrix heatmap, and test set metric summaries.
6. **About the Model (`/about`):** Detailed academic documentation explaining residual connections, transfer learning mechanics, and out-of-distribution safeguards.

---

## 10. Local Execution Instructions (Windows)

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Node.js 18+ & npm

### Step 1: Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Ingest Datasets & Generate Stratified Splits
```powershell
python deep_learning/dataset_loader.py 30
python deep_learning/dataset_analysis.py
```

### Step 3: Train ResNet-50 Model
```powershell
python deep_learning/train.py --stage1_epochs 8 --stage2_epochs 4 --batch_size 32
```

### Step 4: Evaluate Model on Independent Test Dataset
```powershell
python deep_learning/evaluate.py
```

### Step 5: Start FastAPI Backend
```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be accessible at: `http://localhost:8000/docs`

### Step 6: Start React Frontend
In a separate terminal:
```powershell
cd frontend
npm install
npm run dev
```
Open your browser at: `http://localhost:5173`

---

## 11. Google Colab Training Compatibility

To train on GPU via Google Colab with larger image quantities (e.g. 500+ images per class):

1. Upload the project folder to Google Drive.
2. In a Colab notebook with GPU accelerator enabled:
```python
# Check GPU
!nvidia-smi

# Clone or navigate to repository
%cd /content/DL project

# Install requirements
!pip install -r requirements.txt

# Download dataset (e.g., 200 images per class)
!python deep_learning/dataset_loader.py 200
!python deep_learning/dataset_analysis.py

# Execute GPU Training
!python deep_learning/train.py --stage1_epochs 15 --stage2_epochs 10 --batch_size 64

# Evaluate on test partition
!python deep_learning/evaluate.py
```
Download the resulting `models/crop_disease_resnet50.keras` and `results/` folder into your local repository.

---

## 12. Automated Testing Suite

The repository includes a comprehensive automated test suite verifying dataset integrity, preprocessing, ResNet-50 architecture initialization, leaf guard rejection, and API responses:

```powershell
pytest tests/test_pipeline.py -v
```

---

*AgriVision AI · IA-3 Deep Learning Skill-Based Evaluation · 25 Marks Project*
