# IMAGENE-ESCC-Survival

IMAGENE (Subproject) - Python Implementation of "A Multimodal Deep Learning Framework for Postoperative Overall Survival Prediction in Esophageal Squamous Cell Carcinoma"

## Project Overview

This project provides a complete workflow for analyzing pathology images, with four main components:

1. **Cell Classification** - Identify and classify different cell types in pathology images
2. **Tissue Classification** - Classify tissue regions using deep learning models
3. **Feature Calculation** - Extract spatial and morphological features from cell distributions
4. **Survival Analysis** - Analyze patient survival data with multiple feature types

## Project Structure

```
public_version/
├── cell_classfication/           # Cell classification module
│   ├── train.py                    # Train cell classification model
│   ├── pred_cell_label.py          # Predict cell labels
│   ├── get_data.py                 # Data loading utilities
│   ├── cell_feature_model.py       # Feature extraction models
│   └── main.py                     # Main training script
│
├── tissue_classfication/           # Tissue classification module
│   ├── config/                     # Configuration files
│   │   ├── swin.py                # Swin Transformer config
│   │   └── gigapath.py           # GigaPath config
│   ├── models/                     # Model definitions
│   │   └── giga_path_patch_encoder.py  # GigaPath patch encoder
│   └── pred_tissue_label.py        # Predict tissue labels
│
├── calculate_features/          # Feature extraction module
│   ├── calculate_region_features.py    # Calculate region-based features
│   └── nearest_neighbor_distance.py   # GPU-accelerated spatial analysis
│
└── survival_analysis/           # Survival analysis module
    ├── main.py                     # Main pipeline entry
    ├── train.py                    # Training and evaluation
    ├── model_surv.py               # Survival models (Cox, RSF, GB)
    ├── load_data.py                # Data loading and preprocessing
    ├── select_survival_features.py  # Feature selection methods
    └── visualize.py                # Visualization utilities
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/OpenGene/IMAGENE-ESCC-Survival.git
cd IMAGENE-ESCC-Survival
```

1. Create conda environment

```bash
conda create -n pathology_analysis python=3.8
conda activate pathology_analysis
```

1. Install dependencies

```bash
pip install -r requirements.txt
```

## Requirements

See [requirements.txt](requirements.txt) for the complete list of dependencies.

## Usage

### 1. Cell Classification

#### Preparation of Training Data

The training dataset should be organized in the following structure:

```
data_root/
├── info.json                          # Dataset metadata (class_map, class_colors_map, id_img_map)
├── fold0/                             # First fold
│   ├── images/                        # Pathology image patches
│   │   ├── 0_14345_7163.png
│   │   ├── 0_14345_7307.png
│   │   └── ...
│   ├── labels/                        # Ground truth labels
│   │   ├── 0_14345_7163.npy          # Contains inst_map and type_map
│   │   ├── 0_14345_7307.npy
│   │   └── ...
│   ├── cell_detection/                # Cell detection results
│   │   ├── cell_detection/           # JSON format cell annotations
│   │   │   ├── 0_14345_7163.json
│   │   │   └── ...
│   │   ├── cell_detection_geojson/    # GeoJSON format for visualization
│   │   │   ├── 0_14345_7163.geojson
│   │   │   └── ...
│   │   └── cell_graph/                # PyTorch graph tensors (optional)
│   │       ├── 0_14345_7163.pt
│   │       └── ...
│   ├── features.npy                   # Cached cell features
│   ├── labels.npy                     # Cached labels
│   ├── image_paths.json               # List of image file names
│   └── cell_count.csv                 # Cell count statistics
├── fold1/                             # Second fold
│   └── ...
└── fold2/                             # Third fold
    └── ...
```

**File Format Details:**

- **Images**: PNG format patches extracted from whole slide images
  - Naming convention: `{slide_id}_{x_coord}_{y_coord}.png`
- **Labels (.npy)**: Dictionary containing:
  - `inst_map`: Instance segmentation map (H×W numpy array)
  - `type_map`: Cell type classification map (H×W numpy array)
- **Cell Detection (.json)**: JSON format with cell annotations including:
  - Cell coordinates
  - Cell boundaries
  - Detection confidence scores
- **Cell Graph (.pt)**: PyTorch tensor format for graph-based features
- **info.json**: Dataset metadata:
  - `class_map`: Mapping from cell type IDs to names
  - `class_colors_map`: RGB colors for visualization
  - `id_img_map`: Mapping from fold IDs to image identifiers

**Data Preparation Steps:**

1. Extract image patches from whole slide images
2. Generate ground truth labels with instance and type maps
3. Run cell detection to generate cell annotations
4. Generate cell graph representations
5. Create info.json with class mappings
6. Organize data into fold directories for cross-validation

#### Training

```bash
cd cell_classfication
python main.py \
    --mode train \
    --data_root path/to/single_cell_dataset \
    --train_folds fold0 fold1 fold2 \
    --val_folds fold3 \
    --model_path path/to/model.pkl \
    --feature_type all \
    --confusion_matrix_path path/to/confusion_matrix.png \
    --roc_curves_path path/to/roc_curves.png
```

**Parameters:**

- `--mode`: Training mode (train/test)
- `--data_root`: Path to dataset root directory
- `--train_folds`: List of training fold names (default: all folds starting with 'fold')
- `--val_folds`: List of validation fold names (default: 20% of training data)
- `--model_path`: Path to save/load model
- `--feature_type`: Type of features to use (`traditional_feature`, `deep_feature`, or `all`)
- `--confusion_matrix_path`: Path to save confusion matrix (optional)
- `--roc_curves_path`: Path to save ROC curves (optional)

**Example with traditional features only:**

```bash
python main.py \
    --mode train \
    --data_root path/to/single_cell_dataset \
    --model_path path/to/traditional_model.pkl \
    --feature_type traditional_feature
```

**Example with deep features only:**

```bash
python main.py \
    --mode train \
    --data_root path/to/single_cell_dataset \
    --model_path path/to/deep_model.pkl \
    --feature_type deep_feature
```

#### Testing

```bash
python main.py \
    --mode test \
    --data_root path/to/single_cell_dataset \
    --val_folds fold3 \
    --model_path path/to/model.pkl \
    --feature_type all \
    --confusion_matrix_path path/to/test_confusion_matrix.png \
    --roc_curves_path path/to/test_roc_curves.png
```

#### Prediction (Single Image)

```bash
python pred_cell_label.py \
    --single \
    --cell_detection_dir path/to/image_dir \
    --model_path path/to/model.pkl
```

#### Prediction (Batch)

```bash
python pred_cell_label.py \
    --batch \
    --data_roots path/to/data_root1 path/to/data_root2 \
    --model_path path/to/model.pkl \
    --n_jobs 4
```

### 2. Tissue Classification

#### Preparation of Training Data

The training dataset should be organized in the following structure:

```
data_root/
├── esophageal_gland/              # Esophageal gland tissue
│   ├── slide_001.svs_44948_9940.png
│   ├── slide_001.svs_45469_9419.png
│   └── ...
├── interstitial_region/           # Interstitial region
│   ├── slide_001.svs_44948_9940.png
│   └── ...
├── mucosal_epithelium/          # Mucosal epithelium
│   ├── slide_001.svs_44948_9940.png
│   └── ...
├── muscle_tissue/               # Muscle tissue
│   ├── slide_001.svs_44948_9940.png
│   └── ...
├── submucosa/                  # Submucosa
│   ├── slide_001.svs_44948_9940.png
│   ├── slide_002.svs_6463_38916.png
│   └── ...
├── tumor_necrosis_region/       # Tumor necrosis region
│   ├── slide_001.svs_44948_9940.png
│   └── ...
└── tumor_region/                # Tumor region
    ├── slide_001.svs_44948_9940.png
    └── ...
```

**File Format Details:**

- **Images**: PNG format patches extracted from whole slide images
  - Naming convention: `{slide_id}_{x_coord}_{y_coord}.png`
  - Each directory represents a tissue type (class)
  - Images are organized by tissue type for easy dataset management

**Tissue Types:**

| ID | Tissue Type             | Description             |
| -- | ----------------------- | ----------------------- |
| 0  | esophageal\_gland       | Esophageal gland tissue |
| 1  | interstitial\_region    | Interstitial region     |
| 2  | mucosal\_epithelium     | Mucosal epithelium      |
| 3  | muscle\_tissue          | Muscle tissue           |
| 4  | submucosa               | Submucosa layer         |
| 5  | tumor\_necrosis\_region | Tumor necrosis region   |
| 6  | tumor\_region           | Tumor region            |

**Data Preparation Steps:**

1. Extract image patches from whole slide images
2. Organize patches by tissue type into corresponding directories
3. Ensure consistent naming convention for all image files
4. Verify image quality and annotations
5. Split data into training and validation sets if needed

#### Training with Swin Transformer

Note: This requires mmpretrain library. Install with:

```bash
pip install mmpretrain
```

Then run training:

```bash
# Train
python path/to/mmpretrain/tools/train.py \
    tissue_classfication/swin.py \

# Test
python path/to/mmpretrain/tools/test.py \
    tissue_classfication/swin.py \
    path/to/workdirs/epoch_8.pth \
    --out path/to/predictions_epoch_8.pkl \
    --out-item pred

# Confusion matrix
python path/to/mmpretrain/tools/analysis_tools/confusion_matrix.py \
    tissue_classfication/swin.py \
    path/to/workdirs/epoch_8.pth \
    --show \
    --show-path path/to/epoch_8_confusion_matrix.png \
    --include-values
```

#### Prediction

```bash
cd tissue_classfication
python pred_tissue_label.py \
    --input_dir path/to/h5_files \
    --output_dir path/to/output \
    --model_path path/to/model.pth \
    --device cuda
```

### 3. Feature Calculation

#### Preparation of Training Data

The feature calculation requires the following data structure:

```
data_root/
├── preprocessing/                          # Preprocessed cell detection results
│   └── {image_name}/                      # Image-specific directory
│       └── cell_detection_clf/
│           └── cells.json                 # Cell annotations (centroid, type)
├── clam_patches/                          # CLAM patch information
│   └── patches/
│       └── {image_name}.h5                # Patch coordinates and metadata
├── clam_gigapath_pred_label/              # Tissue classification predictions
│   └── h5_files/
│       └── {image_name}.h5                # Tissue type labels for patches
└── features/                              # Output directory (auto-created)
    └── {image_name}.csv                   # Calculated features (output)
```

**Input Data Format:**

- **cells.json**: JSON file containing cell annotations
- **CLAM patches HDF5**: Contains patch coordinates and metadata
  - `coords`: Patch coordinates
  - Attributes: `downsample`, `patch_size`, `downsampled_level_dim`
- **Tissue prediction HDF5**: Contains tissue type predictions for patches
  - `coords`: Patch coordinates
  - Tissue type labels

#### Calculate Region Features

```bash
cd calculate_features
python calculate_region_features.py \
    --data_roots /path/to/dataset1 /path/to/dataset2 \
    --output all_features.csv \
    --cells_path_template preprocessing/{}/cell_detection_clf/cells.json \
    --clam_patches_template clam_patches/patches/{}.h5 \
    --clam_gigapath_pred_template clam_gigapath_pred_label/h5_files/{}.h5 \
    --features_save_path features/{}.csv \
    --grid_size 256 \
    --gaussian_kernel 5 5 \
    --gaussian_sigma 0 \
    --tumor_cell_type 1 \
    --mucosal_epithelium_type 2 \
    --density_thresholds 1 5 10 20 \
    --min_tumor_cells 20 \
    --max_workers 4 \
    --batch_size 10000 \
    --distance_threshold 256.0 \
    --min_cells_for_spatial 2 \
    --save_plots
```

**Parameters:**

**Input/Output**

- `--data_roots`: List of data directories to process (required)
- `--output`: Output feature file path (default: all\_features.csv)

**Path Templates**

- `--cells_path_template`: Path template for cell annotation file (default: preprocessing/{}/cell\_detection\_clf/cells.json)
- `--clam_patches_template`: Path template for CLAM patch file (default: clam\_patches/patches/{}.h5)
- `--clam_gigapath_pred_template`: Path template for tissue prediction file (default: clam\_gigapath\_pred\_label/h5\_files/{}.h5)
- `--features_save_path`: Path template for feature output (default: features/{}.csv)

**KDE Parameters**

- `--grid_size`: Grid size for kernel density estimation (default: 256)
- `--gaussian_kernel`: Gaussian kernel size (height, width) (default: 5 5)
- `--gaussian_sigma`: Gaussian kernel standard deviation (default: 0)

**Region Parameters**

- `--tumor_cell_type`: Tumor cell type ID (default: 1)
- `--mucosal_epithelium_type`: Mucosal epithelium cell type ID (default: 2)
- `--density_thresholds`: Density thresholds for region partitioning (default: 1 5 10 20)
- `--min_tumor_cells`: Minimum number of tumor cells required (default: 20)

**Processing Parameters**

- `--max_workers`: Maximum number of parallel workers (default: 2)
- `--batch_size`: Batch processing size (default: 10000)

**Feature Parameters**

- `--distance_threshold`: Distance threshold for spatial analysis (default: 256.0)
- `--min_cells_for_spatial`: Minimum cells for spatial features (default: 2)

**Debug Parameters**

- `--save_plots`: Save visualization plots (flag)

**Output Features:**

The script calculates the following types of features:

1. **Global Features** (across entire image):
   - Total count and ratio of each cell type
   - Area ratios between regions
   - Density ratios of cell types between regions
2. **Region Features** (within each density-based region):
   - Total cells and area per region
   - Density of each cell type
   - Ratio of each cell type
   - Cell count ratios between types
   - Area ratio with previous region
3. **Spatial Features** (nearest neighbor analysis):
   - Distance statistics: min, p25, median, p75, max, mean, std, skew, kurt, cv
   - Neighbor count statistics: total, mean, max, density
   - Calculated for each cell type pair within each region

### 4. Survival Analysis

#### Run Complete Pipeline

```bash
cd survival_analysis
python main.py \
    --work_dir path/to/work_dir \
    --modal path \
    --feature_selection_method all \
    --model_type cox \
    --do_train \
    --do_test
```

#### Available Modalities

- `clinical` - Clinical data only
- `path` - Pathology features only
- `wes` - Whole exome sequencing data
- `tcr` - TCR diversity data
- `all` - All modalities combined

#### Available Models

- `cox` - Cox proportional hazards model
- `rsf` - Random survival forest
- `gb` - Gradient boosting survival model

#### Feature Selection Methods

- `all` - Use all features
- `univariate` - Univariate Cox regression
- `auc` - AUC-based selection
- `lasso_cox` - Lasso Cox regression

## Configuration

### Path Configuration

All paths in the code use placeholder format `path/to/...`. Before running:

1. Update paths in each module's main script
2. Example paths to configure:
   - Data directories (images, features, clinical data)
   - Model save/load paths
   - Output directories

### Key Parameters

#### Cell Classification

- `feature_type`: `traditional_feature`, `deep_feature`, or `all`
- `use_class_weight`: Enable class weighting
- `use_smote`: Enable SMOTE oversampling

#### Tissue Classification

- `input_size`: Feature dimension (default: 1536)
- `num_classes`: Number of tissue classes (default: 7)
- `device`: Run device (auto/cuda/cpu)

#### Feature Calculation

- `grid_size`: KDE grid size (default: 256)
- `distance_threshold`: Spatial analysis threshold (default: 256.0)
- `min_tumor_cells`: Minimum tumor cells for analysis (default: 20)

#### Survival Analysis

See survival analysis module documentation for detailed parameters.

## Cell Types

The project recognizes the following cell types:

| ID | Type                        | Color           |
| -- | --------------------------- | --------------- |
| 0  | background                  | \[0, 0, 0]      |
| 1  | tumor\_cell                 | \[211, 47, 47]  |
| 2  | lymphocyte                  | \[25, 118, 210] |
| 3  | plasma\_cell                | \[142, 36, 170] |
| 4  | neutrophil                  | \[255, 160, 0]  |
| 5  | eosinophil                  | \[245, 124, 0]  |
| 6  | interstitial\_spindle\_cell | \[56, 142, 60]  |

## Features

### Region-Based Features

- **Density features**: Cell density per region
- **Ratio features**: Proportions of different cell types
- **Spatial features**: Nearest neighbor distances and counts
- **Area features**: Region area calculations

### Spatial Analysis

- GPU-accelerated nearest neighbor distance calculation
- Multi-type cell interaction analysis
- Configurable distance thresholds

## Output

### Cell Classification

- `model.pkl`: Trained classification model
- `cells.json`: Cell annotations with predicted types
- `cells.geojson`: GeoJSON format for visualization
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curves.png`: ROC curves for each class

### Tissue Classification

- `predictions/`: Directory containing predicted tissue labels in H5 format

### Feature Calculation

- `features/*.csv`: Region-based features for each image
- `all_features.csv`: Merged features from all images

### Survival Analysis

- `cv_results.csv`: Cross-validation results
- `km_curve_*.png`: Kaplan-Meier survival curves
- `shap_plot_*.png`: SHAP value visualizations
- `confusion_matrix_*.png`: Feature confusion matrices
- `feature_importance_*.csv`: Feature importance rankings

## Performance Optimization

- **Parallel processing**: Multi-process batch processing
- **GPU acceleration**: CUDA for spatial analysis
- **Caching**: Feature caching to avoid recomputation
- **Chunk processing**: Memory-efficient large-scale processing

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use chunk processing
   - Close unnecessary applications
2. **CUDA Errors**
   - Check GPU availability
   - Reduce model complexity
   - Use CPU fallback
3. **Path Not Found**
   - Verify all paths are configured
   - Check file permissions
   - Ensure data directories exist

## Citation

If you use this project in your research, please cite:

```bibtex
@software{gao2026imagene,
  title = {IMAGENE-ESCC-Survival: A Multimodal Deep Learning Framework for Postoperative Overall Survival Prediction in Esophageal Squamous Cell Carcinoma},
  author = {Gao, Shuaiqiang},
  year = {2026},
  url = {https://github.com/OpenGene/IMAGENE-ESCC-Survival},
  version = {1.0},
  note = {Comprehensive analysis framework integrating cell classification, tissue classification, and survival analysis for esophageal cancer pathology images}
}
```

## License

Please refer to the project license file for usage terms.

## Contact

For questions or issues, please contact the project maintainers.

## Acknowledgments

This project uses the following open-source libraries and tools:

- PyTorch
- scikit-learn
- lifelines
- OpenCV
- optuna
- SHAP

## References

\[1] Hörst F, Rempe M, Heine L, et al. CellViT: Vision Transformers for precise cell segmentation and classification\[J]. Medical Image Analysis, 2024, 94: 103143. <https://doi.org/10.1016/j.media.2024.103143>.

\[2] MMPreTrain Contributors. OpenMMLab's Pre-training Toolbox and Benchmark\[EB/OL]. 2023. <https://github.com/open-mmlab/mmpretrain>.

\[3] Lu MY, Williamson DFK, Chen TY, et al. Data-efficient and weakly supervised computational pathology on whole-slide images\[J]. Nature Biomedical Engineering, 2021, 5(6): 555-570. <https://doi.org/10.1038/s41551-021-00755-2>.

\[4] Venkatachalapathy S, Jokhun DS, Shivashankar GV. Multivariate analysis reveals activation-primed fibroblast geometric states in engineered 3D tumor microenvironments\[J]. Molecular Biology of the Cell, 2020, 31(8): 803-812. <https://doi.org/10.1091/mbc.E19-08-0479>.

\[5] Xu H, Usuyama N, Bagga J, et al. A whole-slide foundation model for digital pathology from real-world data\[J]. Nature, 2024. <https://doi.org/10.1038/s41586-024-07241-0>.

