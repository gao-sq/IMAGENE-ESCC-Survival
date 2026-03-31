# IMAGENE-ESCC-Survival
IMAGENE (Subproject) - Python Implementation of "A Multimodal Deep Learning Framework for Postoperative Overall Survival Prediction in Esophageal Squamous Cell Carcinoma"

## Project Overview

This project provides a complete workflow for analyzing pathology images, with three main components:

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
│   ├── pred_tissue_label.py        # Predict tissue labels
│   └── swin.py                     # Swin Transformer config
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

## Requirements

### Python Dependencies
- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- opencv-python
- h5py
- lifelines
- scikit-survival
- optuna
- shap
- matplotlib
- seaborn
- tqdm
- ujson

### Hardware Requirements
- GPU (CUDA-capable) for deep learning models
- Sufficient RAM for large-scale image processing
- Storage space for intermediate results

## Installation

1. Clone the repository
```bash
cd /path/to/project
```

2. Create conda environment
```bash
conda create -n pathology_analysis python=3.8
conda activate pathology_analysis
```

3. Install dependencies
```bash
pip install torch torchvision
pip install scikit-learn pandas numpy opencv-python h5py
pip install lifelines scikit-survival optuna shap
pip install matplotlib seaborn tqdm ujson
```

## Usage

### 1. Cell Classification

#### Training
```bash
cd cell_classfication
python main.py
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

#### Training with Swin Transformer

Note: This requires mmpretrain library. Install with:
```bash
pip install mmpretrain
```

Then run training:
```bash
# Train
python path/to/mmpretrain/tools/train.py \
    tissue_classfication/config/swin_dump_fixed.py \
    path/to/workdirs/tissue/epoch_8.pth

# Test
python path/to/mmpretrain/tools/test.py \
    tissue_classfication/config/swin_dump_fixed.py \
    path/to/workdirs/tissue/epoch_8.pth \
    --out path/to/tissue/predictions_epoch_8.pkl \
    --out-item pred

# Confusion matrix
python path/to/mmpretrain/tools/analysis_tools/confusion_matrix.py \
    tissue_classfication/config/swin_dump_fixed.py \
    path/to/workdirs/tissue/epoch_8.pth \
    --show \
    --show-path path/to/tissue/epoch_8_confusion_matrix.png \
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

#### Calculate Region Features
```bash
cd calculate_features
python calculate_region_features.py \
    --data_roots path/to/data1 path/to/data2 \
    --output all_features.csv \
    --grid_size 256 \
    --tumor_cell_type 1 \
    --density_thresholds 1 5 10 20
```

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

## Cell Types

The project recognizes the following cell types:

| ID | Type | Color |
|-----|-------|--------|
| 0 | background | [0, 0, 0] |
| 1 | tumor_cell | [211, 47, 47] |
| 2 | lymphocyte | [25, 118, 210] |
| 3 | plasma_cell | [142, 36, 170] |
| 4 | neutrophil | [255, 160, 0] |
| 5 | eosinophil | [245, 124, 0] |
| 6 | interstitial_spindle_cell | [56, 142, 60] |

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
@software{guangdongshengyi,
  title = {Pathology Image Analysis Pipeline},
  author = {Shuaiqiang Gao},
  year = {2026},
  note = {Comprehensive analysis framework for esophageal cancer pathology images}
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
>>>>>>> d96f429 (feat: add initial code for IMAGENE-ESCC-Survival)
