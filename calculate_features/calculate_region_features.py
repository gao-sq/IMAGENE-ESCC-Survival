import os
import sys
import cv2
import h5py
import ujson
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from functools import lru_cache
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations
import argparse

from nearest_neighbor_distance import analyze_nearest_neighbor_distance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TissueAnalysis")


@dataclass
class SpatialParameters:
    bounds: Tuple[float, float, float, float]
    grid_size: int
    patch_size: float
    x_bins: int
    y_bins: int

@dataclass
class FilteredData:
    centroid: np.ndarray
    types: np.ndarray

@dataclass
class KDEResult:
    pdf: np.ndarray
    grid_coords: np.ndarray

@dataclass
class RegionResult:
    regions: np.ndarray
    areas: np.ndarray
    cell_region_ids: np.ndarray
    region_num: int

@lru_cache(maxsize=32)
def read_hdf5(path: str) -> Dict[str, Any]:
    """Cache reading HDF5 file"""
    with h5py.File(path, "r") as f:
        return {k: (f[k][()], dict(f[k].attrs)) for k in f.keys()}

def calculate_spatial_parameters(patch_data: Dict, config: Dict[str, Any]) -> Optional[SpatialParameters]:
    """Calculate spatial parameters"""
    try:
        coords, attrs = patch_data["coords"]
        downsample = np.array(attrs["downsample"])
        patch_size = attrs["patch_size"] * downsample
        level_dim = attrs["downsampled_level_dim"]
        
        x_max = level_dim[0] * downsample[0]
        y_max = level_dim[1] * downsample[1]
        grid_size = config["kde_params"]["grid_size"]
        
        return SpatialParameters(
            bounds=(0, x_max, 0, y_max),
            grid_size=grid_size,
            patch_size=patch_size[0],  # Assume x and y directions are the same
            x_bins=int(np.ceil(x_max / grid_size)),
            y_bins=int(np.ceil(y_max / grid_size))
        )
    except KeyError as e:
        logger.error(f"Missing spatial parameter: {str(e)}")
        return None

def filter_valid_cells(
    centroid: np.ndarray,
    cell_types: np.ndarray,
    pred_data: Dict,
    spatial: SpatialParameters
) -> Optional[FilteredData]:
    """Filter valid cell data"""
    try:
        # Build spatial index
        coords, _ = pred_data["coords"]
        patch_boxes = np.hstack([coords, coords + spatial.patch_size])
        tree = cKDTree(patch_boxes.reshape(-1, 2))
        
        # Batch query cells belonging to patches
        distances, indices = tree.query(
            centroid,
            k=4,  # Query 4 nearest vertices
            distance_upper_bound=spatial.patch_size*np.sqrt(2)
        )
        
        # Determine valid patches
        filtered_mask = np.any(distances < spatial.patch_size*np.sqrt(2), axis=1)

        return FilteredData(
            centroid=centroid[filtered_mask],
            types=cell_types[filtered_mask]
        )


    except Exception as e:
        logger.error(f"Cell filtering failed: {str(e)}")
        return None

def compute_kernel_density(
    data: FilteredData,
    spatial: SpatialParameters,
    config: Dict[str, Any]
) -> Optional[KDEResult]:
    """Calculate kernel density estimation"""
    try:
        tumor_mask = (data.types == config["region_params"]["tumor_cell_type"])

        logger.info(f"Total tumor cells: {np.sum(tumor_mask)}")
        
        if np.sum(tumor_mask) < config["region_params"]["min_tumor_cells"]:
            logger.warning("Insufficient tumor cells")
            return None
        
        # Discretize coordinates
        x_coords = np.clip(
            (data.centroid[:,0] // spatial.grid_size).astype(int),
            0, spatial.x_bins-1
        )
        y_coords = np.clip(
            (data.centroid[:,1] // spatial.grid_size).astype(int),
            0, spatial.y_bins-1
        )
        
        # Generate density matrix
        pdf = np.bincount(
            x_coords[tumor_mask] * spatial.y_bins + y_coords[tumor_mask],
            minlength=spatial.x_bins*spatial.y_bins
        ).reshape(spatial.x_bins, spatial.y_bins).astype(np.float32)
        
        # Gaussian smoothing
        pdf = cv2.GaussianBlur(
            pdf,
            config["kde_params"]["gaussian_kernel"],
            config["kde_params"]["sigma"]
        )
        if config["debug"]["save_plots"]:
            visualize_pdf(pdf)
        return KDEResult(
            pdf=pdf,
            grid_coords=np.column_stack((x_coords, y_coords))
        )
    except Exception as e:
        logger.error(f"KDE computation failed: {str(e)}")
        return None
    
def visualize_regions(regions: np.ndarray, config: Dict[str, Any], save_path: str = None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.figure(figsize=(10, 8))
    region_num = len(config["region_params"]["density_thresholds"]) + 1
    plt.imshow(regions.T * region_num / (region_num - 1) - 0.5 , cmap=ListedColormap(['white', 'blue', 'green', 'yellow', 'red']))
    plt.colorbar(label="Region Level")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def visualize_pdf(pdf: np.ndarray, save_path: str = None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.figure(figsize=(10, 8))
    plt.imshow(pdf.T, cmap='hot')
    plt.colorbar(label="Density")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def visualize_density_thresholds(pdf: np.ndarray, thresholds: List[float], save_path: str = None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.figure(figsize=(10, 8))
    pdf_sorted = np.sort(pdf.flatten())
    plt.plot(pdf_sorted)
    # Draw threshold lines and threshold positions
    for t in thresholds:
        plt.axhline(y=t, color='r', linestyle='--') # Draw threshold line
        plt.axvline(x=np.sum(pdf_sorted < t), color='r', linestyle='--') # Draw threshold position
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def perform_region_analysis(kde_result: KDEResult, config: Dict[str, Any]) -> Optional[RegionResult]:
    """Perform region partition analysis"""
    try:
        # Region partition
        regions = np.digitize(
            kde_result.pdf,
            config["region_params"]["density_thresholds"]
        )
        region_num = len(config["region_params"]["density_thresholds"]) + 1
        
        # Calculate region area
        grid_area = config["kde_params"]["grid_size"] ** 2
        areas = np.bincount(regions.ravel(), minlength=region_num) * grid_area
        
        # Map cells to regions
        cell_region_ids = regions[
            kde_result.grid_coords[:,0], 
            kde_result.grid_coords[:,1]
        ]
        if config["debug"]["save_plots"]:
            visualize_regions(regions, config)

        if config["debug"]["save_plots"]:
            visualize_density_thresholds(kde_result.pdf, config["region_params"]["density_thresholds"])
        
        return RegionResult(
            regions=regions,
            areas=areas,
            cell_region_ids=cell_region_ids,
            region_num=region_num
        )
    except Exception as e:
        logger.error(f"Region analysis failed: {str(e)}")
        return None

# ================== Feature Extraction Module ==================

def calculate_region_features(spatial_params: SpatialParameters, 
                             cell_data: FilteredData,
                             type_num: int,
                             distance_threshold: float,
                             config: Dict[str, Any],
                             calculate_region_feature: bool = False,
                             calculate_spatial_features: bool = False
                             ) -> Dict[str, float]:

    """Calculate region features (including spatial features)"""
    features = {}

    # ================== Total count and ratio of each cell type ==================
    total_cells = len(cell_data.centroid)

    for t in range(1, type_num+1):
        type_total = (cell_data.types==t).sum()
        features[f"global_total_cells_type_{t}"] = type_total
        if total_cells > 0:
            features[f"global_ratio_type_{t}"] = type_total / total_cells
        else:
            features[f"global_ratio_type_{t}"] = 0.0

    if calculate_region_feature:
        # Density analysis
        kde_result = compute_kernel_density(cell_data, spatial_params, config)
        if not kde_result:
            return None
            
        # Region partition
        region_data = perform_region_analysis(kde_result, config)
        if not region_data:
            return None
        
        # ========================= Inter-region features =========================
        valid_regions = [r for r in range(region_data.region_num) 
                        if region_data.areas[r] > 0]
        
        # Region area ratio
        for i, j in combinations(valid_regions, 2):
            ratio = region_data.areas[i] / region_data.areas[j]
            features[f"global_area_ratio_{i}_{j}"] = ratio

        # Density ratio of same cell type between regions
        for i, j in combinations(valid_regions, 2):
            for t in range(1, type_num+1):
                key = f"global_density_ratio_{i}_{j}_type_{t}"
                density_i = features.get(f"region_{i}_density_type_{t}", 0)
                density_j = features.get(f"region_{j}_density_type_{t}", 0)
                ratio = density_i / (density_j + 1e-8)
                features[key] = ratio
        
        # ========================= Intra-region features =========================
        for region_id in range(region_data.region_num):
            mask = region_data.cell_region_ids == region_id
            if np.sum(mask) == 0:
                continue

            # ================== Basic statistics ==================
            region_cells = cell_data.centroid[mask]
            region_types = cell_data.types[mask]
            
            # Type statistics (excluding background type 0)
            type_counts = np.bincount(region_types, minlength=type_num+1)[1:]
            total_cells = np.sum(type_counts)
            
            # ================== Density features ==================
            area = region_data.areas[region_id]
            density_features = {
                f"region_{region_id}_density_type_{i+1}": (count / area) if area > 0 else 0.0
                for i, count in enumerate(type_counts)
            }
            
            # ================== Ratio features ==================
            ratio_features = {}
            if total_cells > 0:
                ratio_features = {
                    f"region_{region_id}_ratio_type_{i+1}": count / total_cells
                    for i, count in enumerate(type_counts)
                }

            # Cell count ratio within region
            for i, j in combinations(range(type_num), 2):
                key = f"region_{region_id}_ratio_{i+1}_to_{j+1}"
                ratio = type_counts[i] / (type_counts[j] + 1e-8)
                ratio_features[key] = ratio

            # ================== Area ratio with previous region ==================
            if region_id > 0:
                prev_area = region_data.areas[region_id-1]
                area_ratio = area / prev_area if prev_area > 0 else 0.0
                features[f"region_{region_id}_area_ratio_prev"] = area_ratio
            
            # ================== Spatial features ==================
            if calculate_spatial_features and len(region_cells) >= 2:  # At least 2 cells needed for spatial features
                # Calculate nearest neighbor features
                nn_distances, nn_counts = analyze_nearest_neighbor_distance(
                    points=region_cells,
                    types=region_types,
                    type_num=type_num,
                    distance_threshold=distance_threshold
                )
                # Nearest neighbor distance statistics
                nn_distance_features = {}
                for src_type in range(type_num):
                    src_mask = (region_types == (src_type + 1))
                    if not np.any(src_mask):
                        continue
                    
                    for dst_type in range(type_num):
                        distances = nn_distances[src_mask, dst_type]
                        valid_distances = distances[~np.isinf(distances)]
                        
                        if len(valid_distances) > 0:
                            stats = {
                                "min": np.min(valid_distances),
                                "p25": np.percentile(valid_distances, 25),
                                "median": np.median(valid_distances),
                                "p75": np.percentile(valid_distances, 75),
                                "max": np.max(valid_distances),
                                "mean": np.mean(valid_distances),
                                "std": np.std(valid_distances),
                                "skew": pd.Series(valid_distances).skew(), # Skewness
                                "kurt": pd.Series(valid_distances).kurt(), # Kurtosis
                                "cv": np.std(valid_distances) / (np.mean(valid_distances) + 1e-8) # Coefficient of variation
                            }
                        else:
                            stats = {k: np.nan for k in ["min", "p25", "median", "p75", "max", 
                                                        "mean", "std", "skew", "kurt", "cv"]}
                        
                        for stat_name, value in stats.items():
                            key = f"region_{region_id}_nn_{src_type+1}_to_{dst_type+1}_{stat_name}"
                            nn_distance_features[key] = float(value)
                
                # Neighbor count statistics
                nn_count_features = {}
                for src_type in range(type_num):
                    src_mask = (region_types == (src_type + 1))
                    if not np.any(src_mask):
                        continue
                    
                    for dst_type in range(type_num):
                        counts = nn_counts[src_mask, dst_type]
                        count_stats = {
                            "total": np.sum(counts),
                            "mean": np.mean(counts),
                            "max": np.max(counts),
                            "density": np.mean(counts) / area if area > 0 else 0.0
                        }
                        
                        for stat_name, value in count_stats.items():
                            key = f"region_{region_id}_nc_{src_type+1}_to_{dst_type+1}_{stat_name}"
                            nn_count_features[key] = float(value)
                
                # Merge spatial features
                spatial_features = {**nn_distance_features, **nn_count_features}
            else:
                spatial_features = {}
            
            # ================== Merge all features ==================
            features.update({
                f"region_{region_id}_total_cells": total_cells,
                f"region_{region_id}_area": area,
                **density_features,
                **ratio_features,
                **spatial_features
            })

    return features

# ================== Main Processing Pipeline ==================

def process_single_image(data: Dict, config: Dict[str, Any]) -> Optional[Dict]:
    """Single image processing pipeline"""
    try:
        logger.info(f"Processing {data['image_name']}")
        
        # Read input data
        with open(data["cells_path"], "r") as f:
            cells = ujson.load(f)
        centroid = np.array([c["centroid"] for c in cells["cells"]])
        types = np.array([c["type"] for c in cells["cells"]])
        type_num = len(cells['type_map']) - 1
        
        # Load prediction data
        patch_data = read_hdf5(data["clam_patches"])
        pred_data = read_hdf5(data["clam_gigapath_pred_label"])
        
        # Spatial parameters calculation
        spatial_params = calculate_spatial_parameters(patch_data, config)
        if not spatial_params:
            return None
            
        # Data preprocessing
        filtered_data = filter_valid_cells(centroid, types, pred_data, spatial_params)
        if not filtered_data:
            return None
            
        # Feature extraction
        features = calculate_region_features(
            spatial_params, 
            filtered_data, 
            type_num, 
            config["feature_params"]["distance_threshold"],
            config,
            calculate_region_feature=True,
            calculate_spatial_features=True
        )

        features["image_name"] = data["image_name"]

        pd.DataFrame([features]).to_csv(data["feature_path"], index=False)

        return features
        
    except Exception as e:
        logger.error(f"Processing failed for {data['image_name']}: {str(e)}")
        return None

def batch_processing(data_list: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
    """Batch parallel processing"""
    with ProcessPoolExecutor(max_workers=config["processing"]["max_workers"]) as executor:
        # Use partial to pass config parameter
        from functools import partial
        process_func = partial(process_single_image, config=config)
        
        results = list(tqdm(
            executor.map(process_func, data_list),
            total=len(data_list),
            desc="Processing Images"
        ))
    
    valid_results = [r for r in results if r is not None]
    return pd.DataFrame(valid_results)

def get_data_paths(data_root: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get valid data paths"""
    if not os.path.exists(data_root):
        return []
    
    if not os.path.exists(os.path.join(data_root, config["features_save_path"].split("/")[0])):
        os.makedirs(os.path.join(data_root, config["features_save_path"].split("/")[0]))

    valid_data = []
    for img_name in tqdm(os.listdir(os.path.join(data_root, "preprocessing")), desc=f"Scanning {os.path.basename(data_root)}"):
        required_paths = {
            **{
                k: os.path.join(data_root, v.format(img_name))
                for k, v in config["path_templates"].items()
            }
        }
        feature_path = os.path.join(data_root, config["features_save_path"].format(img_name))
        if all(os.path.exists(p) for p in required_paths.values()) and not os.path.exists(feature_path):
            valid_data.append({
                "image_name": img_name,
                "feature_path": feature_path,
                **required_paths
            })

    return valid_data

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Script for calculating region features")
    
    # Input/output parameters
    parser.add_argument(
        "--data_roots",
        nargs="+",
        help="List of data directories to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="all_features.csv",
        help="Output feature file path"
    )
    
    # Path template parameters
    parser.add_argument(
        "--cells_path_template",
        type=str,
        default="preprocessing/{}/cell_detection_clf/cells.json",
        help="Path template for cell annotation file"
    )
    parser.add_argument(
        "--clam_patches_template",
        type=str,
        default="clam_patches/patches/{}.h5",
        help="Path template for CLAM patch file"
    )
    parser.add_argument(
        "--clam_gigapath_pred_template",
        type=str,
        default="clam_gigapath_pred_label/h5_files/{}.h5",
        help="Path template for CLAM Gigapath prediction label file"
    )
    parser.add_argument(
        "--features_save_path",
        type=str,
        default="features/{}.csv",
        help="Path template for feature save"
    )
    
    # KDE parameters
    parser.add_argument(
        "--grid_size",
        type=int,
        default=256,
        help="KDE grid size"
    )
    parser.add_argument(
        "--gaussian_kernel",
        type=int,
        nargs=2,
        default=[5, 5],
        help="Gaussian kernel size (height, width)"
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=0,
        help="Gaussian kernel standard deviation"
    )
    
    # Region parameters
    parser.add_argument(
        "--tumor_cell_type",
        type=int,
        default=1,
        help="Tumor cell type ID"
    )
    parser.add_argument(
        "--mucosal_epithelium_type",
        type=int,
        default=2,
        help="Mucosal epithelium cell type ID"
    )
    parser.add_argument(
        "--density_thresholds",
        type=int,
        nargs='+',
        default=[1, 5, 10, 20],
        help="List of density thresholds"
    )
    parser.add_argument(
        "--min_tumor_cells",
        type=int,
        default=20,
        help="Minimum number of tumor cells"
    )

    # Processing parameters
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Maximum number of worker processes for parallel processing"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch processing size"
    )
    
    # Debug parameters
    parser.add_argument(
        "--save_plots",
        action='store_true',
        help="Whether to save visualization plots"
    )
    
    # Feature parameters
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=256.0,
        help="Distance threshold for spatial analysis"
    )
    parser.add_argument(
        "--min_cells_for_spatial",
        type=int,
        default=2,
        help="Minimum number of cells for spatial feature calculation"
    )
    
    return parser.parse_args()

def build_config_from_args(args):
    """Build configuration dictionary from command line arguments"""
    return {
        "path_templates": {
            "cells_path": args.cells_path_template,
            "clam_patches": args.clam_patches_template,
            "clam_gigapath_pred_label": args.clam_gigapath_pred_template,
        },
        "features_save_path": args.features_save_path,
        "kde_params": {
            "grid_size": args.grid_size,
            "gaussian_kernel": tuple(args.gaussian_kernel),
            "sigma": args.gaussian_sigma
        },
        "region_params": {
            "tumor_cell_type": args.tumor_cell_type,
            "mucosal_epithelium_type": args.mucosal_epithelium_type,
            "density_thresholds": args.density_thresholds,
            "min_tumor_cells": args.min_tumor_cells
        },
        "processing": {
            "max_workers": args.max_workers,
            "batch_size": args.batch_size
        },
        "debug": {
            "save_plots": args.save_plots
        },
        "feature_params": {
            "distance_threshold": args.distance_threshold,
            "min_cells_for_spatial": args.min_cells_for_spatial
        }
    }

# ================== Main Program Entry ==================
if __name__ == "__main__":
    # Parse command line arguments and build configuration
    args = parse_arguments()
    CONFIG = build_config_from_args(args)
    
    # Display configuration information
    logger.info("=== Configuration ===")
    logger.info(f"Input directories: {args.data_roots}")
    logger.info(f"Output directory: {args.features_save_path}")
    logger.info(f"Grid size: {CONFIG['kde_params']['grid_size']}")
    logger.info(f"Max workers: {CONFIG['processing']['max_workers']}")
    logger.info(f"Tumor cell type: {CONFIG['region_params']['tumor_cell_type']}")
    logger.info(f"Distance threshold: {CONFIG['feature_params']['distance_threshold']}")
    logger.info(f"Save plots: {CONFIG['debug']['save_plots']}")
    
    for data_root in args.data_roots:
        # Get data paths
        data_paths = get_data_paths(data_root, CONFIG)
        
        if not data_paths:
            logger.error("No valid data paths found, please check input parameters")
            sys.exit(1)
        
        # Batch process data
        result_df = batch_processing(data_paths, CONFIG)
        
        if result_df.empty:
            logger.error("No result data generated")
            sys.exit(1)
        
        logger.info(f"Processing completed, generated {len(result_df)} records")
        
        # Save results to CSV file
        result_df.to_csv(data_root + "/" + args.output, index=False)
        logger.info(f"Results saved to: {data_root + '/' + args.output}")
