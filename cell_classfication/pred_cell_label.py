from multiprocessing import Pool
import os
import sys
from typing import List
import uuid

import pandas as pd

import numpy as np
import torch
import ujson
import joblib
import logging
import argparse

# Set logging
logging.basicConfig(level=logging.INFO)
sys.path.append('path/to/CellViT')

COLOR_DICT = {
    0: [0, 0, 0],        # background (assumed to be black)
    1: [211, 47, 47],    # tumor_cell (#D32F2F)
    2: [25, 118, 210],   # lymphocyte (#1976D2)
    3: [142, 36, 170],   # plasma_cell (#8E24AA)
    4: [255, 160, 0],    # neutrophil (#FFA000)
    5: [245, 124, 0],    # eosinophil (#F57C00)
    6: [56, 142, 60],    # interstitial_spindle_cell (#388E3C)
}

TYPE_NUCLEI_DICT = {
    1: "tumor_cell",
    2: "lymphocyte",
    3: "plasma_cell",
    4: "neutrophil",
    5: "eosinophil",
    6: "interstitial_spindle_cell",
}

TYPE_MAP = {
    "background": 0,
    "tumor_cell": 1,
    "lymphocyte": 2,
    "plasma_cell": 3,
    "neutrophil": 4,
    "eosinophil": 5,
    "interstitial_spindle_cell": 6,
}

def get_template_segmentation() -> dict:
    """Return a template for a MultiPolygon geojson object

    Returns:
        dict: Template
    """
    template_multipolygon = {
        "type": "Feature",
        "id": "TODO",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [],
            ],
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "TODO", "color": []},
        },
    }
    return template_multipolygon

def get_template_point() -> dict:
    """Return a template for a Point geojson object

    Returns:
        dict: Template
    """
    template_point = {
        "type": "Feature",
        "id": "TODO",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": [
                [],
            ],
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "TODO", "color": []},
        },
    }
    return template_point

def convert_geojson(
        cell_list, polygons: bool = False
    ) -> List[dict]:
        """Convert a list of cells to a geojson object

        Either a segmentation object (polygon) or detection points are converted

        Args:
            cell_list (list[dict]): Cell list with dict entry for each cell.
                Required keys for detection:
                    * type
                    * centroid
                Required keys for segmentation:
                    * type
                    * contour
            polygons (bool, optional): If polygon segmentations (True) or detection points (False). Defaults to False.

        Returns:
            List[dict]: Geojson like list
        """
        if polygons:
            cell_segmentation_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_segmentation_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_segmentation_df[cell_segmentation_df["type"] == cell_type]
                contours = cells["contour"].to_list()
                final_c = []
                for c in contours:
                    c.append(c[0])
                    final_c.append([c])

                cell_geojson_object = get_template_segmentation()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = final_c
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = TYPE_NUCLEI_DICT[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        else:
            cell_detection_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_detection_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_detection_df[cell_detection_df["type"] == cell_type]
                centroids = cells["centroid"].to_list()
                cell_geojson_object = get_template_point()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = centroids
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = TYPE_NUCLEI_DICT[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        return geojson_placeholder


def inference(cell_detection_dir, model_save_path, output_suffix='cell_detection_clf', 
              feature_file='cells.pt', cells_file='cells.json', skip_existing=False):
    """
    Perform inference classification on a single cell detection directory
    
    Args:
        cell_detection_dir (str): Cell detection directory path
        model_save_path (str): Model file save path
        output_suffix (str): Output directory suffix (default: cell_detection_clf）
        feature_file (str): Feature file name (default: cells.pt）
        cells_file (str): Cell JSON file name (default: cells.json）
        skip_existing (bool): Whether to skip existing output files (default: False）
    """
    logging.info(f'cell_detection_dir {cell_detection_dir} start')
    feature_path = os.path.join(cell_detection_dir, 'cell_detection', feature_file)
    cells_path = os.path.join(cell_detection_dir, 'cell_detection', cells_file)
    output_dir = os.path.join(cell_detection_dir, output_suffix)
    
    if not os.path.exists(feature_path):
        logging.info(f'feature_path {feature_path} not exists')
        return
    if not os.path.exists(cells_path):
        logging.info(f'cells_path {cells_path} not exists')
        return
    if not os.path.exists(model_save_path):
        logging.info(f'model_save_path {model_save_path} not exists')
        return
    
    # Check whether to skip existing files
    if skip_existing:
        output_files = [
            os.path.join(output_dir, 'cells.json'),
            os.path.join(output_dir, 'cells.geojson'),
            os.path.join(output_dir, 'cell_detection.geojson')
        ]
        if all(os.path.exists(f) for f in output_files):
            logging.info(f'cell_detection_dir {cell_detection_dir} output files already exist, skipping')
            return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model with warning suppression
    try:
        clf = joblib.load(model_save_path)
        logging.info(f'Model loaded successfully from {model_save_path}')
    except Exception as e:
        logging.error(f'Failed to load model from {model_save_path}: {e}')
        return

    # Load deep features
    deep_features = torch.load(feature_path)
    logging.info(f'deep_features: {deep_features.x.shape}')

    # Ensure feature data type is correct (convert to float64 to match sklearn expectations)
    features_numpy = deep_features.x.numpy()
    
    # Data type validation and conversion
    if features_numpy.dtype != np.float64:
        logging.info(f'Converting feature dtype from {features_numpy.dtype} to float64')
        features_numpy = features_numpy.astype(np.float64)
    
    # Handle NaN values
    if np.any(np.isnan(features_numpy)):
        logging.warning('Found NaN values in features, replacing with 0')
        features_numpy = np.nan_to_num(features_numpy, nan=0.0)
    
    # inference
    try:
        pred_labels = clf.predict(features_numpy)
        logging.info(f'Prediction completed, {len(pred_labels)} labels generated')
    except Exception as e:
        logging.error(f'Prediction failed: {e}')
        return

    # Load geojson
    with open(cells_path, 'r') as f:
        cell_dict_wsi = ujson.load(f)

    cell_dict_wsi['type_map'] = TYPE_MAP

    assert len(pred_labels) == len(cell_dict_wsi['cells'])
    # Update cell type
    cell_types = pred_labels.tolist()
    for cell, cell_type in zip(cell_dict_wsi['cells'], cell_types):
        cell['type'] = cell_type

    cells_path = os.path.join(output_dir, 'cells.json')
    with open(cells_path, 'w') as f:
        ujson.dump(cell_dict_wsi, f, indent=2)

    geojson_list = convert_geojson(cell_dict_wsi["cells"], True)
    save_path = os.path.join(output_dir, 'cells.geojson')
    with open(save_path, 'w') as f:
        ujson.dump(geojson_list, f, indent=2)

    geojson_list = convert_geojson(cell_dict_wsi["cells"], False)
    save_path = os.path.join(output_dir, 'cell_detection.geojson')
    with open(save_path, 'w') as f:
        ujson.dump(geojson_list, f, indent=2)
    
    logging.info(f'cell_detection_dir {cell_detection_dir} done')

# Multi-process inference
def inference_parallel(data_root, model_save_path, n_jobs=4, output_suffix='cell_detection_clf',
                      feature_file='cells.pt', cells_file='cells.json', skip_existing=False):
    """
    Parallel inference for multiple cell detection directories
    
    Args:
        data_root (str): Data root directory path
        model_save_path (str): Model file save path
        n_jobs (int): Number of parallel processes
        output_suffix (str): Output directory suffix (default: cell_detection_clf）
        feature_file (str): Feature file name (default: cells.pt）
        cells_file (str): Cell JSON file name (default: cells.json）
        skip_existing (bool): Whether to skip existing output files (default: False）
    """
    cell_detection_dirs = [os.path.join(data_root, cell_detection_dir) for cell_detection_dir in os.listdir(data_root)]
    with Pool(n_jobs) as pool:
        pool.starmap(inference, [(cell_detection_dir, model_save_path, output_suffix, feature_file, cells_file, skip_existing) 
                               for cell_detection_dir in cell_detection_dirs])
    # for cell_detection_dir in cell_detection_dirs:
    #     inference(cell_detection_dir, model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cell classification inference')
    
    # Mode selection parameters
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--single', action='store_true', 
                           help='Single directory inference mode')
    mode_group.add_argument('--batch', action='store_true',
                           help='Batch inference mode')
    
    # Single directory inference parameters
    parser.add_argument('--cell_detection_dir', type=str,
                        help='Single cell detection directory path (for single directory inference mode)')
    
    # Batch inference parameters
    parser.add_argument('--data_roots', nargs='+',
                        help='Multiple data root directory paths, space separated (for batch inference mode)')
    
    # Common parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Model file path')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel processes (default: 4)')
    parser.add_argument('--output_suffix', type=str, default='cell_detection_clf',
                        help='Output directory suffix (default: cell_detection_clf)')
    parser.add_argument('--feature_file', type=str, default='cells.pt',
                        help='Feature file name (default: cells.pt)')
    parser.add_argument('--cells_file', type=str, default='cells.json',
                        help='Cell JSON file name (default: cells.json)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip existing output files')
    
    args = parser.parse_args()
    
    # Parameter validation
    if args.single and not args.cell_detection_dir:
        parser.error('--single mode requires specifying --cell_detection_dir')
    
    if args.batch and not args.data_roots:
        parser.error('--batch mode requires specifying --data_roots')
    
    # Execute inference
    if args.single:
        # Single directory inference
        inference(cell_detection_dir=args.cell_detection_dir,
                 model_save_path=args.model_path,
                 output_suffix=args.output_suffix,
                 feature_file=args.feature_file,
                 cells_file=args.cells_file,
                 skip_existing=args.skip_existing)
    else:
        # Batch inference
        for data_root in args.data_roots:
            if not os.path.exists(data_root):
                logging.warning(f'Preprocessing directory does not exist: {data_root}')
                continue
                
            inference_parallel(data_root=data_root,
                             model_save_path=args.model_path,
                             n_jobs=args.n_jobs,
                             output_suffix=args.output_suffix,
                             feature_file=args.feature_file,
                             cells_file=args.cells_file,
                             skip_existing=args.skip_existing)
