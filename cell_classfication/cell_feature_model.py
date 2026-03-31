import pandas as pd
from skimage import measure
import numpy as np
import cv2 as cv

import sys
sys.path.append('path/to/chrometrics')

import torch
import nmco
from nmco.nuclear_features import (
    global_morphology as BG,
    img_texture as IT,
    int_dist_features as IDF,
    boundary_local_curvature as BLC
)


def run_nuclear_chromatin_feat_ext(raw_image_path:str, labelled_image_path:str, output_dir:str,
                                   calliper_angular_resolution:int = 10, 
                                   measure_simple_geometry:bool = True, 
                                   measure_calliper_distances:bool = True, 
                                   measure_radii_features:bool = True,
                                   step_size_curvature:int = 2, 
                                   prominance_curvature:float = 0.1, 
                                   width_prominent_curvature:int = 5, 
                                   dist_bt_peaks_curvature:int = 10,
                                   measure_int_dist_features:bool = True, 
                                   measure_hc_ec_ratios_features:bool = True, 
                                   hc_threshold:float = 1, 
                                   gclm_lengths:list = [1, 5, 20],
                                   measure_gclm_features: bool = True, 
                                   measure_moments_features: bool = True,
                                   normalize:bool=False, 
                                   save_output:bool = False):
    """
    Function that reads in the raw and segmented/labelled images for a field of view and computes nuclear features. 
    Note this has been used only for DAPI stained images
    Args:
        raw_image_path: path pointing to the raw image
        labelled_image_path: path pointing to the segmented image
        output_dir: path where the results need to be stored
    """
    if labelled_image_path is str:
        labelled_image = imread(labelled_image_path)
    else:
        labelled_image = labelled_image_path
    if raw_image_path is str:
        raw_image = imread(raw_image_path)
    else:
        raw_image = raw_image_path

    # Check if the image is grayscale
    if raw_image.ndim == 3:
        raw_image = cv.cvtColor(raw_image, cv.COLOR_BGR2GRAY)

    labelled_image = labelled_image.astype(int)
    raw_image = raw_image.astype(int)

    # Insert code for preprocessing image
    # Eg normalize
    if normalize:
        raw_image = cv.normalize(
         raw_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
     )
        raw_image[raw_image < 0] = 0.0
        raw_image[raw_image > 255] = 255.0

    # Get features for the individual nuclei in the image
    props = measure.regionprops(labelled_image, raw_image)
    
    all_features = []
    # Measure scikit's built in features
    
    for i in range(len(props)):
        all_features.append(
            pd.concat(
                [
                 BG.measure_global_morphometrics(props[i].image, 
                                                 angular_resolution = calliper_angular_resolution, 
                                                 measure_simple = measure_simple_geometry,
                                                 measure_calliper = measure_calliper_distances, 
                                                 measure_radii = measure_radii_features).reset_index(drop=True),
                 BLC.measure_curvature_features(props[i].image, step = step_size_curvature, 
                                                prominance = prominance_curvature, 
                                                width = width_prominent_curvature, 
                                                dist_bt_peaks = dist_bt_peaks_curvature).reset_index(drop=True),
                 IDF.measure_intensity_features(props[i].image, props[i].intensity_image, 
                                                measure_int_dist = measure_int_dist_features, 
                                                measure_hc_ec_ratios = measure_hc_ec_ratios_features, 
                                                hc_alpha = hc_threshold).reset_index(drop=True),
                 IT.measure_texture_features(props[i].image, props[i].intensity_image, lengths=gclm_lengths,
                                             measure_gclm = measure_gclm_features,
                                             measure_moments = measure_moments_features)
                ],
                axis=1,
            )
        )
   
    all_features = pd.concat(all_features, ignore_index=True)

    if save_output:
        all_features.to_csv(output_dir+"/"+labelled_image_path.rsplit('/', 1)[-1][:-4]+".csv")

    return all_features


# Input cell image, and cell mask, output cell boundary features
def get_cell_traditional_feature(cell_image, inst_map):
    cell_feature = run_nuclear_chromatin_feat_ext(cell_image, inst_map, output_dir=None)
    return np.array(cell_feature)

# Get deep learning features
def get_cell_deep_feature(cell_image, cell_mask, cell_graph):
    """
    :param cell_image: Cell image
    :param cell_mask: Cell mask
    :param cell_graph: Cell features
    :return: cell_feature: Cell deep learning feature
    """
    center_cell_idx = np.argmin(np.linalg.norm(cell_graph.positions - np.array(cell_image.shape[:2]) / 2, axis=1))
    return cell_graph.x[center_cell_idx].numpy()
