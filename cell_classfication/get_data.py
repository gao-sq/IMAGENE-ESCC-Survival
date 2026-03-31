import json
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from multiprocessing import Pool

from cell_feature_model import get_cell_traditional_feature, get_cell_deep_feature

class CellFeatureDataset():
    def __init__(self, data_root, folds, info_file='info.json', feature_type='deep_feature'):
        info = json.load(open(os.path.join(data_root, info_file), 'r'))
        self.class_map = info['class_map']
        self.class_colors_map = info['class_colors_map']
        self.fold_img_map = {f'fold{k}': v.encode('utf-8').decode('unicode_escape') for k, v in info['id_img_map'].items()}
        self.image_paths, self.features, self.labels = self.get_folds_data(data_root, folds, feature_type)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        return image, self.features[idx], self.labels[idx]
    
    def get_folds_data(self, data_root, folds, feature_type='deep_feature'):

        image_paths = []
        features = [] # [{traditional_feature: [1,n], deep_feature: [m,]}, ...]
        labels = []
        for fold in folds:
            image_names, fold_features, fold_labels = get_features_labels(os.path.join(data_root, fold), use_cache=True)
            image_paths.extend([os.path.join(data_root, fold, 'images', image_name) for image_name in image_names])
            features.extend(fold_features)
            labels.extend(fold_labels)
        # shape: [1,n] + [m,] -> [n+m]
        if feature_type == 'traditional_feature':
            features = [feature['traditional_feature'].reshape(-1) for feature in features]
        elif feature_type == 'deep_feature':
            features = [feature['deep_feature'] for feature in features]
        else:
            features = [np.concatenate([feature['traditional_feature'].reshape(-1), feature['deep_feature']]) for feature in features]
        # shape: [n+m,] + [n+m,] + ... -> [k, n+m]
        features = np.stack(features, axis=0)
        labels = np.array(labels)
        return image_paths, features, labels

def save_all_features_labels(data_root):
    folds = [os.path.join(data_root, fold) for fold in os.listdir(data_root) if fold.startswith('fold')]
    with Pool() as pool:
        results = pool.starmap(get_features_labels, [(fold, False) for fold in folds])

def get_features_labels(data_dir, use_cache=False):
    if os.path.exists(os.path.join(data_dir, 'features.npy')) and use_cache:
        features = np.load(os.path.join(data_dir, 'features.npy'), allow_pickle=True)
        labels = np.load(os.path.join(data_dir, 'labels.npy'))
        image_paths = json.load(open(os.path.join(data_dir, 'image_paths.json'), 'r'))
    else:
        image_paths = []
        features = []
        labels = []
        images_dir = os.path.join(data_dir, 'images')
        labels_dir = os.path.join(data_dir, 'labels')
        cell_detection_dir = os.path.join(data_dir, 'cell_detection', 'cell_graph')
        for file in os.listdir(images_dir):

            image_path = os.path.join(images_dir, file)
            label_path = os.path.join(labels_dir, file.replace('.png', '.npy'))
            cell_detection_path = os.path.join(cell_detection_dir, file.replace('.png', '.pt'))
            if not os.path.exists(cell_detection_path):
                print(f'cell detection file {cell_detection_path} not exists')
                cell_detection_path = None
            else:
                feature, label = get_cell_feature(image_path, label_path, cell_detection_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(label)
                    image_paths.append(file)
        json.dump(image_paths, open(os.path.join(data_dir, 'image_paths.json'), 'w'))
        np.save(os.path.join(data_dir, 'features.npy'), features)
        np.save(os.path.join(data_dir, 'labels.npy'), labels)
    return image_paths, features, labels

def get_cell_feature(image_path, label_path, cell_detection_path):
    if not os.path.exists(label_path):
        return None, None
    
    image = cv2.imread(image_path)
    label = np.load(label_path, allow_pickle=True).item()

    feature = {}

    feature['traditional_feature'] = get_cell_traditional_feature(image, label['inst_map'])

    if cell_detection_path is not None:
        graph = torch.load(cell_detection_path)
        feature['deep_feature'] = get_cell_deep_feature(image, label['inst_map'], graph)
    
    return feature, np.max(label['type_map'])



if __name__ == '__main__':
    data_root = 'path/to/single_cell_dataset'
    save_all_features_labels(data_root)
    
