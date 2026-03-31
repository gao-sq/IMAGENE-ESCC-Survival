import joblib
import numpy as np
import os
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

from get_data import CellFeatureDataset
from train import train_model
from plot_confusion_matrix import plot_confusion_matrix
import sys
sys.path.append('path/to/guangdongshengyi')
from utils.plot import plot_roc_curves


def prepare_data(train_folds=['fold0', 'fold1', 'fold5', 'fold7'], val_folds=['fold9'], data_root=None, feature_type='all'):
    if train_folds is None:
        train_folds = [fold for fold in os.listdir(data_root) if fold.startswith('fold')]
    train_dataset = CellFeatureDataset(data_root, train_folds, feature_type=feature_type)
    train_images, train_data, train_labels = train_dataset.image_paths, train_dataset.features, train_dataset.labels

    if val_folds is None:
        np.random.seed(5)
        n = train_data.shape[0]
        idx = np.random.permutation(n)
        val_idx = idx[:int(0.2*n)]
        train_idx = idx[int(0.2*n):]
        val_data = train_data[val_idx]
        val_labels = train_labels[val_idx]
        val_images = [train_images[i] for i in val_idx]
        train_data = train_data[train_idx]
        train_labels = train_labels[train_idx]
        train_images = [train_images[i] for i in train_idx]
    else:
        val_dataset = CellFeatureDataset(data_root, val_folds, feature_type=feature_type)
        val_images, val_data, val_labels = val_dataset.image_paths, val_dataset.features, val_dataset.labels

    logging.info(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")

    train_data[np.isinf(train_data)] = np.nan
    val_data[np.isinf(val_data)] = np.nan

    train_data = np.nan_to_num(train_data).astype(np.float32)
    val_data = np.nan_to_num(val_data).astype(np.float32)

    return train_images, train_data, train_labels, val_images, val_data, val_labels, train_dataset.class_map

def train_val(train_folds=['fold0', 'fold1', 'fold5', 'fold7'],
              val_folds=['fold9'],
              data_root=None,
              confusion_matrix_save_path='confusion_matrix.png',
              model_path='model.pkl',
              feature_type='all',
              roc_curves_save_path=None
     ):
    train_images, train_data, train_labels, val_images, val_data, val_labels, class_map = prepare_data(train_folds, val_folds, data_root, feature_type=feature_type)

    clf, train_pred, val_pred, train_acc, val_acc = train_model(train_data, train_labels, val_data, val_labels, model_save_path=model_path)

    logging.info(f"Train accuracy: {train_acc}, Validation accuracy: {val_acc}")

    if confusion_matrix_save_path:
        plot_confusion_matrix(val_labels, val_pred, save_path=confusion_matrix_save_path)

    plot_roc_curves(val_labels-1, clf.predict_proba(val_data),
                     class_names=[k for k in class_map.keys()],
                     save_path=roc_curves_save_path)

    return train_images, val_images, train_labels, val_labels, train_pred, val_pred


def test(train_folds=['fold0', 'fold1', 'fold5', 'fold7'],
              val_folds=['fold9'],
              data_root=None,
              confusion_matrix_save_path=None,
              model_path='model.pkl',
              feature_type='all',
              roc_curves_save_path=None
     ):
    train_images, train_data, train_labels, val_images, val_data, val_labels, class_map = prepare_data(train_folds, val_folds, data_root, feature_type=feature_type)

    clf = joblib.load(model_path)

    val_pred = clf.predict(val_data)

    if confusion_matrix_save_path:
        plot_confusion_matrix(val_labels, val_pred, save_path=confusion_matrix_save_path)

    plot_roc_curves(val_labels-1, clf.predict_proba(val_data),
                     class_names=[k for k in class_map.keys()],
                     save_path=roc_curves_save_path)


def main():
    parser = argparse.ArgumentParser(description='Train or test cell classification model')
    
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='Mode: train or test')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to the dataset root directory')
    parser.add_argument('--train_folds', type=str, nargs='+', default=None,
                        help='List of training fold names (e.g., fold0 fold1 fold2)')
    parser.add_argument('--val_folds', type=str, nargs='+', default=None,
                        help='List of validation fold names (e.g., fold3 fold4)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to save/load the model')
    parser.add_argument('--feature_type', type=str, default='all',
                        choices=['traditional_feature', 'deep_feature', 'all'],
                        help='Type of features to use')
    parser.add_argument('--confusion_matrix_path', type=str, default=None,
                        help='Path to save confusion matrix')
    parser.add_argument('--roc_curves_path', type=str, default=None,
                        help='Path to save ROC curves')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        logging.info(f"Training model with feature type: {args.feature_type}")
        train_val(
            train_folds=args.train_folds,
            val_folds=args.val_folds,
            data_root=args.data_root,
            confusion_matrix_save_path=args.confusion_matrix_path,
            model_path=args.model_path,
            feature_type=args.feature_type,
            roc_curves_save_path=args.roc_curves_path
        )
    elif args.mode == 'test':
        logging.info(f"Testing model with feature type: {args.feature_type}")
        test(
            train_folds=args.train_folds,
            val_folds=args.val_folds,
            data_root=args.data_root,
            confusion_matrix_save_path=args.confusion_matrix_path,
            model_path=args.model_path,
            feature_type=args.feature_type,
            roc_curves_save_path=args.roc_curves_path
        )


if __name__ == '__main__':
    main()
