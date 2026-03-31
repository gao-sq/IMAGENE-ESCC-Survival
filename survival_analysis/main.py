import sys
import logging
from pathlib import Path
import argparse
from multiprocessing import Pool
from typing import List, Tuple, Dict, Optional
from enum import Enum
import random
import numpy as np
import pandas as pd

sys.path.append('path/to/project')
from train import CrossValidationManager


class ModalType(str, Enum):
    CLINICAL = 'clinical'
    WES = 'wes'
    TCR = 'tcr'
    PATH = 'path'
    ALL = 'all'
    ALL_RISK = 'all_risk'


class FeatureSelectionMethod(str, Enum):
    ALL = 'all'
    UNIVARIATE = 'univariate'
    AUC = 'auc'
    LASSO_COX = 'lasso_cox'


class ModelType(str, Enum):
    COX = 'cox'
    RSF = 'rsf'
    GB = 'gb'


def get_full_output_dir(args: argparse.Namespace) -> Path:
    """Get complete output directory path"""
    return Path(args.work_dir) / args.output_dir / str(args.random_state) / args.modal / args.feature_selection_method / args.model_type


class SurvivalAnalysisPipeline:
    """Survival analysis experiment pipeline"""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.setup_random_seed()
        self._setup_output_directory()
        self._setup_logger()

    def setup_random_seed(self) -> None:
        """Set global random seed to ensure reproducibility"""
        random.seed(self.args.random_state)
        np.random.seed(self.args.random_state)

    def _setup_output_directory(self) -> None:
        """Setup output directory"""
        output_dir = get_full_output_dir(self.args)
        output_dir.mkdir(exist_ok=True, parents=True)
        self.args.full_output_dir = str(output_dir)

    def _setup_logger(self) -> None:
        """Configure logger"""
        log_path = Path(self.args.full_output_dir) / self.args.log_file
        if log_path.exists():
            log_path.unlink()

        logger = logging.getLogger()
        # Remove existing handlers to prevent duplicate output
        if logger.hasHandlers():
            logger.handlers = []
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)
        self.logger = logger

    def run(self) -> Tuple[Dict, Optional[pd.DataFrame]]:
        """Run survival analysis experiment pipeline"""
        self.logger.info("========== Starting survival analysis model construction ==========")
        self.logger.info(f"Experiment configuration: {self.args}")
        if self.args.do_train:
            CrossValidationManager(self.args, self.logger).run_cross_validation()
        if self.args.do_test:
            CrossValidationManager(self.args, self.logger).run_test()

        self.logger.info("========== Survival analysis model construction completed ==========")


def run_experiment(args: argparse.Namespace) -> None:
    """Run single experiment"""
    pipeline = SurvivalAnalysisPipeline(args)
    pipeline.run()


def run_parallel_experiments(base_args: argparse.Namespace) -> None:
    """Run multiple experiment combinations in parallel"""
    # Generate all parameter combinations
    param_combinations = []
    for random_state in [42, 123, 456, 789, 101112]:
        for modal in ModalType:
            for feature_method in FeatureSelectionMethod:
                for model_type in ModelType:
                    # Create new parameter namespace
                    new_args = argparse.Namespace(**vars(base_args))
                    new_args.random_state = random_state
                    new_args.modal = modal.value
                    new_args.feature_selection_method = feature_method.value
                    new_args.model_type = model_type.value
                    run_experiment(new_args)
                    param_combinations.append(new_args)

    # Use multiprocessing to execute tasks
    # with Pool() as pool:
    #     pool.map(run_experiment, param_combinations)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Survival analysis model construction")

    # Directory configuration
    dir_group = parser.add_argument_group('Directory configuration')
    dir_group.add_argument('--work_dir', type=str, 
                          default="path/to/project/survival_analysis", 
                          help="Working directory")
    dir_group.add_argument('--log_file', type=str, default="analysis.log", help="Log file")
    dir_group.add_argument('--output_dir', type=str, default="outputs", help="Output directory")

    # Data and feature configuration
    data_group = parser.add_argument_group('Data and feature configuration')
    data_group.add_argument('--modal', type=str, default="path", 
                           help="Data file type", 
                           choices=[m.value for m in ModalType])
    data_group.add_argument('--feature_selection_method', type=str, default="all", 
                           help="Feature selection method", 
                           choices=[f.value for f in FeatureSelectionMethod])
    data_group.add_argument('--k_features', type=int, default=100, help="Number of features to use")

    # Model parameters
    model_group = parser.add_argument_group('Model parameters')
    model_group.add_argument('--model_type', type=str, default="cox", 
                            help="Model type", 
                            choices=[mt.value for mt in ModelType])
    model_group.add_argument('--l1_ratio', type=float, default=0.5, help="L1 regularization ratio")
    model_group.add_argument('--penalizer', type=float, default=0.01, help="Penalty weight, reduce default value to avoid over-regularization")
    model_group.add_argument('--n_estimators', type=int, default=55, help="Number of trees")
    model_group.add_argument('--min_samples_split', type=int, default=4, help="Minimum number of samples required to split")
    model_group.add_argument('--min_samples_leaf', type=int, default=10, help="Minimum number of samples in leaf")
    model_group.add_argument('--max_depth', type=int, default=10, help="Maximum depth")
    model_group.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate")

    # Process control
    control_group = parser.add_argument_group('Process control')
    control_group.add_argument('--re_load_data', action='store_true', default=True, help="Whether to reload data")
    control_group.add_argument('--n_splits', type=int, default=3, help="Number of cross-validation folds")

    control_group.add_argument('--do_train', action='store_true', default=False, help="Whether to perform model training")
    control_group.add_argument('--do_test', action='store_true', default=True, help="Whether to perform model testing")

    control_group.add_argument('--do_hyper_search', action='store_true', default=True, help="Whether to perform hyperparameter search")
    control_group.add_argument('--re_hyper_search', action='store_true', default=True, help="Whether to re-perform hyperparameter search")
    control_group.add_argument('--hyper_search_n_splits', type=int, default=3, help="Number of cross-validation folds for hyperparameter search")
    control_group.add_argument('--n_trials', type=int, default=30, help="Number of trials for hyperparameter search")

    control_group.add_argument('--re_select_features', action='store_true', default=False, help="Whether to re-perform feature selection")
    control_group.add_argument('--plot_shap', action='store_true', default=True, help="Whether to plot SHAP value charts")
    control_group.add_argument('--return_importance', action='store_true', default=False, help="Whether to return feature importance")
    
    control_group.add_argument('--random_state', type=int, default=42, help="Random seed")
    control_group.add_argument('--n_jobs', type=int, default=-1, help="Specify number of threads to run in parallel")

    control_group.add_argument('--parallel', action='store_true', default=False, help="Whether to run all parameter combinations in parallel")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.parallel:
        run_parallel_experiments(args)
    else:
        run_experiment(args)