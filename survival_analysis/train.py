import argparse
import sys
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import json
import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
import optuna
from optuna.samplers import TPESampler
import shap
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
import seaborn as sns

sys.path.append('path/to/project')
# from survival_analysis.mode_lifelines import train_val_cox, lifelines_objective
from model_surv import hyperparameter_search_objective, ModelResult, train_and_evaluate
from load_data import load_data, filter_data, preprocess_data
from visualize import plot_km_curve


class CrossValidationManager:
    """Cross-validation manager, encapsulating the entire cross-validation process"""
    def __init__(self, args: argparse.Namespace, logger):
        """"Initialize cross-validation manager

        Args:
            args: Command line arguments
        """
        self.args = args
        self.logger = logger
        self.load_cv_folds()

    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters

        Returns:
            Model parameter dictionary
        """
        model_params = {
            'rsf': {
                "n_estimators": self.args.n_estimators,
                "min_samples_split": self.args.min_samples_split,
                "min_samples_leaf": self.args.min_samples_leaf,
                "max_depth": self.args.max_depth,
                "random_state": self.args.random_state,
            },
            'cox': {
                'l1_ratio': self.args.l1_ratio,
            },
            'cox_lifelines': {
                'l1_ratio': self.args.l1_ratio,
                'penalizer': self.args.penalizer
            },
            'gb': {
                "n_estimators": self.args.n_estimators,
                "learning_rate": self.args.learning_rate,
                'min_samples_split': self.args.min_samples_split,
                'min_samples_leaf': self.args.min_samples_leaf,
                "max_depth": self.args.max_depth,
                "random_state": self.args.random_state,
            }
        }
        return model_params.get(self.args.model_type, {})

    def _get_data_subset(self, data: pd.DataFrame, indices: np.ndarray) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Get data subset

        Args:
            data: Complete dataset
            indices: Data indices

        Returns:
            Feature data, time data, and status data
        """
        X = data[self.feature_cols].iloc[indices].reset_index(drop=True)
        time = data['time'].iloc[indices].reset_index(drop=True)
        status = data['status'].iloc[indices].reset_index(drop=True)
        subset_df = data.iloc[indices].reset_index(drop=True)
        return X, time, status, subset_df

    def prepare_cv_folds(self, data: pd.DataFrame, n_splits: int = 3) -> None:
        """Prepare cross-validation fold data

        Args:
            data: Complete dataset
        """
        self.logger.info(f"Using {n_splits}-fold cross-validation to split data...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.args.random_state)

        cv_folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data[self.feature_cols], data['status'])):
            X_train, time_train, status_train, train_df = self._get_data_subset(data, train_idx)
            X_val, time_val, status_val, val_df = self._get_data_subset(data, val_idx)

            y_train = np.array([(s, t) for s, t in zip(status_train, time_train)],
                              dtype=[('status', bool), ('time', float)])
            y_val = np.array([(s, t) for s, t in zip(status_val, time_val)],
                             dtype=[('status', bool), ('time', float)])
            fold_data = {
                'fold': fold_idx,
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'train_df': train_df,
                'val_df': val_df
            }
            cv_folds.append(fold_data)
            self.logger.info(f"Fold {fold_idx + 1} - Training set: {X_train.shape}, Validation set: {X_val.shape}")
            self.logger.info(f"Event distribution - Training set: {sum(status_train)}/{len(status_train)}, "
                          f"Validation set: {sum(status_val)}/{len(status_val)}")
        return cv_folds

    def encode_fold_data(self, fold_data: Dict) -> Dict:
        """Encode fold data

        Args:
            fold_data: Fold data

        Returns:
            Encoded fold data
        """

        df = pd.concat([fold_data['train_df'], fold_data['val_df']], axis=0)[self.feature_cols]

        _, fold_data['X_train'] = preprocess_data(df, fold_data['X_train'])
        _, fold_data['X_val'] = preprocess_data(df, fold_data['X_val'])
        return fold_data

    def load_cv_folds(self) -> None:
        """Load cross-validation fold data"""        
        cv_folds_path = Path(self.args.output_dir) / str(self.args.random_state) / self.args.modal / "cv_folds.pkl"
        if not cv_folds_path.exists() or self.args.re_load_data:

            data, self.feature_cols = load_data(modal=self.args.modal)
            data = filter_data(data, self.feature_cols)
            self.cv_folds = self.prepare_cv_folds(data, self.args.n_splits)
            
            # Save cv_folds
            with open(cv_folds_path, 'wb') as f:
                pickle.dump(self.cv_folds, f)
        else:
            # Load cv_folds
            self.logger.info(f"Loading cv_folds file: {cv_folds_path}")
            with open(cv_folds_path, 'rb') as f:
                self.cv_folds = pickle.load(f)
                self.feature_cols = self.cv_folds[0]['X_train'].columns

    def search_hyperparameters_on_fold(self, fold: int, cv_folds: List[Dict]) -> Dict:
        """Hyperparameter search

        Args:
            fold: Fold number
            cv_folds: Cross-validation fold data

        Returns:
            Best hyperparameter dictionary
        """
        try:
            self.logger.info("Starting hyperparameter search...")
            # objective = lifelines_objective if self.args.model_type == 'cox_lifelines' else hyperparameter_search_objective
            objective = hyperparameter_search_objective
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.args.random_state))
            study.optimize(lambda trial: objective(trial, deepcopy(self.args), self.logger, fold, cv_folds), n_trials=self.args.n_trials, 
                        #    n_jobs=self.args.n_jobs
            )
            best_params = study.best_params
            if study.best_value > -np.inf:
                self.logger.info(f"Best hyperparameters: {best_params}, Validation C-index: {study.best_value:.4f}")
                return best_params
            else:
                self.logger.warning("Hyperparameter search found no valid results")
                return {}
        except Exception as e:
            self.logger.warning(f"Hyperparameter search failed: {str(e)}")
            return {}

    def search_hyperparameters(self,) -> List[Dict]:
        current_params_list = []
        for fold_data in self.cv_folds:
            best_params_path = Path(self.args.full_output_dir) / f"fold_{fold_data['fold']}" / f"best_params.json"
            if not self.args.re_hyper_search:
                if best_params_path.exists():
                    with open(best_params_path, 'r') as f:
                        best_params = json.load(f)
                else:
                    self.logger.info(f"Best hyperparameter file not found: {best_params_path}")
                    continue
            else:
                train_cv_folds_path = Path(self.args.output_dir) / str(self.args.random_state) / self.args.modal / f"fold_{fold_data['fold']}_train_cv_folds.pkl"
                if train_cv_folds_path.exists(): 
                    self.logger.info(f"Loading train_cv_folds file: {train_cv_folds_path}")
                    with open(train_cv_folds_path, 'rb') as f:
                        train_cv_folds = pickle.load(f)
                else:
                    train_cv_folds = self.prepare_cv_folds(fold_data['train_df'], self.args.hyper_search_n_splits)
                    train_cv_folds = [
                        self.encode_fold_data(fold_data)
                        for fold_data in train_cv_folds
                    ]
                    with open(train_cv_folds_path, 'wb') as f:
                        pickle.dump(train_cv_folds, f)
                best_params = self.search_hyperparameters_on_fold(fold_data['fold'], train_cv_folds)
                if best_params:
                    best_params_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(best_params_path, 'w') as f:
                        json.dump(best_params, f)

            current_params_list.append(best_params)
        self.args.do_hyper_search = False
        return current_params_list
    
    def _get_selected_features_filepath(self, fold: int) -> Path:
        """Get feature selection result file path"""
        output_dir = Path(self.args.full_output_dir).parent
        return output_dir / 'selected_features' / f"f{fold}.json"

    def run_cross_validation(self):
        """Run cross-validation
        Returns:
            Cross-validation results and feature importance
        """
        self.logger.info("Starting hyperparameter search...")
        current_params_list = self.search_hyperparameters() if self.args.do_hyper_search else []
        self.logger.info("Starting cross-validation training and evaluation...")

        cv_results = []
        for fold_data in self.cv_folds:
            if len(current_params_list) == len(self.cv_folds):
                self.args.__dict__.update(current_params_list[fold_data['fold']])
            else:
                self.logger.info(f"Fold {fold_data['fold']} did not find best hyperparameters, using default parameters")

            fold_data = self.encode_fold_data(fold_data)
            result = train_and_evaluate(self.args, self.logger, fold_data, self._get_selected_features_filepath(fold_data['fold']))
            if result.val_c_index is not None:
                cv_results.append(result)
            # Print training and validation set C-index
            if result.train_c_index is not None and result.val_c_index is not None:
                self.logger.info(f"Fold {fold_data['fold']} results: {self.args.model_type} model C-index: train={result.train_c_index:.4f}, val={result.val_c_index:.4f}")
            else:
                self.logger.warning(f"Fold {fold_data['fold']} results: Training or validation set C-index calculation failed")

        self.process_train_result(cv_results)

    def run_test(self):
        """Run test"""
        self.logger.info("Starting test...")
        try:
            if Path(self.args.full_output_dir, f'confusion_matrix_s{self.args.random_state}.png').exists():
                self.logger.info(f"confusion_matrix_s{self.args.random_state}.png already exists, skipping test")
                return
            cv_results = self.load_result()
            for result, fold_data in zip(cv_results, self.cv_folds):
                fold_data = self.encode_fold_data(fold_data)
                result.model.args = self.args
                p_value, feature_importance, shap_values = result.model.test(fold_data, result.selected_features)
                result.p_value = p_value
                result.feature_importance = feature_importance
                result.shap_values = shap_values
            self.process_test_result(cv_results)
        except Exception as e:
            self.logger.error(f"测试运行失败: {str(e)}", exc_info=True)

    def process_test_result(self, cv_results: List[ModelResult]) -> None:
        """Process cross-validation results and generate report

        This function is responsible for summarizing cross-validation results, calculating statistical metrics, 
        saving result files, plotting visualization charts, and processing feature importance data.

        Args:
            cv_results: List of cross-validation results, each element is a ModelResult object
        """
        if len(cv_results) == 0:
            self.logger.warning("All fold model training failed, unable to generate results")
            return

        output_dir = Path(self.args.full_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._plot_km_curve(output_dir)
            self._plot_confusion_matrix(output_dir)
            # 6. Process feature importance
            # if self.args.return_importance and feature_importance_list:
            #     importance_df = pd.concat(feature_importance_list, ignore_index=True)
            #     importance_df = importance_df.fillna(0)
            #     importance_df = importance_df.groupby('feature').agg({'importance': 'mean'}).reset_index()
            #     importance_df = importance_df.sort_values('importance', ascending=False)
            #     importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

            self._plot_shap_plots(cv_results, output_dir)

        except Exception as e:
            self.logger.error(f"Error occurred while processing results: {str(e)}", exc_info=True)
            raise  # Re-raise exception for caller to handle

    def process_train_result(self, cv_results: List[ModelResult]) -> None:
        """Process cross-validation results and generate report

        This function is responsible for summarizing cross-validation results, calculating statistical metrics, 
        saving result files, plotting visualization charts, and processing feature importance data.

        Args:
            cv_results: List of cross-validation results, each element is a ModelResult object
            feature_importance_list: List of feature importance data
            cv_folds: Cross-validation fold data, used for plotting KM curves
        """
        if len(cv_results) == 0:
            self.logger.warning("All fold model training failed, unable to generate results")
            return

        output_dir = Path(self.args.full_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.save_result(cv_results)
            summary_stats = self._calculate_summary_stats(cv_results)
            self._log_summary_results(summary_stats)
            result_df = self._create_result_dataframe(cv_results, summary_stats)
            result_df.to_csv(output_dir / "cv_results.csv", index=False)

            # 6. Process feature importance
            # if self.args.return_importance and feature_importance_list:
            #     importance_df = pd.concat(feature_importance_list, ignore_index=True)
            #     importance_df = importance_df.fillna(0)
            #     importance_df = importance_df.groupby('feature').agg({'importance': 'mean'}).reset_index()
            #     importance_df = importance_df.sort_values('importance', ascending=False)
            #     importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

            # 7. Plot SHAP charts
            # self._plot_shap_plots(self.cv_folds, cv_results, output_dir)

        except Exception as e:
            self.logger.error(f"Error occurred while processing results: {str(e)}", exc_info=True)
            raise  # Re-raise exception for caller to handle

    def _save_full_results(self, cv_results: List[ModelResult], output_dir: Path) -> None:
        """Save complete cross-validation results to JSON file"""
        try:
            with open(output_dir / 'cv_results_full.json', 'w') as f:
                json.dump([r.to_dict() for r in cv_results], f, indent=2, ensure_ascii=False)
            self.logger.info(f"Complete results saved to {output_dir / 'cv_results_full.json'}")
        except IOError as e:
            self.logger.error(f"Failed to save complete results: {str(e)}")
            raise

    def _calculate_summary_stats(self, cv_results: List[ModelResult]) -> Dict[str, float]:
        """Calculate summary statistics for cross-validation results"""
        val_c_indices = [result.val_c_index for result in cv_results]
        return {
            'val_c_index_mean': np.mean(val_c_indices),
            'val_c_index_std': np.std(val_c_indices),
            'train_c_index_mean': np.mean([result.train_c_index for result in cv_results]),
        }

    def _log_summary_results(self, summary_stats: Dict[str, float]) -> None:
        """Log cross-validation summary results"""
        self.logger.info("======= Cross-validation Summary =======")
        self.logger.info(f"{self.args.model_type} model performance metrics:")
        self.logger.info(f"- Training C-index mean: {summary_stats['train_c_index_mean']:.4f}")
        self.logger.info(f"- Validation C-index mean: {summary_stats['val_c_index_mean']:.4f} (±{summary_stats['val_c_index_std']:.4f})")

    def _create_result_dataframe(self, cv_results: List[ModelResult], summary_stats: Dict[str, float]) -> pd.DataFrame:
        """Create DataFrame from cross-validation results"""
        data = {
            'random_state': [self.args.random_state] * len(cv_results),
            'fold': [r.fold for r in cv_results],
            'timestamp': [r.timestamp for r in cv_results],
            'train_c_index': [r.train_c_index for r in cv_results],
            'val_c_index': [r.val_c_index for r in cv_results],
            'val_c_index_mean': summary_stats['val_c_index_mean'],
            'val_c_index_std': summary_stats['val_c_index_std']
        }     
        return pd.DataFrame(data)

    def _plot_km_curve(self, output_dir: Path) -> None:
        """Plot Kaplan-Meier survival curve"""
        try:
            val_dfs = [fold_data['val_df'] for fold_data in self.cv_folds]
            combined_df = pd.concat(val_dfs, ignore_index=True)

            combined_df.to_csv(output_dir / f'km_curve_s{self.args.random_state}.csv', index=False)

            if 'risk_group' not in combined_df.columns:
                self.logger.warning("'risk_group' column not found in data, unable to plot KM curve")
                return

            km_plot_path = output_dir / f'km_curve_s{self.args.random_state}.png'
            plot_km_curve(
                combined_df[combined_df['risk_group'] == 0]['time'],
                combined_df[combined_df['risk_group'] == 1]['time'],
                combined_df[combined_df['risk_group'] == 0]['status'],
                combined_df[combined_df['risk_group'] == 1]['status'],
                'low risk group', 'high risk group', str(km_plot_path)
            )
            self.logger.info(f"KM curve saved to {km_plot_path}")
        except Exception as e:
            self.logger.error(f"Failed to plot KM curve: {str(e)}", exc_info=True)

    def _plot_confusion_matrix(self, output_dir: Path) -> None:
        """Plot confusion matrix"""
        try:
            val_dfs = []
            for fold_data in self.cv_folds:
                val_df = fold_data['val_df'][['time', 'status', 'risk_group']]
                for col in fold_data['X_train'].columns:
                    try:
                        model = CoxPHSurvivalAnalysis()
                        model.fit(fold_data['X_train'][[col]], fold_data['y_train'])
                        score = model.predict(fold_data['X_val'][[col]])
                    except Exception as e:
                        self.logger.error(f"Model training failed: {str(e)}")
                        continue
                    val_df[col] = (score > np.median(score)).astype(int)
                    
                val_dfs.append(val_df)

            # Merge results from all folds, only keep columns present in every fold
            combined_df = pd.concat(val_dfs)
            combined_df = combined_df.dropna(axis=1)

            combined_df.to_csv(output_dir / f'confusion_matrix_s{self.args.random_state}.csv', index=False)

            if 'risk_group' not in combined_df.columns:
                self.logger.warning("'risk_group' column not found in data, unable to plot confusion matrix")
                return

            p_value_dict = {}
            for col in combined_df.columns:
                if col not in ['time', 'status', 'risk_group']:
                    p_value = chi2_contingency(pd.crosstab(combined_df[col], combined_df['risk_group']))[1]
                    p_value_dict[col] = p_value

            p_value_dict = {k: v for k, v in sorted(p_value_dict.items(), key=lambda item: item[1])}
            # Take top 9 features with smallest p-values
            p_value_dict = {k: v for k, v in p_value_dict.items() if k in list(p_value_dict.keys())[:9]}
            
            confusion_matrix_path = output_dir / f'confusion_matrix_s{self.args.random_state}.png'
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            for i, ax in enumerate(axes.flat):
                sns.heatmap(confusion_matrix(combined_df[list(p_value_dict.keys())[i]], combined_df['risk_group']), annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'{list(p_value_dict.keys())[i]}')
                ax.set_xlabel('feature_group')
                ax.set_ylabel('risk_group')
            plt.tight_layout()
            plt.savefig(confusion_matrix_path)
            plt.close()

            self.logger.info(f"Confusion matrix saved to {confusion_matrix_path}")
        except Exception as e:
            self.logger.error(f"Failed to plot confusion matrix: {str(e)}", exc_info=True)

    def _plot_shap_plots(self, cv_results: List[ModelResult], output_dir: Path) -> None:
        """Plot SHAP charts"""
        try:
            # Get all features
            all_features = set()
            shap_data_list = []
            
            for result in cv_results:
                if result.shap_values is not None:
                    all_features.update(result.shap_values.feature_names)
                    shap_data_list.append({
                        'shap_values': result.shap_values,
                        'selected_features': result.selected_features
                    })
            
            all_features = sorted(list(all_features))
            
            # Create empty SHAP value matrix
            shap_values_combined = []
            
            for data in shap_data_list:
                # Create full feature SHAP value matrix for current fold
                shap_df = pd.DataFrame(
                    data['shap_values'].values,
                    columns=data['shap_values'].feature_names
                )
                # Reindex to full feature set
                shap_df = shap_df.reindex(columns=all_features, fill_value=0)
                shap_values_combined.append(shap_df.values)
            
            shap_values_all = np.vstack(shap_values_combined)
            
            # Merge validation set data
            X_val = pd.concat([fold_data['X_val'] for fold_data in self.cv_folds], ignore_index=True)
            
            # Ensure X_val feature order matches SHAP values
            X_val = X_val.reindex(columns=all_features, fill_value=0)
            
            # Plot SHAP summary chart
            shap.summary_plot(shap_values_all, X_val, show=False)

            shap_plot_path = output_dir / f'shap_plot_s{self.args.random_state}.png'
            plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Failed to plot SHAP charts: {str(e)}", exc_info=True)

    def save_result(self, cv_results: List[ModelResult]) -> None:
        """Save cross-validation results"""
        for result in cv_results:
            save_path = Path(self.args.full_output_dir) / f"fold_{result.fold}" / f"result.pkl"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(result, f)

    def load_result(self) -> List[ModelResult]:
        """Load cross-validation results"""
        cv_results = []
        for fold_data in self.cv_folds:
            load_path = Path(self.args.full_output_dir) / f"fold_{fold_data['fold']}" / f"result.pkl"
            with open(load_path, 'rb') as f:
                result = pickle.load(f)
                cv_results.append(result)
        return cv_results
