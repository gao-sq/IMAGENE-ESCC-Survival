import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from feature_engine.selection import MRMR
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    VarianceThreshold,
)

from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
import copy


class FeatureSelectionMethod(str, Enum):
    UNIVARIATE = "univariate"
    AUC = "auc"
    MRMR = "mrmr"
    LASSO_COX = "lasso_cox"
    ALL = "all"
    FEATURE_IMPORTANCE = "feature_importance"


class FeatureSelector:
    """Survival analysis feature selector, encapsulating multiple feature selection methods"""

    def __init__(self, args: argparse.Namespace, logger):
        self.args = args
        self.logger = logger
        self.selected_features: Optional[List[str]] = None
        self._selection_methods = {
            FeatureSelectionMethod.UNIVARIATE: self._select_univariate_features,
            FeatureSelectionMethod.AUC: self._select_auc_features,
            FeatureSelectionMethod.MRMR: self._select_mrmr_features,
            FeatureSelectionMethod.LASSO_COX: self._select_lasso_cox_features,
            FeatureSelectionMethod.ALL: self._select_all_features,
            # FeatureSelectionMethod.FEATURE_IMPORTANCE: self._select_from_feature_importance
        }

    def select_features(self, fold_data: Dict, sorted_features_file: Path) -> List[str]:
        """Execute feature selection and return results"""

        # If re-selection is not needed and file exists, load directly
        if not self.args.re_select_features and sorted_features_file.exists():
            with open(sorted_features_file, "r") as f:
                sorted_features = json.load(f)
        else:
            # Execute feature selection
            selection_method = self._selection_methods.get(
                self.args.feature_selection_method
            )
            if not selection_method:
                self.logger.warning(
                    f"Unknown feature selection method: {self.args.feature_selection_method}"
                )
                sorted_features = []
            else:
                sorted_features = selection_method(fold_data)
            # Save selection results
            self._save_selected_features(sorted_features_file, sorted_features)
        self.selected_features = (
            sorted_features[: self.args.k_features]
            if self.args.feature_selection_method != FeatureSelectionMethod.ALL
            else sorted_features
        )
        if not self.args.do_hyper_search:
            self.logger.info(f"Selected {len(self.selected_features)} features using {self.args.feature_selection_method} method")
        return self.selected_features

    def _save_selected_features(
        self, file_path: Path, sorted_features: List[str]
    ) -> None:
        """Save selected features to file"""
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w") as f:
            json.dump(sorted_features, f)

    def _select_univariate_features(self, fold_data: Dict) -> List[str]:
        """Univariate Cox regression feature selection"""
        X_train = fold_data["X_train"]
        train_df = fold_data["train_df"]
        significant_features = {}

        for feature in X_train.columns:
            try:
                # Create univariate Cox model
                cph = CoxPHFitter()
                cph.fit(
                    pd.DataFrame(
                        {
                            "time": train_df["time"],
                            "status": train_df["status"],
                            "feature": X_train[feature],
                        }
                    ),
                    "time",
                    "status",
                )
                p = cph.summary.loc[["feature"], "p"].values[0]
                if p < 0.05:  # Set significance level
                    significant_features[feature] = p
            except Exception as e:
                self.logger.warning(f"Feature {feature} analysis failed: {str(e)}")

        self.logger.info(f"Univariate Cox analysis found {len(significant_features)} significant features")
        sorted_features = sorted(significant_features.items(), key=lambda x: x[1])
        return [f for f, p in sorted_features]

    def _select_auc_features(self, fold_data: Dict) -> List[str]:
        """AUC-based feature selection"""
        X_train = fold_data["X_train"]
        train_df = fold_data["train_df"]
        y_train = fold_data["y_train"]
        auc_scores = {}

        for feature in X_train.columns:
            try:
                # Determine valid time points
                times = np.array(
                    [
                        t
                        for t in [12, 24, 36]
                        if t <= train_df[train_df["status"] == True]["time"].max()
                    ]
                )
                if len(times) == 0:
                    continue

                # Calculate concordance index
                ret = concordance_index_censored(
                    train_df["status"], train_df["time"], X_train[feature]
                )

                # Decide whether to invert feature based on CI value
                if ret[0] > 0.6:
                    _, mean_auc = cumulative_dynamic_auc(
                        y_train, y_train, X_train[feature], times
                    )
                    auc_scores[feature] = mean_auc
                elif ret[0] < 0.4:  # Take opposite of feature
                    _, mean_auc = cumulative_dynamic_auc(
                        y_train, y_train, -X_train[feature], times
                    )
                    auc_scores[feature] = mean_auc
            except Exception as e:
                self.logger.warning(f"Feature {feature} analysis failed: {str(e)}")

        self.logger.info(f"AUC analysis found {len(auc_scores)} qualified features")
        sorted_features = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in sorted_features]

    def _select_mrmr_features(self, fold_data: Dict) -> List[str]:
        """MRMR feature selection"""
        X_train = fold_data["X_train"]
        train_df = fold_data["train_df"]

        # Create binary classification target variable
        median_time = train_df[train_df["status"] == True]["time"].median()
        mask = (train_df["time"] >= median_time) | (train_df["status"] == 1)
        X_train_subset = X_train[mask]
        target = np.where(
            (train_df[mask]["time"] <= median_time) & (train_df[mask]["status"] == 1),
            1,
            0,
        )

        # Check if data is empty
        if X_train_subset.empty or len(target) == 0:
            self.logger.warning(
                "Input data or target data for MRMR feature selection is empty, skipping MRMR feature selection"
            )
            return []

        k = min(self.args.k_features, len(X_train_subset.columns))
        mrmr = MRMR(
            max_features=k, random_state=self.args.random_state, method="MID", n_jobs=-1
        )
        mrmr.fit(X_train_subset, target)
        return mrmr.get_feature_names_out()

    def _select_lasso_cox_features(self, fold_data: Dict) -> List[str]:
        """Lasso Cox feature selection"""
        X_train = fold_data["X_train"]
        y_train = fold_data["y_train"]

        coxnet = CoxnetSurvivalAnalysis(
            l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=100
        )
        coxnet.fit(X_train, y_train)

        # Count non-zero coefficient occurrences for each feature
        nonzero_counts = (coxnet.coef_ != 0).sum(axis=1)
        sorted_features = sorted(
            zip(X_train.columns, nonzero_counts), key=lambda x: x[1], reverse=True
        )

        return [f for f, c in sorted_features if c > 0]

    def _select_all_features(self, fold_data: Dict) -> List[str]:
        """Select all features"""
        return fold_data["X_train"].columns.tolist()
