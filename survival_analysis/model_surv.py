import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import optuna
import datetime
import dataclasses
from dataclasses import dataclass
import pickle
import json

from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis

from select_survival_features import FeatureSelector


@dataclass
class ModelResult:
    """Model training result data class, encapsulating all training outputs"""
    # Model and core metrics
    model: Any = None
    train_c_index: float = None
    val_c_index: float = None
    train_score_df: Optional[pd.Series] = None
    val_score_df: Optional[pd.Series] = None
    p_value: float = None
    # Feature related
    selected_features: Optional[List[str]] = None
    feature_importance: Optional[pd.DataFrame] = None
    # Hyperparameters and metadata
    hyperparameters: Optional[Dict[str, Any]] = None
    fold: Optional[int] = None
    timestamp: Optional[datetime.datetime] = None
    shap_values: Optional[shap.Explanation] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result_dict = dataclasses.asdict(self)
        # Handle non-serializable objects
        result_dict['model'] = None  # Model saved separately
        result_dict['train_score_df'] = self.train_score_df.to_dict() if self.train_score_df is not None else None
        result_dict['val_score_df'] = self.val_score_df.to_dict() if self.val_score_df is not None else None
        result_dict['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        result_dict['shap_values'] = None  # SHAP values saved separately
        return result_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelResult':
        """Rebuild object from dictionary"""
        data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
        return cls(**data)

class SurvivalModel:
    """Survival analysis model class, encapsulating model training, evaluation and interpretation functions"""
    def __init__(self, args, logger):
        """Initialize survival model

        Args:
            args: Model configuration parameters
        """
        self.args = args
        self.model = self._initialize_model()
        self.logger = logger

    def _initialize_model(self) -> Any:
        """Initialize model instance"""
        model_map = {
            'rsf': RandomSurvivalForest,
            'cox': CoxnetSurvivalAnalysis,
            'gb': GradientBoostingSurvivalAnalysis,
        }

        if self.args.model_type not in model_map:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")

        model_params = {
            'rsf': {
                'n_estimators': self.args.n_estimators,
                'min_samples_split': self.args.min_samples_split,
                'min_samples_leaf': self.args.min_samples_leaf,
                'max_depth': self.args.max_depth,
                'random_state': self.args.random_state,
                'n_jobs': self.args.n_jobs,
            },
            'cox': {
                'l1_ratio': self.args.l1_ratio,
                'alphas': [self.args.penalizer],
            },
            'gb': {
                'learning_rate': self.args.learning_rate,
                'n_estimators': self.args.n_estimators,
                'min_samples_split': self.args.min_samples_split,
                'min_samples_leaf': self.args.min_samples_leaf,
                'max_depth': self.args.max_depth,
                'random_state': self.args.random_state,
            },
        }

        return model_map[self.args.model_type](**model_params[self.args.model_type])
    
    def fit(self, X: pd.DataFrame, y: pd.Series, fold: int) -> None:
        """Train model

        Args:
            X: Feature data
            y: Target data
            fold: Fold index
        """
        self.model.fit(X, y)
    
    def score(self, X: pd.DataFrame, y: pd.Series, fold: int) -> float:
        """Calculate model C-index

        Args:
            X: Feature data
            y: Target data
            fold: Fold index

        Returns:
            C-index value
        """
        return self.model.score(X, y)
    
    def predict_risk(self, X: pd.DataFrame, fold: int) -> np.ndarray:
        """Predict risk scores

        Args:
            X: Feature data
            fold: Fold index

        Returns:
            Risk score array
        """
        return self.model.predict(X)

    def plot_shap(self, X: pd.DataFrame, save_path: str, fold: int) -> None:
        """Plot SHAP value summary chart
    
        Args:
            X: Feature data
            save_path: Image save path
            fold: Fold index
        """
        def model_wrapper(X_data):
            return self.model.predict(X_data)
    
        # Dynamically calculate required max_evals value
        num_features = len(X.columns)
        required_max_evals = 2 * num_features + 1
        
        # Set appropriate max_evals value, can adjust upper limit as needed
        max_evals = min(required_max_evals, 10000)  # Set upper limit to avoid excessive computation time
        
        explainer = shap.Explainer(model_wrapper, X, feature_names=X.columns)
        shap_values = explainer(X, max_evals=max_evals)
        shap.summary_plot(shap_values, X)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        return shap_values

    def calculate_feature_importance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature importance

        Args:
            X_train: Training set feature data

        Returns:
            DataFrame containing feature names and importance scores
        """
        self.logger.info("Calculating feature importance...")

        try:
            # Try to get feature importance directly
            importances = self.model.feature_importances_
        except AttributeError:
            # If model doesn't have feature_importances_ attribute, use permutation importance
            self.logger.info("Model doesn't support direct feature importance, using permutation importance...")
            result = permutation_importance(
                self.model, X_train, self.y_train, 
                n_repeats=10, random_state=self.args.random_state, n_jobs=self.args.n_jobs
            )
            importances = result.importances_mean

        return pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

    def assign_risk_groups(self, risk_scores: np.ndarray) -> np.ndarray:
        """Assign risk groups based on risk scores

        Args:
            risk_scores: Model predicted risk scores

        Returns:
            Risk grouping results (0: low risk, 1: high risk)
        """
        median_score = np.median(risk_scores)
        return (risk_scores > median_score).astype(int)

    def logrank_test(self, time: pd.Series, status: pd.Series, risk_groups: np.ndarray) -> float:
        """Execute logrank test

        Args:
            time: Survival time
            status: Event status
            risk_groups: Risk grouping results

        Returns:
            p-value of logrank test
        """
        group0_time = time[risk_groups == 0]
        group1_time = time[risk_groups == 1]
        group0_status = status[risk_groups == 0]
        group1_status = status[risk_groups == 1]

        result = logrank_test(group0_time, group1_time, group0_status, group1_status)
        return result.p_value

    def test(self, fold_data: Dict, selected_features: List[str]) -> float:
        train_risk_scores = self.predict_risk(fold_data['X_train'][selected_features], fold_data['fold'])
        val_risk_scores = self.predict_risk(fold_data['X_val'][selected_features], fold_data['fold'])

        # Assign risk groups
        fold_data['train_df']['risk_group'] = self.assign_risk_groups(train_risk_scores)
        fold_data['val_df']['risk_group'] = self.assign_risk_groups(val_risk_scores)

        # fold_data['train_df']['score'] = train_risk_scores
        # fold_data['val_df']['score'] = val_risk_scores

        # Execute logrank test
        p_value = self.logrank_test(
            fold_data['val_df']['time'],
            fold_data['val_df']['status'],
            fold_data['val_df']['risk_group']
        )

        # Calculate feature importance
        feature_importance = self.calculate_feature_importance(fold_data['X_train']) if self.args.return_importance else None

        # Plot SHAP charts
        shap_path = f"{self.args.full_output_dir}/rsf_shap_summary_plot_{fold_data['fold']}.png"
        shap_values = self.plot_shap(fold_data['X_val'][selected_features], shap_path, fold_data['fold']) if self.args.plot_shap else None

        return p_value, feature_importance, shap_values
        


class MultiModalModel(SurvivalModel):
    def __init__(self, args, logger):
        super().__init__(args, logger)

        best_method = json.load(open("best_method.json", "r"))

        self.unimodal_result = {}
        for fold in range(args.n_splits):
            self.unimodal_result[fold] = {}
            for modal, model_type in best_method[args.random_state].items():
                load_path = Path(args.work_dir) / args.output_dir / str(args.random_state) / modal / model_type['feature_selection_method'] / model_type['model_type'] / f"fold_{fold}" /"result.pkl"
                with open(load_path, 'rb') as f:
                    result = pickle.load(f)
                self.unimodal_result[fold][modal] = result

    def get_modal_risk_df(self, X: pd.DataFrame, fold: int) -> pd.DataFrame:
        """获取模态风险数据框

        Args:
            X: 特征数据
            fold: 折叠索引

        Returns:
            包含模态风险数据的数据框
        """
        modal_risk = {}
        for modal, result in self.unimodal_result[fold].items():
            risk = result.model.predict_risk(X[result.selected_features], fold)
            modal_risk[modal] = risk

        # 转换为pd.DataFrame
        modal_risk_df = pd.DataFrame(modal_risk)
        return modal_risk_df

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: int) -> None:
        """训练模型

        Args:
            X: 特征数据
            y: 目标数据
        """
        modal_risk_df = self.get_modal_risk_df(X, fold)
        self.model.fit(modal_risk_df, y)

    def score(self, X: pd.DataFrame, y: pd.Series, fold: int) -> float:
        modal_risk_df = self.get_modal_risk_df(X, fold)
        return self.model.score(modal_risk_df, y)
    
    def predict_risk(self, X: pd.DataFrame, fold: int) -> np.ndarray:
        """预测风险

        Args:
            X: 特征数据
            fold: 折叠索引

        Returns:
            风险分数
        """
        modal_risk_df = self.get_modal_risk_df(X, fold)
        return self.model.predict(modal_risk_df)

    def plot_shap(self, X: pd.DataFrame, save_path: str, fold: int) -> None:
        """绘制SHAP值摘要图
    
        Args:
            X: 特征数据
            save_path: 图像保存路径
            fold: 折叠索引
        """
        def model_wrapper(X_data):
            modal_risk_df = self.get_modal_risk_df(X_data, fold)
            return self.model.predict(modal_risk_df)
    
        # 动态计算所需的max_evals值
        num_features = len(X.columns)
        required_max_evals = 2 * num_features + 1
        
        # 设置适当的max_evals值，可以根据需要调整上限
        max_evals = min(required_max_evals, 10000)  # 设置上限避免计算时间过长
        
        explainer = shap.Explainer(model_wrapper, X, feature_names=X.columns)
        shap_values = explainer(X, max_evals=max_evals)
        shap.summary_plot(shap_values, X)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        return shap_values


def train_and_evaluate(args, logger, fold_data: Dict, selected_features_file: Path) -> ModelResult:
    """训练并评估模型

    Args:
        fold_data: 包含训练和验证数据的字典

    Returns:
        模型训练结果
    """

    try:
        start_time = datetime.datetime.now()

        feature_selector = FeatureSelector(args, logger)
        selected_features = feature_selector.select_features(fold_data, selected_features_file)

        if args.modal == 'muti_modal':
            model = MultiModalModel(args, logger)
        else:
            model = SurvivalModel(args, logger)
        model.fit(fold_data['X_train'][selected_features], fold_data['y_train'], fold=fold_data['fold'])

        val_c_index = model.score(fold_data['X_val'][selected_features], fold_data['y_val'], fold=fold_data['fold'])
        if args.do_hyper_search:
            return ModelResult(val_c_index=val_c_index)
        else:
            train_c_index = model.score(fold_data['X_train'][selected_features], fold_data['y_train'], fold=fold_data['fold'])
            return ModelResult(
                model=model,
                train_c_index=train_c_index,
                val_c_index=val_c_index,
                selected_features=selected_features,
                fold=fold_data.get('fold', None),
                timestamp=start_time,
            )
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        return ModelResult()

def hyperparameter_search_objective(trial: optuna.Trial, args, logger, fold: int, cv_folds: List[Dict]) -> float:
    """超参数搜索目标函数

    Args:
        trial: Optuna试验对象
        fold: 折数
        cv_folds: 交叉验证数据

    Returns:
        平均验证C-index
    """

    if args.model_type == 'rsf':
        args.n_estimators = trial.suggest_int('n_estimators', 50, 200)
        args.min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        args.min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        args.max_depth = trial.suggest_int('max_depth', 5, 20)
    elif args.model_type == 'cox':
        args.l1_ratio = trial.suggest_float('l1_ratio', 1e-5, 1.0, log=True)
        # args.penalizer = trial.suggest_float('penalizer', 0.0, 1.0)
        if args.modal in ['all', 'path']:
            args.penalizer = trial.suggest_float('penalizer', 1e-1, 1.0)
        elif args.modal in ['multi_modal']:
            args.penalizer = trial.suggest_float('penalizer', 1e-1, 1.0, log=True)
        else:
            args.penalizer = trial.suggest_float('penalizer', 1e-5, 1.0, log=True)
    elif args.model_type == 'gb':
        args.learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        args.n_estimators = trial.suggest_int('n_estimators', 50, 200)
        args.min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        args.min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        args.max_depth = trial.suggest_int('max_depth', 3, 15)

    # 如果不是使用所有特征，则搜索最佳特征数量
    if args.feature_selection_method != 'all':
        max_features = max([len(fold_data['X_train'].columns) for fold_data in cv_folds])
        args.k_features = trial.suggest_int('k_features', 2, max_features)

    val_c_indices = []
    for fold_data in cv_folds:
        selected_features_file = Path(args.full_output_dir).parent / 'hyper_search_selected_features' / f"f{fold}_cv{fold_data['fold']}.json"

        val_c_index = train_and_evaluate(args, logger, fold_data, selected_features_file).val_c_index
        if val_c_index is None:
            return -np.inf
        val_c_indices.append(val_c_index)

    return np.mean(val_c_indices)


